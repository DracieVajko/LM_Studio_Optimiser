"""Context Explorer: geometric bracketing, speed thresholds, fill ratios, retrieval.

Implements the master spec (LM_STUDIO_OPTIMISER_MASTER_SPEC_SK.md) section 8:

- 8.3 Search algorithm: 1.5x geometric bracketing (NOT power-of-two),
  binary refinement to a settable resolution (512/1024 tokens).
- 8.2 Configuration of boundaries: decode TPS < 1.0 is FAIL_TOO_SLOW (NOT
  OOM), with hysteresis: median >= 1.05 PASS, 0.95-1.05 BORDERLINE,
  < 0.95 FAIL_TOO_SLOW. Borderline points are re-measured.
- 8.4 Two levels of context testing: allocation-only vs real fill.
- 8.5 Context filler: deterministic, token-controlled, representative of
  prose / code / structured records / mixed task.
- 8.6 Retrieval validation: needles at 5/25/50/75/95% - a context that
  merely ALLOCATES is not "quality usable".

Everything here is pure (no I/O) so it can be unit tested offline.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# --- 8.2 Speed thresholds -------------------------------------------------

#: Decode TPS below this is "too slow" for interactive use.
MIN_DECODE_TPS = 1.0
#: Hysteresis band around MIN_DECODE_TPS.
BORDERLINE_LOW = 0.95
BORDERLINE_HIGH = 1.05


class PointVerdict(str, Enum):
    """Verdict for a single measured context point (spec section 8.2)."""

    PASS = "PASS"
    BORDERLINE = "BORDERLINE"
    FAIL_TOO_SLOW = "FAIL_TOO_SLOW"
    FAIL_OOM = "FAIL_OOM"
    FAIL_LOAD = "FAIL_LOAD"
    FAIL_TIMEOUT = "FAIL_TIMEOUT"
    FAIL_CRASH = "FAIL_CRASH"
    FAIL_UNSUPPORTED = "FAIL_UNSUPPORTED"


def classify_point(
    median_tok_s: float | None,
    ok: bool = True,
    error: str = "",
    min_tps: float = MIN_DECODE_TPS,
) -> PointVerdict:
    """Classify one measured context point.

    A load/OOM/timeout failure is NOT "too slow" - the two must stay
    distinguishable (spec: "this result must not be labelled as OOM").
    `min_tps` is the user's interactive floor (spec 8.2 calls it a
    *default user limit*); the hysteresis band scales with it.
    """
    if not ok:
        text = (error or "").lower()
        if "oom" in text or "out of memory" in text or ("cuda" in text and "alloc" in text):
            return PointVerdict.FAIL_OOM
        if "timeout" in text or "timed out" in text:
            return PointVerdict.FAIL_TIMEOUT
        if "unsupported" in text or "not supported" in text:
            return PointVerdict.FAIL_UNSUPPORTED
        if "crash" in text or "segfault" in text or "terminated" in text:
            return PointVerdict.FAIL_CRASH
        return PointVerdict.FAIL_LOAD
    if median_tok_s is None:
        return PointVerdict.FAIL_LOAD
    floor = max(0.0, float(min_tps))
    band = max(0.05 * floor, 0.01)
    if median_tok_s >= floor + band:
        return PointVerdict.PASS
    if median_tok_s >= max(0.0, floor - band):
        return PointVerdict.BORDERLINE
    return PointVerdict.FAIL_TOO_SLOW


def needs_repeat(verdict: PointVerdict) -> bool:
    """Borderline points are re-measured (spec 8.2)."""
    return verdict is PointVerdict.BORDERLINE


# --- 8.3 Geometric bracketing --------------------------------------------


def geometric_contexts(
    min_ctx: int,
    max_ctx: int,
    factor: float = 1.5,
    round_to: int = 512,
) -> list[int]:
    """Geometric ladder with growth factor 1.5 (NOT power-of-two).

    Values are rounded to `round_to` so every rung is a context length the
    user can actually select. Endpoints are always included.

    Example (min 4096, max 65536, factor 1.5, round 512):
        4096, 6144, 9216, 13824, 20736, 31104, 46656, 65536
    """
    if min_ctx <= 0:
        min_ctx = 512
    max_ctx = max(max_ctx, min_ctx)
    round_to = max(1, int(round_to or 1))
    out: list[int] = []
    v = float(min_ctx)
    guard = 0
    while v < max_ctx and guard < 256:
        guard += 1
        rung = int(round(v / round_to) * round_to)
        rung = max(min_ctx, rung)
        if rung >= max_ctx:
            break
        if not out or rung > out[-1]:
            out.append(rung)
        v *= factor
    if not out or out[-1] != max_ctx:
        out.append(max_ctx)
    return sorted(set(out))


def refine_boundary(
    last_pass: int,
    first_fail: int,
    step: int = 512,
    max_probes: int = 12,
) -> list[int]:
    """Bisection candidate order between a PASS and a FAIL point.

    Returns the contexts to probe, most-informative first, so the caller
    can stop as soon as the interval is narrow enough. The result can be a
    non-power-of-two boundary (e.g. 76288), which is the point.
    """
    step = max(1, int(step or 1))
    lo, hi = int(last_pass), int(first_fail)
    out: list[int] = []
    guard = 0
    while hi - lo > step and len(out) < max_probes and guard < 64:
        guard += 1
        mid = (lo + hi) // 2
        snapped = (mid // step) * step
        if snapped <= lo:
            snapped = lo + step
        if snapped >= hi:
            snapped = hi - step
            snapped = (snapped // step) * step
        if snapped <= lo or snapped >= hi:
            break
        out.append(snapped)
        # Probe the midpoint; the caller updates lo/hi accordingly.
        lo = snapped if snapped in out else lo
    return out


# --- 8.4 Fill ratios ------------------------------------------------------

#: Fast mode: allocation + one deep fill + one near-limit fill.
FILL_RATIOS_FAST: tuple[float, ...] = (0.0, 0.5, 0.9)
#: Detailed mode: full sweep of the context budget.
FILL_RATIOS_DETAILED: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9, 0.98)


def fill_ratios(detailed: bool = False) -> tuple[float, ...]:
    """Fill ratios for the chosen mode (spec 8.4)."""
    return FILL_RATIOS_DETAILED if detailed else FILL_RATIOS_FAST


# --- 8.5 Context filler ---------------------------------------------------

_PROSE_LINES = (
    "The harbour master logged every vessel before the fog lifted.",
    "A narrow gauge railway connected the valley to the northern harbour.",
    "Quartz veins in the northern outcrop produced unusually pure silicon.",
    "The municipal archive stored deeds in a converted grain warehouse.",
    "Sailing schedules were posted twice daily at the customs house.",
    "Ferry operators negotiated a shared berth with the coal merchants.",
    "Surveyors marked the watershed with copper plates set into granite.",
    "The observatory logged clear nights with an obsessive precision.",
    "Cask makers used local oak for barrels that held distilled spirit.",
    "Telegraph operators translated weather reports into shipping advice.",
)

_CODE_LINES = (
    "def parse_header(blob: bytes) -> dict[str, str]:",
    "    fields = blob.split(b'\\n')",
    "    return {k: v for k, v in (f.split(b':', 1) for f in fields) if b':' in f}",
    "async def fetch_all(urls: list[str]) -> list[bytes]:",
    "    async with httpx.AsyncClient() as client:",
    "        return list(await asyncio.gather(*(client.get(u) for u in urls)))",
    "class Ledger:",
    "    def __init__(self, entries: list[Entry]) -> None:",
    "        self.entries = sorted(entries, key=lambda e: e.timestamp)",
    "def median(values: Sequence[float]) -> float:",
    "    ordered = sorted(values)",
    "    mid = len(ordered) // 2",
    "    return ordered[mid] if len(ordered) % 2 else (ordered[mid-1] + ordered[mid]) / 2",
)

_STRUCTURED_LINES = (
    "id=0001 host=alpha status=ok latency_ms=12",
    "id=0002 host=beta status=degraded latency_ms=184",
    "id=0003 host=gamma status=ok latency_ms=9",
    "id=0004 host=delta status=timeout latency_ms=5000",
    "id=0005 host=epsilon status=ok latency_ms=21",
)


def _token_estimate(text: str) -> int:
    """Rough token count (word + punctuation split). Good enough for a filler."""
    return len(re.findall(r"\S+", text))


def build_filler(
    target_tokens: int,
    kind: str = "mixed",
    seed: int = 42,
) -> str:
    """Deterministic, token-controlled context filler (spec 8.5).

    `kind` selects the representative content: prose, code, structured
    records or a mix. The same (target_tokens, kind, seed) always yields
    the SAME text - reproducible comparisons require it.
    """
    if target_tokens <= 0:
        return ""
    pools = {
        "prose": _PROSE_LINES,
        "code": _CODE_LINES,
        "structured": _STRUCTURED_LINES,
    }
    if kind == "mixed":
        pool = list(_PROSE_LINES) + list(_CODE_LINES) + list(_STRUCTURED_LINES)
    else:
        pool = list(pools.get(kind, _PROSE_LINES))

    # Deterministic shuffle driven by the seed (no RNG state, no ordering
    # surprises between runs).
    digest = hashlib.sha256(f"{seed}:{kind}:{target_tokens}".encode()).digest()
    order = list(range(len(pool)))
    for i in range(len(order) - 1, 0, -1):
        j = digest[i % len(digest)] % (i + 1)
        order[i], order[j] = order[j], order[i]

    lines: list[str] = []
    tokens = 0
    idx = 0
    while tokens < target_tokens and idx < 100_000:
        line = pool[order[idx % len(order)]]
        lines.append(line)
        tokens += _token_estimate(line) + 1
        idx += 1
    return "\n".join(lines)


# --- 8.6 Retrieval validation --------------------------------------------

#: Needle positions as a fraction of the filled context.
NEEDLE_POSITIONS: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

#: Fixed needle content - the model must return the key verbatim.
NEEDLE_KEY = "VERDIGEN-8412"


@dataclass
class Needle:
    """One retrieval probe embedded into a filled context."""

    position: float
    key: str
    question: str
    prompt: str = ""


def build_needles(
    filler: str,
    positions: tuple[float, ...] = NEEDLE_POSITIONS,
    key: str = NEEDLE_KEY,
) -> list[Needle]:
    """Insert needles at relative positions inside the filler text.

    Each needle is a short factual sentence the model must be able to
    retrieve later. The surrounding prose is untouched.
    """
    lines = filler.split("\n") if filler else []
    if not lines:
        return []
    out: list[Needle] = []
    for pos in positions:
        idx = min(len(lines) - 1, max(0, int(round(pos * (len(lines) - 1)))))
        statement = (
            f"The access code for the north depot is {key}, "
            f"recorded in ledger section {int(pos * 100):02d}."
        )
        lines.insert(idx, statement)
        out.append(
            Needle(
                position=pos,
                key=key,
                question=(
                    "What is the access code for the north depot? "
                    "Answer with the code only."
                ),
            )
        )
    out[-1].prompt = "\n".join(lines) if out else ""
    return out


def score_retrieval(answer: str, key: str = NEEDLE_KEY) -> bool:
    """Exact-match retrieval check (the model must return the key)."""
    if not answer:
        return False
    return key.lower() in answer.lower()


# --- 8.1 Report assembly --------------------------------------------------


@dataclass
class ContextPoint:
    """One measured context point with its verdict."""

    ctx: int
    verdict: PointVerdict = PointVerdict.PASS
    median_tok_s: float | None = None
    fill_ratio: float = 0.0
    error: str = ""
    measured: bool = True
    retrieval_passed: bool | None = None

    @property
    def ok(self) -> bool:
        return self.verdict in (PointVerdict.PASS, PointVerdict.BORDERLINE)


@dataclass
class ContextReport:
    """Full Context Explorer result (spec section 8.1)."""

    points: list[ContextPoint] = field(default_factory=list)
    max_loadable_context: int | None = None
    max_stable_context: int | None = None
    max_usable_context: int | None = None
    max_retrieval_context: int | None = None
    max_quality_context: int | None = None
    recommended_interactive_context: int | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_loadable_context": self.max_loadable_context,
            "max_stable_context": self.max_stable_context,
            "max_usable_context": self.max_usable_context,
            "max_retrieval_context": self.max_retrieval_context,
            "max_quality_context": self.max_quality_context,
            "recommended_interactive_context": self.recommended_interactive_context,
            "points": [
                {
                    "ctx": p.ctx,
                    "verdict": p.verdict.value,
                    "median_tok_s": p.median_tok_s,
                    "fill_ratio": p.fill_ratio,
                    "error": p.error,
                    "retrieval_passed": p.retrieval_passed,
                }
                for p in self.points
            ],
            "notes": list(self.notes),
        }


def build_report(points: list[ContextPoint]) -> ContextReport:
    """Assemble the report from measured points (spec 8.1 separation).

    - max loadable: largest context that LOADED at all
    - max stable:   largest context that passed the speed threshold
    - max usable:   largest context that is stable AND above MIN_DECODE_TPS
    - max retrieval: largest context that also passed the needle test
    - recommended:  largest usable context (falls back to the fastest)
    """
    report = ContextReport(points=list(points))
    loaded = [p for p in points if p.measured and p.verdict is not PointVerdict.FAIL_LOAD
              and p.verdict is not PointVerdict.FAIL_OOM]
    report.max_loadable_context = max((p.ctx for p in loaded), default=None)

    stable = [p for p in points if p.verdict is PointVerdict.PASS]
    report.max_stable_context = max((p.ctx for p in stable), default=None)

    usable = [
        p
        for p in points
        if p.verdict is PointVerdict.PASS
        and p.median_tok_s is not None
        and p.median_tok_s >= MIN_DECODE_TPS
    ]
    report.max_usable_context = max((p.ctx for p in usable), default=None)

    retrieval = [p for p in usable if p.retrieval_passed is True]
    report.max_retrieval_context = max((p.ctx for p in retrieval), default=None)
    report.max_quality_context = report.max_retrieval_context or report.max_usable_context

    if usable:
        report.recommended_interactive_context = max(p.ctx for p in usable)
    elif stable:
        report.recommended_interactive_context = max(p.ctx for p in stable)
    return report


def format_report(report: ContextReport, model: str) -> str:
    """Human-readable report lines (spec 8.1 example format)."""
    lines = [f"Context Explorer: {model}", ""]
    for p in report.points:
        speed = f"{p.median_tok_s:.1f} tok/s" if p.median_tok_s is not None else "-"
        fill = f" fill={p.fill_ratio:.0%}" if p.fill_ratio else ""
        extra = ""
        if p.retrieval_passed is not None:
            extra = f" retrieval={'PASS' if p.retrieval_passed else 'FAIL'}"
        lines.append(f"  {p.ctx:<8} {p.verdict.value:<15} {speed}{fill}{extra}")

    def _row(label: str, value: int | None) -> None:
        lines.append(f"{label + ':':<34}{value if value is not None else 'n/a'}")

    lines.append("")
    _row("Max loadable context", report.max_loadable_context)
    _row("Max stable context", report.max_stable_context)
    _row("Max >= 1 tok/s context", report.max_usable_context)
    _row("Max retrieval-passing", report.max_retrieval_context)
    _row("Recommended interactive", report.recommended_interactive_context)
    for note in report.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)
