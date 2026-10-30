"""Strict configuration failure state machine (§1-§4, §25).

Phases: CREATED -> LOAD_REQUESTED -> LOADED -> VERIFIED -> PREHEATED ->
BENCHMARKED -> QUALITY_EVALUATED -> SCORED -> CANDIDATE_ELIGIBLE.
Any phase failure lands on a terminal failure class; failures are never
collapsed into one bucket:

- INCOMPATIBLE: hard lock (structural: unsupported/unrecognized parameter,
  model incompatibility). Never retried blindly.
- OOM / TIMEOUT / LOAD_FAILED / BENCHMARK_FAILED / VERIFY_FAILED /
  PREHEAT_FAILED: soft/learned failures.
- QUALITY_FAILED: benchmark metrics preserved, quality recorded, score None,
  excluded from winner selection (not a runtime failure).
"""

from __future__ import annotations

import re

from lm_optimizer.domain.models import SOFT_FAILURES, ConfigurationStatus

_OOM_RE = re.compile(r"\boom\b|out of memory|memory pressure|cuda out of")
_TIMEOUT_SIGNALS = ("timeout", "timed out", "deadline")
_HARD_SIGNALS = (
    "unrecognized",
    "unsupported",
    "not supported",
    "invalid parameter",
    "incompatible",
    "no such model",
    "model_not_found",
)
_LOAD_SIGNALS = ("load failed", "failed to load", "refus", "connection", "econn")


def classify_error(error: str | None) -> ConfigurationStatus:
    """Map a raw error string to a terminal failure class (order matters)."""
    err = (error or "").lower()
    if not err:
        return ConfigurationStatus.BENCHMARK_FAILED
    if _OOM_RE.search(err):
        return ConfigurationStatus.OOM
    if any(s in err for s in _TIMEOUT_SIGNALS):
        return ConfigurationStatus.TIMEOUT
    if any(s in err for s in _HARD_SIGNALS):
        return ConfigurationStatus.INCOMPATIBLE
    if any(s in err for s in _LOAD_SIGNALS):
        return ConfigurationStatus.LOAD_FAILED
    return ConfigurationStatus.BENCHMARK_FAILED


def is_hard(status: ConfigurationStatus) -> bool:
    """Hard lock (INCOMPATIBLE) vs soft/learned failure."""
    return status == ConfigurationStatus.INCOMPATIBLE


def is_soft(status: ConfigurationStatus) -> bool:
    """Soft failure that may succeed under other conditions."""
    return status in SOFT_FAILURES


def is_terminal_failure(status: ConfigurationStatus) -> bool:
    """Any non-passing terminal state (nothing here may be scored a winner)."""
    return status != ConfigurationStatus.PASSED


_SIG_NUM_RE = re.compile(r"0x[0-9a-f]+|\b\d[\d._-]*\b")
_SIG_WS_RE = re.compile(r"\s+")


def normalize_error_signature(error: str | None) -> str:
    """Stable group key for an error: lowercase, numbers/ids masked.

    "CUDA OOM ctx=8192" and "cuda oom ctx=16384" share a signature;
    distinct causes stay distinct. Empty/blank → "unknown".
    """
    text = (error or "").strip().lower()
    if not text:
        return "unknown"
    text = _SIG_NUM_RE.sub("#", text)
    text = _SIG_WS_RE.sub(" ", text).strip()
    return text or "unknown"


def _config_fingerprint(cfg) -> tuple:
    """Materially distinguishing dims of a tested config."""
    c = getattr(cfg, "config", None)
    dims = (
        getattr(cfg, "context_length", None),
        getattr(c, "flash_attention", None),
        getattr(c, "offload_kv_cache_to_gpu", None),
        getattr(c, "eval_batch_size", None),
        getattr(c, "physical_batch_size", None),
        getattr(c, "parallel", None),
        getattr(c, "gpu_ratio", None),
    )
    return tuple(dims)


def should_stop_repeating(configs: list, threshold: int = 3) -> dict:
    """Circuit-breaker over repeated identical failures (display/advisory).

    Trips only when the same normalized signature hits >= threshold
    materially different configs with no PASSED baseline success.
    Flash-attention-specific failures never trip the global breaker (they
    are dimension-specific and owned by the flash-required lesson).
    QUALITY_FAILED suites are exploration evidence, not error cascades.
    """
    failures = [
        c for c in (configs or [])
        if getattr(c, "status", None) != ConfigurationStatus.PASSED
        and getattr(c, "status", None) != ConfigurationStatus.QUALITY_FAILED
        and (getattr(c, "error", None) or "").strip()
    ]
    if any(
        getattr(c, "status", None) == ConfigurationStatus.PASSED
        and getattr(c, "score", None) is not None
        for c in (configs or [])
    ):
        return {"trip": False, "reason": "baseline success exists"}
    groups: dict[str, list] = {}
    for c in failures:
        groups.setdefault(normalize_error_signature(c.error), []).append(c)
    for sig, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(items) < threshold:
            continue
        if len({_config_fingerprint(c) for c in items}) < threshold:
            continue
        if "flash" in sig:
            continue  # dimension-specific; never a global stop
        return {
            "trip": True,
            "signature": sig,
            "count": len(items),
            "reason": f"{len(items)}x identical failure across distinct configs",
        }
    if any("flash" in sig and len(items) >= threshold for sig, items in groups.items()):
        return {"trip": False, "reason": "flash-specific"}
    return {"trip": False, "reason": "no repeated identical failure"}


def failure_signature_rows(configs: list) -> list[str]:
    """Display rows per error signature with counts, contexts and durations."""
    groups: dict[str, list] = {}
    for c in (configs or []):
        if getattr(c, "status", None) == ConfigurationStatus.PASSED:
            continue
        groups.setdefault(normalize_error_signature(getattr(c, "error", None)), []).append(c)
    rows = []
    for sig, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        ctxs = sorted({getattr(c, "context_length", "?") for c in items})
        total_ms = sum(float(getattr(c, "duration_ms", 0.0) or 0.0) for c in items)
        rows.append(
            f"- {sig}: {len(items)}x across ctx {ctxs}, total {total_ms:.0f} ms"
        )
    return rows or ["- No failures recorded."]
