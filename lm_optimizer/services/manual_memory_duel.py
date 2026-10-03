"""Manual mmap / keep-in-memory duel (MANUAL_ONLY re-measure + verdict).

`try_mmap` / `keep_model_in_memory` are GUI/CLI-only: this module never
toggles anything. It loads the auto best (numbers from the run DB, identity
from the best `.md`), the CLI pauses for the user to flip the GUI switch,
re-measures the SAME `LoadConfiguration`, and `verdict()` decides keep/revert.

Sourcing decision: gen tok/s + TTFT are NOT reliably parseable from the
`.md` — the "Typical generation/TTFT (median)" lines exist only when 2+
identical measurements were recorded, the "Stored repeat" line carries gen
but no TTFT, and the per-test/ crowns lines are different aggregates. Any
`.md` regex would silently compare different aggregates across runs. So the
`.md` is used only for model-identity match + canonical path, and the
machine-readable (gen, TTFT, load config, context, style) come from the run
DB (`run_repo.get_by_model` newest-first, full run via `get`). The `.md`
Typical lines are a fallback when the DB has no run for the model.
"""

import re
from pathlib import Path

from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)

KEEP_THRESHOLD = 0.05

MMAP_PROMPT = "Switch mmap OFF in LM Studio GUI (default ON), then press Enter"
KEEP_PROMPT = "Turn OFF Keep Model in Memory in LM Studio GUI, then press Enter"

GUIDANCE_KEEP = "Keep OFF"
GUIDANCE_REVERT = "Revert to ON"

_MODEL_ID_RE = re.compile(r"^- Model ID:\s*(.+?)\s*$", re.MULTILINE)
_TYPICAL_GEN_RE = re.compile(r"Typical generation:\s*([\d.]+)\s*tok/s\s*\(median\)")
_TYPICAL_TTFT_RE = re.compile(r"Typical est\. TTFT:\s*([\d.]+)\s*ms\s*\(median\)")


def verdict(
    auto_gen: float,
    manual_gen: float,
    auto_ttft: float,
    manual_ttft: float,
    threshold: float = KEEP_THRESHOLD,
) -> dict:
    """Keep/revert verdict for a manual re-measure.

    Keep requires >= `threshold` relative generation tok/s gain; TTFT is a
    tiebreak reported in the reason, never a substitute for the gen gate.
    A zero/negative baseline cannot prove a win: revert fail-closed.
    """
    try:
        auto_gen = float(auto_gen)
        manual_gen = float(manual_gen)
        auto_ttft = float(auto_ttft)
        manual_ttft = float(manual_ttft)
    except (TypeError, ValueError):
        return {
            "decision": "revert",
            "reason": f"unparseable speeds ({GUIDANCE_REVERT}; no measurable win)",
        }
    gain = (manual_gen - auto_gen) / auto_gen if auto_gen > 0 else None
    ttft_delta = manual_ttft - auto_ttft
    ttft_note = (
        f"TTFT {manual_ttft:.0f} vs {auto_ttft:.0f} ms "
        f"({ttft_delta:+.0f} ms, tiebreak only)"
    )
    if gain is not None and gain >= threshold:
        return {
            "decision": "keep",
            "reason": (
                f"manual +{gain:.1%} gen tok/s "
                f"(>= +{threshold:.0%}); {ttft_note} - {GUIDANCE_KEEP}"
            ),
        }
    if gain is None:
        return {
            "decision": "revert",
            "reason": (
                f"auto baseline {auto_gen:.1f} tok/s cannot prove a gain; "
                f"revert guidance stands - {GUIDANCE_REVERT}"
            ),
        }
    return {
        "decision": "revert",
        "reason": (
            f"manual {gain:+.1%} gen tok/s (< +{threshold:.0%}): "
            f"no measurable difference, revert guidance stands; "
            f"{ttft_note} - {GUIDANCE_REVERT}"
        ),
    }


def _expected_best_name(model_id: str) -> str:
    from lm_optimizer.services.reporting import sanitize_model_filename

    return f"{sanitize_model_filename(model_id)}-best.md"


def _resolve_best_file(path: str | Path, model_id: str, results_dir: str | Path) -> Path:
    """Resolve the best `.md` for `model_id`, refusing cross-model compares."""
    expected = _expected_best_name(model_id)
    given = Path(path)
    if (
        given.suffix == ".md"
        and given.stem.endswith("-best")
        and given.name != expected
    ):
        raise ValueError(
            f"no matching auto best: '{given.name}' names another model "
            f"(expected '{expected}')"
        )
    if given.is_dir():
        candidate = given / expected
        if candidate.is_file():
            return candidate
    elif given.is_file():
        return given
    canonical = Path(results_dir) / expected
    if canonical.is_file():
        return canonical
    raise ValueError(
        f"no matching auto best for '{model_id}': "
        f"missing '{expected}' in '{results_dir}' (run auto/optimize first)"
    )


def _parse_model_id(text: str) -> str | None:
    match = _MODEL_ID_RE.search(text)
    return match.group(1).strip() if match else None


def _best_from_db(model_id: str, repo=None) -> dict | None:
    """Newest run with a measured best config (numbers + load config)."""
    if repo is None:
        from lm_optimizer.database.repositories import run_repo as repo
    try:
        lean = repo.get_by_model(model_id, 50)
    except Exception as e:
        logger.warning("Run DB lookup failed", model=model_id, error=str(e)[:160])
        return None
    for row in lean:
        try:
            full = repo.get(str(row.id)) or row
            best = full.get_best_config()
        except Exception:
            continue
        if best is None:
            continue
        gen = best.get_avg_generation_tok_s()
        if gen <= 0:
            continue
        return {
            "model_id": model_id,
            "gen_tok_s": gen,
            "ttft_ms": best.get_avg_estimated_ttft_ms(),
            "context_length": best.context_length,
            "load_config": best.config,
            "style": (best.generation or {}).get("style", "balanced"),
            "source": "run-db",
            "run_id": str(full.id),
        }
    return None


def _best_from_md(text: str, model_id: str) -> dict | None:
    """Fallback: Typical medians from the best `.md` (no load config)."""
    gen_match = _TYPICAL_GEN_RE.search(text)
    ttft_match = _TYPICAL_TTFT_RE.search(text)
    if not gen_match or not ttft_match:
        return None
    return {
        "model_id": model_id,
        "gen_tok_s": float(gen_match.group(1)),
        "ttft_ms": float(ttft_match.group(1)),
        "context_length": None,
        "load_config": None,
        "style": "balanced",
        "source": "best-md",
        "run_id": None,
    }


def load_auto_best(
    path: str | Path,
    model_id: str,
    results_dir: str | Path = "results",
    repo=None,
) -> dict:
    """Load the auto best for `model_id`. Raises ValueError on mismatch.

    Refuses (rather than comparing across models) when: the `.md` filename
    names another model, the file is missing, its Model ID line differs or
    is absent, or neither the run DB nor the `.md` Typical lines yield
    machine-readable (gen, TTFT) numbers.
    """
    best_file = _resolve_best_file(path, model_id, results_dir)
    try:
        text = best_file.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"no matching auto best: cannot read '{best_file}': {e}")
    file_model = _parse_model_id(text)
    if file_model != model_id:
        raise ValueError(
            f"no matching auto best: '{best_file.name}' is for "
            f"'{file_model or 'unknown'}', not '{model_id}'"
        )
    db_best = _best_from_db(model_id, repo=repo)
    if db_best is not None:
        return {**db_best, "path": str(best_file)}
    md_best = _best_from_md(text, model_id)
    if md_best is not None:
        logger.warning("Run DB has no best; using .md Typical medians", model=model_id)
        return {**md_best, "path": str(best_file)}
    raise ValueError(
        f"no matching auto best for '{model_id}': run DB has no measured best "
        f"and '{best_file.name}' holds no Typical medians"
    )
