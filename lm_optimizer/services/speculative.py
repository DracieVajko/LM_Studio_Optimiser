"""Optional Speculative Decoding Optimization stage.

Runs ONLY when: a suitable draft model exists locally, the runtime supports
the method, and the benchmark can measure it. Otherwise skipped with a reason.
Reports baseline vs speculative speed + VRAM cost; acceptance statistics only
if the server exposes them (it currently does not).
"""

import contextlib
import statistics

from lm_optimizer.domain.models import LoadConfiguration
from lm_optimizer.logging_config import get_logger
from lm_optimizer.services.hostguard import gpu_free_mb

logger = get_logger(__name__)

DRAFT_HINTS = ("draft", "medusa", "eagle", "speculative")
PROBE_PROMPT = "Say hi in 5 words."
PROBE_TOKENS = 30
PROBE_REPS = 3


def discover_drafts(models) -> list:
    """Local models that look like speculative draft models (name heuristic)."""
    found = []
    for m in models:
        key = (getattr(m, "id", "") or "").lower()
        name = (getattr(m, "name", "") or "").lower()
        if any(h in key or h in name for h in DRAFT_HINTS):
            found.append(m)
    return found


def _vram_used_mb() -> float | None:
    for g in gpu_free_mb():
        if "error" not in g:
            return g["total_mb"] - g["free_mb"]
    return None


async def _probe_toks(client, model_id: str) -> list[float]:
    speeds = []
    for _ in range(PROBE_REPS):
        out = await client.chat_completion(
            model=model_id, input_text=PROBE_PROMPT, temperature=0.3,
            max_output_tokens=PROBE_TOKENS)
        stats = out.get("_stats", {}) or {}
        tok = stats.get("tokens_per_second", 0) or 0
        if tok > 0:
            speeds.append(float(tok))
    return speeds


async def ab_compare(client, model_id: str, base_cfg: LoadConfiguration,
                     draft_key: str, use_mtp: bool = True) -> dict:
    """Baseline vs speculative A/B at a fixed config. Never raises fatally."""
    result: dict = {"model": model_id, "draft": draft_key, "ran": False}
    try:
        vram_before = _vram_used_mb()
        res = await client.load_model(model_id, base_cfg)
        if not res.success:
            result["error"] = f"baseline load failed: {res.error}"
            return result
        try:
            base = await _probe_toks(client, model_id)
        finally:
            await client.ensure_unloaded(model_id)
        if not base:
            result["error"] = "baseline produced no measurable generations"
            return result

        spec_cfg = LoadConfiguration(
            context_length=base_cfg.context_length,
            flash_attention=base_cfg.flash_attention,
            offload_kv_cache_to_gpu=base_cfg.offload_kv_cache_to_gpu,
            eval_batch_size=base_cfg.eval_batch_size,
        )
        if use_mtp:
            spec_cfg.speculative_draft_mtp = True
        else:
            spec_cfg.speculative_draft_simple = True
            spec_cfg.speculative_draft_model = draft_key
        res = await client.load_model(model_id, spec_cfg)
        if not res.success:
            result["error"] = f"speculative load failed: {res.error}"
            result["baseline_tok_s"] = round(statistics.median(base), 1)
            return result
        try:
            spec = await _probe_toks(client, model_id)
        finally:
            await client.ensure_unloaded(model_id)
        vram_after = _vram_used_mb()

        result["ran"] = True
        result["baseline_tok_s"] = round(statistics.median(base), 1)
        result["spec_tok_s"] = (
            round(statistics.median(spec), 1) if spec else 0.0)
        if result["spec_tok_s"] > 0:
            result["speedup"] = round(
                result["spec_tok_s"] / max(result["baseline_tok_s"], 1e-9), 2)
        if vram_before is not None and vram_after is not None:
            result["vram_delta_mb"] = round(vram_after - vram_before, 0)
        result["acceptance"] = "not exposed by server stats"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"[:200]
        with contextlib.suppress(Exception):
            await client.ensure_unloaded(model_id)
    return result
