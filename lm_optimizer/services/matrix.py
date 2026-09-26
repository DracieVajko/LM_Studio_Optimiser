"""VRAM fit/speed matrix core: context x flash-attention x KV placement.

Ultra-short generations measure fit/speed, NOT quality. Every configuration
load -> chat -> unload (finally) with server-state verification.
"""

import statistics
import time

from lm_optimizer.domain.models import LoadConfiguration
from lm_optimizer.logging_config import get_logger
from lm_optimizer.services.lm_studio import LMStudioClient

logger = get_logger(__name__)

PROMPT = "Say hi in 5 words."
MAX_TOKENS = 30
DEFAULT_CONTEXTS = [2048, 4096, 8192]


def build_matrix(
    contexts: list[int],
    flash_opts: list[bool] | None = None,
    kv_opts: list[bool] | None = None,
) -> list[LoadConfiguration]:
    """All context x flash x KV combinations (deterministic order)."""
    flash_iter = (True, False) if flash_opts is None else flash_opts
    kv_iter = (True, False) if kv_opts is None else kv_opts
    return [
        LoadConfiguration(context_length=ctx, flash_attention=flash, offload_kv_cache_to_gpu=kv)
        for ctx in sorted(contexts)
        for flash in flash_iter
        for kv in kv_iter
    ]


def fmt_cfg(cfg: LoadConfiguration) -> str:
    kv = "gpu" if cfg.offload_kv_cache_to_gpu else "cpu"
    flash = "on" if cfg.flash_attention else "off"
    return f"{cfg.context_length:<6} flash={flash:<3} kv={kv:<3}"


def recommendation_line(results: list[dict]) -> str:
    """Data-driven recommendation from measured medians (never hardcoded claims)."""
    gpu = [r["tok_s"] for r in results if r["ok"] and r["kv"] == "gpu"]
    cpu = [r["tok_s"] for r in results if r["ok"] and r["kv"] == "cpu"]
    if gpu and cpu:
        g, c = statistics.median(gpu), statistics.median(cpu)
        faster = "GPU" if g >= c else "CPU"
        return (
            f"KV on GPU median: {g:.1f} tok/s, KV on CPU median: {c:.1f} tok/s "
            f"(ratio {g / max(c, 1e-9):.1f}x). "
            f"Measured faster: KV on {faster}. "
            f"Typical 6GB default: context 8192, flash on, KV on GPU."
        )
    if gpu:
        g = statistics.median(gpu)
        return (
            f"KV on GPU median: {g:.1f} tok/s (no passing KV-on-CPU config). "
            f"Typical 6GB default: context 8192, flash on, KV on GPU."
        )
    if cpu:
        c = statistics.median(cpu)
        return (
            f"KV on CPU median: {c:.1f} tok/s (no passing KV-on-GPU config). "
            f"Check VRAM headroom and KV cache settings."
        )
    return (
        "No passing configuration measured - check errors above "
        "(e.g. KV cache quantization requires flash attention)."
    )


async def run_one(client: LMStudioClient, model: str, cfg: LoadConfiguration) -> dict:
    """Load -> chat -> verified unload. Returns a result dict (never raises)."""
    res = {
        "ctx": cfg.context_length,
        "flash": bool(cfg.flash_attention),
        "kv": "gpu" if cfg.offload_kv_cache_to_gpu else "cpu",
        "ok": False,
        "tok_s": 0.0,
        "ttft_ms": 0.0,
        "load_s": 0.0,
        "error": "",
    }
    try:
        pre = await client.get_loaded_instances()
        if pre:
            res["error"] = f"stale instances before load: {len(pre)}"
            await client.unload_all()
        t0 = time.perf_counter()
        loaded = await client.load_model(model, cfg)
        res["load_s"] = time.perf_counter() - t0
        if not loaded.success:
            res["error"] = f"load failed: {loaded.error}"
            return res
        out = await client.chat_completion(
            model=model, input_text=PROMPT, temperature=0.3, max_output_tokens=MAX_TOKENS
        )
        choices = out.get("choices") or []
        text = choices[0].get("message", {}).get("content", "") if choices else ""
        if not text.strip():
            # Degenerate generation (immediate EOS): failed, not 500 tok/s.
            res["ok"] = False
            res["error"] = "empty output"
            return res
        stats = out.get("_stats", {}) or {}
        res["tok_s"] = stats.get("tokens_per_second", 0) or 0
        res["ttft_ms"] = (stats.get("time_to_first_token_seconds", 0) or 0) * 1000
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"[:160]
    else:
        res["ok"] = True
    finally:
        try:
            if not await client.ensure_unloaded(model):
                res["ok"] = False
                res["error"] += " | model still loaded after unload"
        except Exception as e:
            res["ok"] = False
            res["error"] += f" | unload failed: {e}"
    return res
