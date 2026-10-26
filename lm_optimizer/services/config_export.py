"""Reusable load-config export (LM Studio REST, vLLM engine config, instructions)."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def build_load_config_export(run: Any) -> dict:
    """Build the export payload for a run's validated winner.

    Raises ValueError when the run has no validated winner.
    """
    best = run.get_best_config() if hasattr(run, "get_best_config") else None
    if best is None:
        raise ValueError("No validated winner in this run")
    cfg = best.config
    params = cfg.to_api_params()
    vllm = {
        "name": f"{run.model.name or run.model.id} optimized",
        "load_params": {
            "n_ctx": cfg.context_length,
            "n_batch": cfg.eval_batch_size,
            "flash_attn": cfg.flash_attention,
            "no_kv_offload": (
                None
                if cfg.offload_kv_cache_to_gpu is None
                else not cfg.offload_kv_cache_to_gpu
            ),
            "rope_freq_base": cfg.rope_freq_base,
            "rope_freq_scale": cfg.rope_freq_scale,
        },
    }
    vllm["load_params"] = {k: v for k, v in vllm["load_params"].items() if v is not None}
    return {
        "run_id": str(run.id),
        "model_id": run.model.id,
        "config_id": str(best.id),
        "config": params,
        "vllm_engine_config": vllm,
        "lmstudio_load_body": {"model": run.model.id, **params},
        "instructions": [
            "llama.cpp/GGUF: LM Studio has no public API to write per-model "
            "defaults; open My Models -> gear and paste the values from 'config'.",
            "vLLM: save 'vllm_engine_config' to a .json file and run "
            "`lms load <model> --engine-config-file <file>`.",
            "Any engine: POST /api/v1/models/load with 'lmstudio_load_body'.",
        ],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
