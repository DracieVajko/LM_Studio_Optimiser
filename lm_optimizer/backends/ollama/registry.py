"""Ollama options registry + search space (static table, no probing).

Ollama silently ignores unknown ``options`` keys server-side, so there is no
400-probe equivalent to the LM Studio ``unrecognized_keys`` channel — the
supported set below is a static table (Ollama documented ``options``), never
a probing claim.

Sweep model: per-request ``options`` on ``POST /api/generate`` (see
``lm_optimizer/backends/ollama/client.py``). ``num_ctx`` changes force a
model reload server-side; that load cost surfaces in ``load_duration``
metrics rather than being hidden.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lm_optimizer.domain.models import HardwareInfo

#: Exact static set of Ollama ``options`` keys the optimizer may set.
OLLAMA_OPTIONS = [
    "num_ctx",
    "num_batch",
    "num_thread",
    "num_gpu",
    "use_mmap",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "repeat_penalty",
]

#: Load/perf keys swept per-request (subset of OLLAMA_OPTIONS).
LOAD_KEYS = ["num_ctx", "num_batch", "num_thread", "num_gpu", "use_mmap"]

#: Sampling keys are NOT swept here: they pass through to the existing
#: generation profiles (``lm_optimizer/services/generation_profiles.py``).
SAMPLING_KEYS = ["temperature", "top_k", "top_p", "min_p", "repeat_penalty"]

_DEFAULT_NUM_CTX = [2048, 4096, 8192]
_DEFAULT_NUM_BATCH = [64, 256, 512]


def context_length_from_show(model_info: dict) -> int | None:
    """Extract ``llama.context_length`` from raw ``POST /api/show`` details."""
    if not isinstance(model_info, dict):
        return None
    details = model_info.get("model_info")
    if not isinstance(details, dict):
        return None
    for key in ("llama.context_length", "llama.context_length_keys", "context_length"):
        value = details.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return None


def parameter_count_from_show(model_info: dict) -> int | None:
    """Extract the parameter count from raw ``POST /api/show`` details."""
    if not isinstance(model_info, dict):
        return None
    details = model_info.get("model_info")
    if isinstance(details, dict):
        for key in ("general.parameter_count", "general.total_parameters", "parameter_count"):
            value = details.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
    return None


def _num_thread_default(hardware: HardwareInfo) -> list[int]:
    logical = getattr(hardware, "cpu_cores_logical", 0) or 0
    if logical <= 0:
        logical = os.cpu_count() or 8
    return [int(logical)]


def _num_gpu_default(hardware: HardwareInfo) -> list[int]:
    """Coarse CPU-vs-GPU probe: 0 = CPU-only baseline, max = GPUs engaged."""
    gpus = getattr(hardware, "gpus", None) or []
    count = getattr(hardware, "gpu_count", 0) or 0
    maximum = max(int(count), len(gpus))
    if maximum <= 0:
        return [0]
    return sorted({0, maximum})


def ollama_space(
    model_info: dict,
    hardware: HardwareInfo,
    advanced: dict | None = None,
) -> dict[str, list]:
    """Build the Ollama per-request ``options`` search space.

    ``model_info`` is the raw ``POST /api/show`` details dict (as returned
    by ``OllamaClient.show()``); ``num_ctx`` candidates are capped by
    ``llama.context_length`` (filter-only: values above the cap are dropped,
    the cap itself is never appended, so large-context models keep exactly
    the ``[2048, 4096, 8192]`` defaults). Sampling keys are intentionally
    absent — they stay owned by the existing generation profiles.
    """
    options = dict(advanced or {})

    cap = context_length_from_show(model_info)
    if isinstance(options.get("num_ctx"), list) and options["num_ctx"]:
        num_ctx = sorted(set(options["num_ctx"]))
    elif cap is None or cap <= 0:
        num_ctx = list(_DEFAULT_NUM_CTX)
    else:
        num_ctx = [c for c in _DEFAULT_NUM_CTX if c <= cap] or [cap]

    num_batch = (
        sorted(set(options["num_batch"]))
        if isinstance(options.get("num_batch"), list) and options["num_batch"]
        else list(_DEFAULT_NUM_BATCH)
    )
    num_thread = (
        sorted(set(options["num_thread"]))
        if isinstance(options.get("num_thread"), list) and options["num_thread"]
        else _num_thread_default(hardware)
    )
    num_gpu = (
        sorted(set(options["num_gpu"]))
        if isinstance(options.get("num_gpu"), list) and options["num_gpu"]
        else _num_gpu_default(hardware)
    )

    return {
        "num_ctx": num_ctx,
        "num_batch": num_batch,
        "num_thread": num_thread,
        "num_gpu": num_gpu,
    }


__all__ = [
    "LOAD_KEYS",
    "OLLAMA_OPTIONS",
    "SAMPLING_KEYS",
    "context_length_from_show",
    "ollama_space",
    "parameter_count_from_show",
]
