"""Model recommendations for a VRAM budget.

No live probing here: fit is estimated from the downloaded file size
(weights) plus headroom for KV cache / context / runtime overhead.
"""

from dataclasses import dataclass

from lm_optimizer.domain.models import ModelIdentity
from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)

# Share of VRAM reserved for KV cache + context + runtime overhead.
HEADROOM_FACTOR = 1.3


def manual_checklist(vram_gb: float) -> list[str]:
    """Manual GUI/host checklist for fitting models into a VRAM budget.

    These are the levers the REST API cannot set (GUI/SDK/CLI on the host only).
    KV cache quantization starter is Q4 (needs flash attention ON).
    """
    return [
        f"K/V cache quantization -> Q4 STARTER in LM Studio GUI (requires Flash Attention "
        f"ON) [biggest {vram_gb:g}GB lever; REST cannot set this]. If loads fail with "
        f"'requires flash attention', keep flash on or set KV quant to F16.",
        "KV cache placement -> GPU (~3x tok/s vs CPU); use CPU fallback for 9B+ models",
        "Context: 8192 default; 4096 for 8-14B models; 2048 + KV-on-CPU for 20B+",
        "GPU offload -> max via `lms load <model> --gpu max` (REST only toggles KV placement)",
        "CPU threads -> maximum logical cores in LM Studio GUI (REST cannot set n_threads)",
        "Weights: prefer Q4_K_M <=4B (~3GB); Q3_*/IQ3_XXS for 8-14B; check with "
        "`lms load <model> --estimate-only` before testing",
        "Small models first (<12B, 4B preferred); large (>12B) only on explicit request",
        "Discipline: before AND after each test, GET /api/v1/models -> "
        "loaded_instances must be empty; never leave a model loaded",
    ]


def expert_advice(model: ModelIdentity) -> list[str] | None:
    """num_experts recommendations for MoE models (None when not MoE).

    The exact total expert count is not exposed via REST; check the model card.
    num_experts itself IS REST-settable (load parameter).
    """
    if not model.is_moe:
        return None
    return [
        f"MoE architecture detected ({model.architecture or 'unknown arch'}).",
        "num_experts is REST-settable: use full experts when the model fits; "
        "on load refusal try total/2, then total/4.",
        "Total expert count is not exposed via REST; check the model card "
        "for the exact number.",
        "Expert reduction trades reasoning capacity for fit; re-run quality "
        "checks after changing it.",
    ]


def threads_advice(physical: int, logical: int, current: int | None = None) -> list[str]:
    """CPU thread guidance: recommend maximum, report current setting."""
    lines = [
        f"Detected CPU: {physical} physical / {logical} logical cores.",
        f"Recommended LM Studio CPU threads: {logical} (maximum).",
    ]
    if current is not None:
        if current >= logical:
            lines.append(f"Currently set: {current} (at maximum).")
        else:
            lines.append(
                f"Currently set: {current} (below maximum {logical}; "
                f"raise it in LM Studio GUI if CPU-bound)."
            )
    lines.append("Note: n_threads is not settable via REST; set it in LM Studio GUI.")
    return lines


@dataclass
class ModelRecommendation:
    """Fit verdict for one model against a VRAM budget."""

    model_id: str
    name: str
    size_gb: float | None
    quantization: str | None
    architecture: str | None
    context_limit: int | None
    fits: bool
    marginal: bool
    reason: str


def estimate_need_gb(model: ModelIdentity) -> float | None:
    """Estimated VRAM need: file size (weights) + headroom, or None if unknown."""
    if not model.size_bytes:
        return None
    return model.size_bytes / 1024**3 * HEADROOM_FACTOR


def diagnose_load_error(error: str | None) -> dict:
    """Classify a model-load failure into category + manual guidance.

    The REST API cannot change KV quantization / offload ratio / threads, so
    every category ends with concrete host-side (GUI/CLI) steps. Deterministic
    heuristics only - no AI decider needed.
    """
    text = (error or "").lower()
    if "requires flash attention" in text:
        return {
            "category": "flash_required",
            "retryable": False,
            "prune_flash": True,
            "guidance": [
                "Server requires flash attention with the current KV cache quantization.",
                "Keep flash_attention=true for this run (auto-pruned), or",
                "set V Cache Quantization Type to F16 in LM Studio GUI and reload.",
            ],
        }
    if any(k in text for k in ("out of memory", "oom", "tried to allocate", "cuda error", "allocation failed")):
        return {
            "category": "oom",
            "retryable": False,
            "prune_flash": False,
            "guidance": [
                "Lower context_length (try 4096, then 2048).",
                "Move KV cache to CPU.",
                "Lower KV cache quantization (Q8 -> Q4) in LM Studio GUI.",
                "Or pick smaller weights (Q3/IQ3); check `lms load --estimate-only` first.",
            ],
        }
    if any(k in text for k in ("500", "internal server", "timeout", "timed out", "connection")):
        return {
            "category": "transient",
            "retryable": True,
            "prune_flash": False,
            "guidance": ["Transient server error: retry once, then continue with next model."],
        }
    if any(k in text for k in ("not found", "no such model", "invalid model")):
        return {
            "category": "model_not_found",
            "retryable": False,
            "prune_flash": False,
            "guidance": ["Check the model key spelling against `models` output."],
        }
    if "unrecognized" in text:
        return {
            "category": "api_drift",
            "retryable": False,
            "prune_flash": False,
            "guidance": [
                "Server rejected a parameter the probe believed supported.",
                "Re-run `status` to re-probe capabilities; check `lms --version` "
                "and report the version drift.",
            ],
        }
    if "failed to load model" in text:
        return {
            "category": "load_refused",
            "retryable": False,
            "prune_flash": False,
            "guidance": [
                "Backend refused the load (not OOM): check `lms load <model> "
                "--estimate-only` and LM Studio server logs.",
                "Try a different quantization of the same model or update GPU drivers/runtime.",
            ],
        }
    return {
        "category": "unknown",
        "retryable": False,
        "prune_flash": False,
        "guidance": ["Unclassified load error; check LM Studio server logs and retry once."],
    }


def recommend_for_vram(models: list[ModelIdentity], vram_gb: float) -> list[ModelRecommendation]:
    """Rank local models by fit for a VRAM budget (smallest fitting first)."""
    recs = []
    for m in models:
        size_gb = m.size_bytes / 1024**3 if m.size_bytes else None
        need = estimate_need_gb(m)
        if need is None:
            recs.append(
                ModelRecommendation(
                    model_id=m.id,
                    name=m.name,
                    size_gb=None,
                    quantization=m.quantization,
                    architecture=m.architecture,
                    context_limit=m.context_limit,
                    fits=False,
                    marginal=False,
                    reason="unknown size: run `lms load --estimate-only` first",
                )
            )
        elif need <= vram_gb:
            recs.append(
                ModelRecommendation(
                    model_id=m.id,
                    name=m.name,
                    size_gb=size_gb,
                    quantization=m.quantization,
                    architecture=m.architecture,
                    context_limit=m.context_limit,
                    fits=True,
                    marginal=False,
                    reason=f"est. {need:.1f}GB <= {vram_gb:g}GB budget",
                )
            )
        elif size_gb is not None and size_gb <= vram_gb:
            recs.append(
                ModelRecommendation(
                    model_id=m.id,
                    name=m.name,
                    size_gb=size_gb,
                    quantization=m.quantization,
                    architecture=m.architecture,
                    context_limit=m.context_limit,
                    fits=False,
                    marginal=True,
                    reason=f"weights fit ({size_gb:.1f}GB) but KV is tight: "
                    f"small context and/or KV on CPU",
                )
            )
        else:
            recs.append(
                ModelRecommendation(
                    model_id=m.id,
                    name=m.name,
                    size_gb=size_gb,
                    quantization=m.quantization,
                    architecture=m.architecture,
                    context_limit=m.context_limit,
                    fits=False,
                    marginal=False,
                    reason=f"est. {need:.1f}GB > {vram_gb:g}GB budget",
                )
            )

    def sort_key(r: ModelRecommendation) -> tuple[int, float]:
        group = 0 if r.fits else (1 if r.marginal else 2)
        return (group, r.size_gb if r.size_gb is not None else float("inf"))

    return sorted(recs, key=sort_key)
