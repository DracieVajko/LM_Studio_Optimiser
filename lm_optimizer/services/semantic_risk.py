"""Semantic-risk tiers for quality-recovery culprit ranking (mission §15).

Explicit metadata: how likely is a changed runtime parameter to alter output
semantics? HIGH = execution path (speculative/MTP, experts, KV
representation). MEDIUM = memory/runtime path. LOW = scheduling/memory
management. Unknown future keys default to LOW (never hidden, never certain).
The parameter registry is untouched; this module only ranks recovery order.
"""

HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"

_TIERS: dict[str, str] = {
    # Execution path: can change what the model computes.
    "speculative_draft_mtp": HIGH,
    "speculative_draft_simple": HIGH,
    "speculative_draft_model": HIGH,
    "speculative_draft_max_tokens": HIGH,
    "speculative_draft_min_tokens": HIGH,
    "speculative_draft_min_continue_probability": HIGH,
    "num_experts": HIGH,
    "num_cpu_expert_layers_ratio": HIGH,
    "llama_k_cache_quantization_type": HIGH,
    "llama_v_cache_quantization_type": HIGH,
    "use_fp16_for_kv_cache": HIGH,
    # Memory/runtime path: placement and kernels.
    "offload_kv_cache_to_gpu": MEDIUM,
    "gpu_ratio": MEDIUM,
    "flash_attention": MEDIUM,
    "context_checkpoints": MEDIUM,
    "unified_kv_cache": MEDIUM,
    "context_length": MEDIUM,
    # Scheduling/memory management: should not change semantics.
    "eval_batch_size": LOW,
    "physical_batch_size": LOW,
    "parallel": LOW,
    "try_mmap": LOW,
    "keep_model_in_memory": LOW,
    "n_threads": LOW,
    "cpu_threads": LOW,
    "cpu_thread_pool_size": LOW,
}


def tier_of(canonical: str) -> str:
    """Risk tier for one parameter (LOW default for unknown keys)."""
    return _TIERS.get(canonical, LOW)
