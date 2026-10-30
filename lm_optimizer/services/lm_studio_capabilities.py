"""P1 capability findings for this LM Studio build (probed 2026-10-21).

Probed live against POST /api/v1/models/load with a fake model key.
A key is SUPPORTED when the server answers 404 (model_not_found = key
recognized) instead of 400 unrecognized_keys.

RESULT
------
Load-time parameters SUPPORTED by the REST v1 API:
  context_length, flash_attention, offload_kv_cache_to_gpu,
  eval_batch_size, physical_batch_size, parallel, context_checkpoints,
  num_experts, reasoning_budget_message, speculative_draft_max_tokens,
  speculative_draft_min_tokens, speculative_draft_min_continue_probability,
  speculative_draft_mtp, speculative_draft_simple, speculative_draft_model

NOT supported by the REST v1 API (400 unrecognized_keys):
  - KV cache quantization in ANY spelling:
      llama_k_cache_quantization_type, llama_v_cache_quantization_type,
      k_cache_quantization_type, v_cache_quantization_type,
      kv_cache_quantization_type, cache_type_k, cache_type_v,
      kv_cache_quantization
  - CPU threads in ANY spelling:
      cpu_threads, threads, num_threads, n_threads
  - mmap / keep-in-memory:
      keep_model_in_memory, try_mmap, mmap

Consequences for the optimiser
------------------------------
1. KV cache quantization is NOT a REST-visible lever on this build. The
   user's KV quantization is set in the LM Studio GUI, which the runtime
   applies independently of the REST load request. The optimizer must
   therefore RECORD it (as an observed runtime fact) and must NOT claim
   it as a searched dimension.
2. CPU threads are not settable through REST -> the optimizer cannot
   sweep them. They are a fixed runtime fact for the process.
3. mmap / keep-in-memory: not settable through REST either.

These three are therefore recorded as "runtime facts" (observed, not
searched) instead of being added to the load search space. Adding them
to the search space would produce configs the server silently ignores,
which the master spec explicitly forbids (requested vs applied config).
"""
from __future__ import annotations

# Keys probed and confirmed SUPPORTED (404 model_not_found) by the live server.
REST_SUPPORTED_LOAD_KEYS: frozenset[str] = frozenset(
    {
        "context_length",
        "flash_attention",
        "offload_kv_cache_to_gpu",
        "eval_batch_size",
        "physical_batch_size",
        "parallel",
        "context_checkpoints",
        "num_experts",
        "reasoning_budget_message",
        "speculative_draft_max_tokens",
        "speculative_draft_min_tokens",
        "speculative_draft_min_continue_probability",
        "speculative_draft_mtp",
        "speculative_draft_simple",
        "speculative_draft_model",
    }
)

# Probed and confirmed UNSUPPORTED (400 unrecognized_keys) by the live server.
REST_UNSUPPORTED_LOAD_KEYS: frozenset[str] = frozenset(
    {
        # KV cache quantization (all spellings tried)
        "llama_k_cache_quantization_type",
        "llama_v_cache_quantization_type",
        "k_cache_quantization_type",
        "v_cache_quantization_type",
        "kv_cache_quantization_type",
        "cache_type_k",
        "cache_type_v",
        "kv_cache_quantization",
        # CPU threads (all spellings tried)
        "cpu_threads",
        "threads",
        "num_threads",
        "n_threads",
        # mmap / keep-in-memory
        "keep_model_in_memory",
        "try_mmap",
        "mmap",
    }
)

# Runtime facts: observable, not settable through the REST load request.
# Recorded in the report so a reader knows they were NOT searched.
RUNTIME_ONLY_FACTS: tuple[str, ...] = (
    "kv_cache_quantization",
    "cpu_threads",
    "mmap",
    "keep_model_in_memory",
)
