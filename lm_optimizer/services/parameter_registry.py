"""Central parameter registry: canonical metadata + support surfaces.

Single source of truth. Statuses use LIVE evidence first:
- rest=supported/unsupported: verified request-by-request on the live endpoint
  (unknown keys 400 without loading; accepted keys confirmed via loads/echo).
- sdk=schema-known: present in the official SDK/runtime schema surface, but NOT
  exercised through the SDK by this optimizer (programmatically unverified).
- cli=supported: verified via `lms <cmd> --help` and live loads.
- Never inject undocumented keys; never modify LM Studio internals.

Docs table docs/LM_STUDIO_PARAMETER_MATRIX.md is GENERATED from this module
(see scripts/generate_parameter_matrix.py) so docs cannot drift from code.
"""

from dataclasses import dataclass

# Support status vocabulary (audit section 25 + schema-known extension).
SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
DETECTED_ONLY = "detected-only"
EXPERIMENTAL = "experimental"
VERSION_DEPENDENT = "version-dependent"
NOT_VERIFIED = "not-verified"
SCHEMA_KNOWN = "schema-known"


@dataclass(frozen=True)
class ParameterSpec:
    """Canonical metadata for one runtime parameter."""

    canonical: str
    display: str
    category: str
    backend: str  # any | llama.cpp | mlx | vllm | yuzu | transformers | ...
    phase: str  # load | inference
    dtype: str
    valid: str
    experimental: bool = False
    hw_dependent: bool = False
    model_dependent: bool = False
    rest: str = NOT_VERIFIED
    sdk: str = NOT_VERIFIED
    cli: str = NOT_VERIFIED
    schema: str = NOT_VERIFIED
    optimizer_enabled: bool = False
    default_behavior: str = ""
    failure_behavior: str = ""


def _s(  # noqa: PLR0913, PLR0917 - data DSL: one row per parameter by design
    canonical: str,
    display: str,
    category: str,
    backend: str = "any",
    phase: str = "load",
    dtype: str = "",
    valid: str = "",
    experimental: bool = False,
    hw_dependent: bool = False,
    model_dependent: bool = False,
    rest: str = NOT_VERIFIED,
    sdk: str = NOT_VERIFIED,
    cli: str = NOT_VERIFIED,
    schema: str = NOT_VERIFIED,
    optimizer_enabled: bool = False,
    default_behavior: str = "",
    failure_behavior: str = "",
) -> ParameterSpec:
    return ParameterSpec(
        canonical=canonical,
        display=display,
        category=category,
        backend=backend,
        phase=phase,
        dtype=dtype,
        valid=valid,
        experimental=experimental,
        hw_dependent=hw_dependent,
        model_dependent=model_dependent,
        rest=rest,
        sdk=sdk,
        cli=cli,
        schema=schema,
        optimizer_enabled=optimizer_enabled,
        default_behavior=default_behavior,
        failure_behavior=failure_behavior,
    )


# NOTE: rest=supported entries were verified live (2026-09-19/21) against the
# bundled server; rejected ones returned HTTP 400 unrecognized_keys.
PARAMETERS: list[ParameterSpec] = [
    # ---- Native REST load (verified supported) ----
    _s(
        "model",
        "Model key",
        "identity",
        phase="load",
        dtype="string",
        rest=SUPPORTED,
        optimizer_enabled=False,
        default_behavior="Explicit per run.",
        failure_behavior="404 model_not_found.",
    ),
    _s(
        "context_length",
        "Context length",
        "memory",
        dtype="integer",
        valid=">= 512",
        hw_dependent=True,
        model_dependent=True,
        rest=SUPPORTED,
        sdk=SCHEMA_KNOWN,
        cli=SUPPORTED,
        optimizer_enabled=True,
        default_behavior="Model max unless capped.",
        failure_behavior="OOM/refusal recorded; ladder stops that path.",
    ),
    _s(
        "eval_batch_size",
        "Eval (logical) batch size",
        "batching",
        dtype="integer",
        rest=SUPPORTED,
        sdk=SCHEMA_KNOWN,
        optimizer_enabled=True,
        default_behavior="Server default 2048; stage 4 searches 64-1024.",
        failure_behavior="Recorded as config failure.",
    ),
    _s(
        "flash_attention",
        "Flash Attention",
        "compute",
        dtype="boolean",
        hw_dependent=True,
        model_dependent=True,
        rest=SUPPORTED,
        sdk=SCHEMA_KNOWN,
        optimizer_enabled=True,
        default_behavior="On. Auto-pruned off when server requires it (non-F16 KV).",
        failure_behavior="'requires flash attention' load error; flash=False pruned.",
    ),
    _s(
        "num_experts",
        "Active experts (MoE)",
        "moe",
        dtype="integer",
        model_dependent=True,
        rest=SUPPORTED,
        sdk=SCHEMA_KNOWN,
        optimizer_enabled=True,
        default_behavior="Only set for MoE models.",
        failure_behavior="Non-MoE 400 possible if forced; gated on is_moe.",
    ),
    _s(
        "offload_kv_cache_to_gpu",
        "KV cache placement",
        "memory",
        dtype="boolean",
        hw_dependent=True,
        rest=SUPPORTED,
        optimizer_enabled=True,
        default_behavior="GPU unless --kv cpu or no GPU.",
        failure_behavior="Recorded; CPU fallback for big models.",
    ),
    _s(
        "physical_batch_size",
        "Physical batch size",
        "batching",
        dtype="integer",
        rest=SUPPORTED,
        sdk=SCHEMA_KNOWN,
        optimizer_enabled=True,
        default_behavior="Server default 512; stage 4 searches 256-1024. "
        "Empirically verified on the live server; treat as version-dependent elsewhere.",
        failure_behavior="Recorded as config failure.",
    ),
    _s(
        "parallel",
        "Max concurrency (parallel sessions)",
        "batching",
        dtype="integer",
        rest=SUPPORTED,
        sdk=SCHEMA_KNOWN,
        cli=SUPPORTED,
        optimizer_enabled=True,
        default_behavior="Server default 4; stage 4 searches 1-4 (single-user). "
        "Empirically verified on the live server; treat as version-dependent elsewhere.",
        failure_behavior="Recorded as config failure.",
    ),
    _s(
        "context_checkpoints",
        "Context checkpoints",
        "memory",
        dtype="integer",
        rest=SUPPORTED,
        optimizer_enabled=True,
        default_behavior="Server default 32; stage 4 searches 16-64. "
        "Empirically verified on the live server; treat as version-dependent elsewhere.",
        failure_behavior="Recorded as config failure.",
    ),
    _s(
        "llama_cpp_arguments_override",
        "Llama.cpp Arguments Override (GUI section)",
        "introspection",
        dtype="free-form",
        default_behavior="GUI-visible section. Programmatic control UNKNOWN: never "
        "modify LM Studio internals, never inject undocumented REST keys. Upgrades "
        "only via a verified official path through ParameterControlAdapter.",
        failure_behavior="Not programmatically controlled.",
    ),
    _s(
        "reasoning_budget_message",
        "Reasoning budget message",
        "reasoning",
        dtype="string",
        valid="model-dependent strings, '' = default",
        model_dependent=True,
        rest=SUPPORTED,
        optimizer_enabled=False,
        default_behavior="Pass-through only (not searched).",
        failure_behavior="400 invalid_type on wrong type.",
    ),
    _s(
        "speculative_draft_max_tokens",
        "Speculative draft max tokens",
        "speculative",
        dtype="integer",
        rest=SUPPORTED,
        sdk=SCHEMA_KNOWN,
        cli=SUPPORTED,
        optimizer_enabled=False,
        default_behavior="Server default 3; ineffective without a draft model.",
        failure_behavior="Recorded as config failure.",
    ),
    _s(
        "echo_load_config",
        "Echo applied config in load response",
        "introspection",
        dtype="boolean",
        rest=SUPPORTED,
        optimizer_enabled=False,
        default_behavior="Sent true on every load; falls back to instances echo.",
        failure_behavior="Falls back to GET instances echo.",
    ),
    # ---- Native REST load (verified rejected) ----
    _s(
        "gpu_ratio",
        "GPU offload ratio (layers fraction)",
        "memory",
        dtype="float 0-1",
        hw_dependent=True,
        rest=UNSUPPORTED,
        sdk=SCHEMA_KNOWN,
        cli=SUPPORTED,
        optimizer_enabled=False,
        default_behavior="Never sent via REST; CLI `lms load --gpu` when requested.",
        failure_behavior="Would 400 unrecognized_keys; filtered before send.",
    ),
    _s(
        "rope_freq_base",
        "RoPE frequency base",
        "context-extension",
        dtype="float",
        experimental=True,
        rest=UNSUPPORTED,
        sdk=SCHEMA_KNOWN,
        default_behavior="Experimental gate only (enable_rope).",
        failure_behavior="Would 400; filtered.",
    ),
    _s(
        "rope_freq_scale",
        "RoPE frequency scale",
        "context-extension",
        dtype="float",
        experimental=True,
        rest=UNSUPPORTED,
        sdk=SCHEMA_KNOWN,
        default_behavior="Experimental gate only.",
        failure_behavior="Would 400; filtered.",
    ),
    _s(
        "keep_model_in_memory",
        "Keep model in memory",
        "memory",
        dtype="boolean",
        rest=UNSUPPORTED,
        sdk=SCHEMA_KNOWN,
        default_behavior="GUI/CLI-only; test only on spill/RAM pressure.",
        failure_behavior="Would 400; filtered.",
    ),
    _s(
        "seed",
        "Seed (load)",
        "sampling",
        dtype="integer",
        rest=UNSUPPORTED,
        sdk=SCHEMA_KNOWN,
        default_behavior="Not used for loads.",
        failure_behavior="Would 400; filtered.",
    ),
    _s(
        "use_fp16_for_kv_cache",
        "FP16 KV cache",
        "memory",
        dtype="boolean",
        rest=UNSUPPORTED,
        sdk=SCHEMA_KNOWN,
        default_behavior="GUI-only (Q4 starter preset).",
        failure_behavior="Would 400; filtered.",
    ),
    _s(
        "try_mmap",
        "Memory-map weights",
        "storage",
        dtype="boolean",
        rest=UNSUPPORTED,
        sdk=SCHEMA_KNOWN,
        default_behavior="Storage-aware only; not searched per model.",
        failure_behavior="Would 400; filtered.",
    ),
    _s(
        "llama_k_cache_quantization_type",
        "K-cache quantization",
        "memory",
        dtype="string",
        rest=UNSUPPORTED,
        sdk=SCHEMA_KNOWN,
        default_behavior="GUI-only (Q4 starter).",
        failure_behavior="Would 400; filtered.",
    ),
    _s(
        "llama_v_cache_quantization_type",
        "V-cache quantization",
        "memory",
        dtype="string",
        rest=UNSUPPORTED,
        sdk=SCHEMA_KNOWN,
        default_behavior="GUI-only (Q4 starter; requires flash).",
        failure_behavior="Would 400; filtered.",
    ),
    _s(
        "n_threads",
        "CPU thread count (load)",
        "compute",
        dtype="integer",
        hw_dependent=True,
        rest=UNSUPPORTED,
        default_behavior="GUI-only (set to logical max).",
        failure_behavior="Would 400; filtered.",
    ),
    _s(
        "unified_kv_cache",
        "Unified KV cache",
        "memory",
        dtype="boolean",
        rest=UNSUPPORTED,
        default_behavior="Not searched (single-user).",
        failure_behavior="Would 400; filtered.",
    ),
    _s(
        "ttl",
        "Model TTL",
        "memory",
        dtype="integer seconds",
        rest=UNSUPPORTED,
        cli=SUPPORTED,
        default_behavior="CLI-only (`lms load --ttl`).",
        failure_behavior="Would 400 via REST; filtered.",
    ),
    _s(
        "auto_fit",
        "Auto fit (load mode)",
        "memory",
        dtype="boolean",
        schema=SCHEMA_KNOWN,
        default_behavior="Baseline/compare mode only; not searched.",
        failure_behavior="Unverified control path.",
    ),
    _s(
        "auto_fit_min_context_length",
        "Auto fit min context",
        "memory",
        dtype="integer",
        schema=SCHEMA_KNOWN,
        default_behavior="Discovery hint only.",
        failure_behavior="Unverified control path.",
    ),
    _s(
        "num_layers",
        "Layer count override",
        "memory",
        dtype="integer",
        rest=UNSUPPORTED,
        default_behavior="Not used.",
        failure_behavior="Would 400.",
    ),
    _s(
        "gpu",
        "GPU selector (load)",
        "memory",
        dtype="string/integer",
        hw_dependent=True,
        rest=UNSUPPORTED,
        cli=SUPPORTED,
        default_behavior="CLI `lms load --gpu` handles placement.",
        failure_behavior="Would 400 via REST; filtered.",
    ),
    # ---- SDK/runtime schema-known (programmatically unverified) ----
    _s(
        "gpu.offload_ratio",
        "GPU offload ratio (SDK)",
        "memory",
        dtype="float|string",
        hw_dependent=True,
        sdk=SCHEMA_KNOWN,
        cli=SUPPORTED,
        default_behavior="CLI path preferred when ratio requested.",
        failure_behavior="CLI errors surface with guidance.",
    ),
    _s(
        "gpu.main",
        "Main GPU",
        "multi-gpu",
        dtype="integer",
        hw_dependent=True,
        sdk=SCHEMA_KNOWN,
        default_behavior="Detected-but-not-controlled.",
        failure_behavior="Single-GPU assumed unless exposed.",
    ),
    _s(
        "gpu.disabled",
        "Disabled GPUs",
        "multi-gpu",
        dtype="list",
        hw_dependent=True,
        sdk=SCHEMA_KNOWN,
        default_behavior="Detected-but-not-controlled.",
        failure_behavior="N/A.",
    ),
    _s(
        "gpu.split_strategy",
        "GPU split strategy",
        "multi-gpu",
        dtype="string",
        hw_dependent=True,
        sdk=SCHEMA_KNOWN,
        default_behavior="Detected-but-not-controlled.",
        failure_behavior="N/A.",
    ),
    _s(
        "cpu_thread_pool_size",
        "CPU thread pool size (llama runtime)",
        "compute",
        dtype="integer",
        hw_dependent=True,
        sdk=SCHEMA_KNOWN,
        schema=SCHEMA_KNOWN,
        default_behavior="Separate from inference cpuThreads; not searched.",
        failure_behavior="Unverified control path.",
    ),
    _s(
        "num_cpu_expert_layers_ratio",
        "CPU MoE expert layer ratio",
        "moe",
        dtype="float",
        model_dependent=True,
        experimental=True,
        sdk=SCHEMA_KNOWN,
        schema=SCHEMA_KNOWN,
        default_behavior="MoE-only, experimental, not searched.",
        failure_behavior="Unverified control path.",
    ),
    _s(
        "gpu_strict_vram_cap",
        "Strict VRAM cap",
        "memory",
        dtype="boolean/integer",
        hw_dependent=True,
        experimental=True,
        sdk=SCHEMA_KNOWN,
        default_behavior="Never auto-toggled; experimental opt-in only.",
        failure_behavior="Unverified control path.",
    ),
    _s(
        "try_direct_io",
        "Direct IO",
        "storage",
        dtype="boolean",
        sdk=SCHEMA_KNOWN,
        schema=SCHEMA_KNOWN,
        default_behavior="Storage-aware only.",
        failure_behavior="Unverified control path.",
    ),
    _s(
        "speculative_draft_model",
        "Speculative draft model",
        "speculative",
        dtype="string/model",
        sdk=SCHEMA_KNOWN,
        cli=SUPPORTED,
        default_behavior="Only when a draft model exists (discovered).",
        failure_behavior="Without draft model: no effect.",
    ),
    _s(
        "speculative_draft_min_tokens",
        "Speculative draft min tokens",
        "speculative",
        dtype="integer",
        rest=SUPPORTED,
        sdk=SCHEMA_KNOWN,
        cli=SUPPORTED,
        default_behavior="Server default 0; not searched.",
        failure_behavior="Recorded as config failure.",
    ),
    _s(
        "speculative_draft_min_continue_probability",
        "Speculative min continue probability",
        "speculative",
        dtype="float",
        rest=SUPPORTED,
        sdk=SCHEMA_KNOWN,
        cli=SUPPORTED,
        default_behavior="Server default 0; not searched.",
        failure_behavior="Recorded as config failure.",
    ),
    _s(
        "speculative_draft_mtp",
        "Draft MTP speculative decoding",
        "speculative",
        dtype="boolean",
        model_dependent=True,
        rest=SUPPORTED,
        sdk=SCHEMA_KNOWN,
        cli=SUPPORTED,
        default_behavior="Only when model supports MTP.",
        failure_behavior="Recorded as config failure.",
    ),
    _s(
        "speculative_draft_simple",
        "Draft simple speculative decoding",
        "speculative",
        dtype="boolean",
        rest=SUPPORTED,
        sdk=SCHEMA_KNOWN,
        cli=SUPPORTED,
        default_behavior="Off unless draft model configured.",
        failure_behavior="Recorded as config failure.",
    ),
    _s(
        "speculative_draft_dflash_sidecar",
        "Draft DFlash sidecar",
        "speculative",
        dtype="boolean",
        experimental=True,
        sdk=SCHEMA_KNOWN,
        default_behavior="Not searched.",
        failure_behavior="Unverified.",
    ),
    _s(
        "speculative_draft_dspark_sidecar",
        "Draft DSpark sidecar",
        "speculative",
        dtype="boolean",
        experimental=True,
        sdk=SCHEMA_KNOWN,
        default_behavior="Not searched.",
        failure_behavior="Unverified.",
    ),
    _s(
        "speculative_draft_mtp_sidecar",
        "Draft MTP sidecar",
        "speculative",
        dtype="boolean",
        experimental=True,
        sdk=SCHEMA_KNOWN,
        default_behavior="Not searched.",
        failure_behavior="Unverified.",
    ),
    # ---- Native REST chat (verified supported) ----
    _s(
        "temperature",
        "Temperature",
        "sampling",
        phase="inference",
        dtype="float",
        rest=SUPPORTED,
        optimizer_enabled=False,
        default_behavior="Frozen per suite/style; precision probe tests 0.2.",
        failure_behavior="Case fails; recorded.",
    ),
    _s(
        "top_p",
        "Top-p",
        "sampling",
        phase="inference",
        dtype="float",
        rest=SUPPORTED,
        default_behavior="Server default in benchmarks; per-model defaults in .md.",
        failure_behavior="Case fails; recorded.",
    ),
    _s(
        "top_k",
        "Top-k",
        "sampling",
        phase="inference",
        dtype="integer",
        rest=SUPPORTED,
        default_behavior="Server default in benchmarks.",
        failure_behavior="Case fails; recorded.",
    ),
    _s(
        "min_p",
        "Min-p",
        "sampling",
        phase="inference",
        dtype="float",
        rest=SUPPORTED,
        default_behavior="Server default in benchmarks.",
        failure_behavior="Case fails; recorded.",
    ),
    _s(
        "presence_penalty",
        "Presence penalty",
        "sampling",
        phase="inference",
        dtype="float",
        rest=SUPPORTED,
        default_behavior="Server default in benchmarks.",
        failure_behavior="Case fails; recorded.",
    ),
    _s(
        "reasoning",
        "Reasoning effort",
        "reasoning",
        phase="inference",
        dtype="string",
        valid="model-dependent: off/on or full range",
        model_dependent=True,
        rest=SUPPORTED,
        optimizer_enabled=False,
        default_behavior="off for benchmarks; auto-omitted where unsupported.",
        failure_behavior="400 invalid_value -> retry omitted (remembered).",
    ),
    _s(
        "max_output_tokens",
        "Max output tokens",
        "sampling",
        phase="inference",
        dtype="integer",
        rest=SUPPORTED,
        optimizer_enabled=False,
        default_behavior="Suite caps (256/768/1024/768/256) bounded by ctx/4.",
        failure_behavior="Truncation fails quality checks (by design).",
    ),
    # ---- Native REST chat (verified rejected) ----
    _s(
        "repetition_penalty",
        "Repetition penalty",
        "sampling",
        phase="inference",
        rest=UNSUPPORTED,
        default_behavior="Not sent.",
        failure_behavior="Would 400; filtered.",
    ),
    _s(
        "frequency_penalty",
        "Frequency penalty",
        "sampling",
        phase="inference",
        rest=UNSUPPORTED,
        default_behavior="Not sent.",
        failure_behavior="Would 400; filtered.",
    ),
    _s(
        "typical_p",
        "Typical sampling",
        "sampling",
        phase="inference",
        rest=UNSUPPORTED,
        default_behavior="Not sent.",
        failure_behavior="Would 400; filtered.",
    ),
    _s(
        "mirostat",
        "Mirostat",
        "sampling",
        phase="inference",
        rest=UNSUPPORTED,
        default_behavior="Not sent.",
        failure_behavior="Would 400; filtered.",
    ),
    _s(
        "stop",
        "Stop strings",
        "sampling",
        phase="inference",
        rest=UNSUPPORTED,
        default_behavior="Cannot enforce server-side; checks tolerate fences.",
        failure_behavior="Would 400; filtered.",
    ),
    _s(
        "seed_chat",
        "Seed (chat)",
        "sampling",
        phase="inference",
        rest=UNSUPPORTED,
        default_behavior="Not used.",
        failure_behavior="Would 400; filtered.",
    ),
    # ---- Inference schema-known (unverified live) ----
    _s(
        "cpu_threads",
        "CPU threads (inference-time)",
        "compute",
        phase="inference",
        dtype="integer",
        hw_dependent=True,
        schema=SCHEMA_KNOWN,
        default_behavior="Distinct from cpu_thread_pool_size; GUI-only in practice.",
        failure_behavior="Unverified control path; never injected blindly.",
    ),
    _s(
        "tool_call_stop_strings",
        "Tool-call stop strings",
        "sampling",
        phase="inference",
        schema=SCHEMA_KNOWN,
        default_behavior="Not used.",
        failure_behavior="Unverified.",
    ),
    _s(
        "context_overflow_policy",
        "Context overflow policy",
        "memory",
        phase="inference",
        schema=SCHEMA_KNOWN,
        default_behavior="Not used.",
        failure_behavior="Unverified.",
    ),
    _s(
        "structured",
        "Structured output mode",
        "sampling",
        phase="inference",
        schema=SCHEMA_KNOWN,
        default_behavior="Prompt-level JSON instruction instead.",
        failure_behavior="Unverified.",
    ),
    _s(
        "repeat_penalty_xtc",
        "XTC sampling",
        "sampling",
        phase="inference",
        schema=SCHEMA_KNOWN,
        default_behavior="Not used.",
        failure_behavior="Unverified.",
    ),
    _s(
        "tail_free_sampling",
        "Tail-free sampling",
        "sampling",
        phase="inference",
        schema=SCHEMA_KNOWN,
        default_behavior="Not used.",
        failure_behavior="Unverified.",
    ),
    _s(
        "typical_sampling",
        "Locally typical sampling",
        "sampling",
        phase="inference",
        schema=SCHEMA_KNOWN,
        default_behavior="Not used (typical_p rejected via REST).",
        failure_behavior="Unverified.",
    ),
    _s(
        "logit_bias",
        "Logit bias",
        "sampling",
        phase="inference",
        schema=SCHEMA_KNOWN,
        default_behavior="Not used.",
        failure_behavior="Unverified.",
    ),
    _s(
        "draft_model_inference",
        "Draft model (inference)",
        "speculative",
        phase="inference",
        schema=SCHEMA_KNOWN,
        default_behavior="Only via load-time draft config when discovered.",
        failure_behavior="Unverified.",
    ),
    # ---- llama.cpp catalog (NOT programmatically exposed; reference only) ----
    _s(
        "llama.threads",
        "llama threads",
        "compute",
        backend="llama.cpp",
        dtype="integer",
        rest=UNSUPPORTED,
        schema=SCHEMA_KNOWN,
        default_behavior="See cpu_thread_pool_size / n_threads (GUI-only).",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.threads_batch",
        "llama batch threads",
        "compute",
        backend="llama.cpp",
        rest=UNSUPPORTED,
        default_behavior="Not exposed.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.ctx_size",
        "llama ctx size",
        "memory",
        backend="llama.cpp",
        rest=UNSUPPORTED,
        default_behavior="Use context_length.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.batch_size",
        "llama batch size",
        "batching",
        backend="llama.cpp",
        rest=UNSUPPORTED,
        default_behavior="Use eval_batch_size.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.ubatch_size",
        "llama micro batch",
        "batching",
        backend="llama.cpp",
        rest=UNSUPPORTED,
        default_behavior="Use physical_batch_size.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.flash_attn",
        "llama flash attention",
        "compute",
        backend="llama.cpp",
        rest=UNSUPPORTED,
        default_behavior="Use flash_attention.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.rope_scaling",
        "llama rope scaling",
        "context-extension",
        backend="llama.cpp",
        experimental=True,
        rest=UNSUPPORTED,
        default_behavior="Experimental gate only.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.rope_scale",
        "llama rope scale",
        "context-extension",
        backend="llama.cpp",
        experimental=True,
        rest=UNSUPPORTED,
        default_behavior="Experimental gate only.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.rope_freq_base",
        "llama rope freq base",
        "context-extension",
        backend="llama.cpp",
        experimental=True,
        rest=UNSUPPORTED,
        default_behavior="Experimental gate only.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.rope_freq_scale",
        "llama rope freq scale",
        "context-extension",
        backend="llama.cpp",
        experimental=True,
        rest=UNSUPPORTED,
        default_behavior="Experimental gate only.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.yarn_orig_ctx",
        "YaRN original context",
        "context-extension",
        backend="llama.cpp",
        experimental=True,
        rest=UNSUPPORTED,
        default_behavior="Experimental gate only.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.yarn_ext_factor",
        "YaRN extension factor",
        "context-extension",
        backend="llama.cpp",
        experimental=True,
        rest=UNSUPPORTED,
        default_behavior="Experimental gate only.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.yarn_attn_factor",
        "YaRN attention factor",
        "context-extension",
        backend="llama.cpp",
        experimental=True,
        rest=UNSUPPORTED,
        default_behavior="Experimental gate only.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.yarn_beta_slow",
        "YaRN beta slow",
        "context-extension",
        backend="llama.cpp",
        experimental=True,
        rest=UNSUPPORTED,
        default_behavior="Experimental gate only.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.yarn_beta_fast",
        "YaRN beta fast",
        "context-extension",
        backend="llama.cpp",
        experimental=True,
        rest=UNSUPPORTED,
        default_behavior="Experimental gate only.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.kv_offload",
        "llama KV offload",
        "memory",
        backend="llama.cpp",
        rest=UNSUPPORTED,
        default_behavior="Use offload_kv_cache_to_gpu.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.cache_type_k",
        "llama K cache type",
        "memory",
        backend="llama.cpp",
        rest=UNSUPPORTED,
        default_behavior="GUI-only K quant.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.cache_type_v",
        "llama V cache type",
        "memory",
        backend="llama.cpp",
        rest=UNSUPPORTED,
        default_behavior="GUI-only V quant.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.parallel",
        "llama parallel sequences",
        "batching",
        backend="llama.cpp",
        rest=UNSUPPORTED,
        default_behavior="Use parallel.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.mlock",
        "llama mlock",
        "memory",
        backend="llama.cpp",
        rest=UNSUPPORTED,
        default_behavior="See keep_model_in_memory (GUI-only).",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.mmap",
        "llama mmap",
        "storage",
        backend="llama.cpp",
        rest=UNSUPPORTED,
        default_behavior="See try_mmap (GUI-only).",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.numa",
        "llama NUMA",
        "compute",
        backend="llama.cpp",
        rest=UNSUPPORTED,
        default_behavior="Not exposed.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.device",
        "llama device",
        "memory",
        backend="llama.cpp",
        hw_dependent=True,
        rest=UNSUPPORTED,
        default_behavior="CLI --gpu handles placement.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.cpu_moe",
        "llama CPU MoE",
        "moe",
        backend="llama.cpp",
        model_dependent=True,
        rest=UNSUPPORTED,
        default_behavior="See num_cpu_expert_layers_ratio.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.n_cpu_moe",
        "llama N CPU MoE",
        "moe",
        backend="llama.cpp",
        model_dependent=True,
        rest=UNSUPPORTED,
        default_behavior="See num_cpu_expert_layers_ratio.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.swa_full",
        "llama SWA full",
        "memory",
        backend="llama.cpp",
        rest=UNSUPPORTED,
        default_behavior="Not exposed.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.kv_unified",
        "llama unified KV",
        "memory",
        backend="llama.cpp",
        rest=UNSUPPORTED,
        default_behavior="See unified_kv_cache.",
        failure_behavior="Not exposed; never injected.",
    ),
    _s(
        "llama.defrag_thold",
        "llama defrag threshold",
        "memory",
        backend="llama.cpp",
        rest=UNSUPPORTED,
        default_behavior="Not exposed.",
        failure_behavior="Not exposed; never injected.",
    ),
    # ---- Backend-gated (only when that backend serves the model) ----
    _s(
        "mlx.auto_fit",
        "MLX auto fit",
        "memory",
        backend="mlx",
        rest=UNSUPPORTED,
        default_behavior="Hidden unless backend is MLX.",
        failure_behavior="Not applicable to GGUF/llama.cpp.",
    ),
    _s(
        "mlx.disk_cache",
        "MLX disk cache",
        "storage",
        backend="mlx",
        rest=UNSUPPORTED,
        default_behavior="Hidden unless backend is MLX.",
        failure_behavior="Not applicable to GGUF/llama.cpp.",
    ),
    _s(
        "mlx.kv_cache_quantization",
        "MLX KV cache quantization",
        "memory",
        backend="mlx",
        rest=UNSUPPORTED,
        default_behavior="Hidden unless backend is MLX.",
        failure_behavior="Not applicable to GGUF/llama.cpp.",
    ),
    _s(
        "vllm.gpu_memory_utilization",
        "vLLM GPU memory utilization",
        "memory",
        backend="vllm",
        hw_dependent=True,
        rest=UNSUPPORTED,
        default_behavior="Hidden unless backend is vLLM. NOT llama.cpp offload ratio.",
        failure_behavior="Not applicable to GGUF/llama.cpp.",
    ),
    _s(
        "vllm.reasoning_parser",
        "vLLM reasoning parser",
        "reasoning",
        backend="vllm",
        rest=UNSUPPORTED,
        default_behavior="Hidden unless backend is vLLM.",
        failure_behavior="Not applicable to GGUF/llama.cpp.",
    ),
    _s(
        "vllm.tool_call_parser",
        "vLLM tool call parser",
        "sampling",
        backend="vllm",
        rest=UNSUPPORTED,
        default_behavior="Hidden unless backend is vLLM.",
        failure_behavior="Not applicable to GGUF/llama.cpp.",
    ),
    _s(
        "yuzu.auto_fit",
        "Yuzu auto fit",
        "memory",
        backend="yuzu",
        rest=UNSUPPORTED,
        default_behavior="Hidden unless backend is yuzu.",
        failure_behavior="Not applicable to GGUF/llama.cpp.",
    ),
]


# Model format string -> backend family. No universal "all settings" table
# may be shown without resolving this first (audit section D).
BACKEND_BY_FORMAT = {
    "gguf": "llama.cpp",
    "mlx": "mlx",
    "safetensors": "transformers",
    "onnx": "onnx",
}


def detect_backend(model_format: str | None) -> str:
    """Backend family for a model format string (unknown when absent)."""
    if not model_format:
        return "unknown"
    return BACKEND_BY_FORMAT.get(model_format.lower(), "unknown")


# Lifecycle pipeline (audit section E): Detected -> Supported -> Controllable
# -> Applied -> Verified, else Unsupported. Detection alone never implies
# control ("llama.cpp has it" is not permission to send it).
def lifecycle_of(spec: ParameterSpec, applied: bool = False, verified: bool = False) -> str:
    """Pipeline stage for a parameter (+ optional run evidence)."""
    if verified and applied:
        return "Verified"
    if applied:
        return "Applied"
    if SUPPORTED in {spec.rest, spec.cli}:
        return "Controllable"
    if {spec.rest, spec.sdk, spec.cli, spec.schema} <= {UNSUPPORTED, NOT_VERIFIED}:
        return "Unsupported (backend-gated)" if spec.backend != "any" else "Unsupported"
    return "Detected"


_BY_CANONICAL: dict[str, ParameterSpec] = {p.canonical: p for p in PARAMETERS}


def get(canonical: str) -> ParameterSpec | None:
    """Look up one parameter by canonical name."""
    return _BY_CANONICAL.get(canonical)


def by_category(category: str) -> list[ParameterSpec]:
    """All parameters in a category."""
    return [p for p in PARAMETERS if p.category == category]


def for_optimizer() -> list[ParameterSpec]:
    """Parameters the optimizer is allowed to search."""
    return [p for p in PARAMETERS if p.optimizer_enabled]


def snapshot() -> dict:
    """Name -> {source-best surface, status} for DB recording."""
    out = {}
    for p in PARAMETERS:
        best, status = "unknown", NOT_VERIFIED
        # Source priority: REST > SDK > CLI > schema > catalog.
        for surface in ("rest", "sdk", "cli", "schema"):
            st = getattr(p, surface)
            if st in (SUPPORTED, UNSUPPORTED):
                best, status = surface.upper(), st
                break
        if best == "unknown":
            for surface in ("rest", "sdk", "cli", "schema"):
                st = getattr(p, surface)
                if st != NOT_VERIFIED:
                    best, status = surface.upper(), st
                    break
        out[p.canonical] = {
            "source": best,
            "status": status,
            "optimizer": p.optimizer_enabled,
            "experimental": p.experimental,
        }
    return out


def matrix_rows() -> list[dict]:
    """Flat rows for docs/UI tables."""
    rows = []
    for p in PARAMETERS:
        control = (
            "REST"
            if p.rest == SUPPORTED
            else (
                "CLI"
                if p.cli == SUPPORTED
                else ("SDK/schema" if {p.sdk, p.schema} & {SCHEMA_KNOWN} else "none")
            )
        )
        verified = (
            "yes"
            if {p.rest, p.cli} & {SUPPORTED}
            else ("no" if p.rest == UNSUPPORTED and p.cli != SUPPORTED else "partial")
        )
        rows.append(
            {
                "name": p.canonical,
                "display": p.display,
                "backend": p.backend,
                "phase": p.phase,
                "control": control,
                "verified": verified,
                "optimizer": "yes" if p.optimizer_enabled else "no",
                "experimental": "yes" if p.experimental else "no",
                "lifecycle": lifecycle_of(p),
            }
        )
    return rows
