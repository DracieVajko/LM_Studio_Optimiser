# LM Studio Parameter Matrix (generated)

Generated from `lm_optimizer/services/parameter_registry.py` — do not hand-edit; re-run `scripts/generate_parameter_matrix.py`.

Control: REST = native API verified live; CLI = `lms` flags verified; SDK/schema = known but programmatically unverified; none = not exposed. Verified: yes = apply+verify works; no = rejected/unsupported; partial = version- or model-dependent.

| Name | Meaning | Backend | API | CLI | SDK | Optimizer status | Experimental | Lifecycle | Verification method | Known limitations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model | Model key | any | supported | not-verified | not-verified | not searched | no | Controllable | live request-by-request + echo compare | 404 model_not_found. |
| context_length | Context length | any | supported | supported | schema-known | searched | no | Controllable | live request-by-request + echo compare | OOM/refusal recorded; ladder stops that path. |
| eval_batch_size | Eval (logical) batch size | any | supported | not-verified | schema-known | searched | no | Controllable | live request-by-request + echo compare | Recorded as config failure. |
| flash_attention | Flash Attention | any | supported | not-verified | schema-known | searched | no | Controllable | live request-by-request + echo compare | 'requires flash attention' load error; flash=False pruned. |
| num_experts | Active experts (MoE) | any | supported | not-verified | schema-known | searched | no | Controllable | live request-by-request + echo compare | Non-MoE 400 possible if forced; gated on is_moe. |
| offload_kv_cache_to_gpu | KV cache placement | any | supported | not-verified | not-verified | searched | no | Controllable | live request-by-request + echo compare | Recorded; CPU fallback for big models. |
| physical_batch_size | Physical batch size | any | supported | not-verified | schema-known | searched | no | Controllable | live request-by-request + echo compare | Recorded as config failure. |
| parallel | Max concurrency (parallel sessions) | any | supported | supported | schema-known | searched | no | Controllable | live request-by-request + echo compare | Recorded as config failure. |
| context_checkpoints | Context checkpoints | any | supported | not-verified | not-verified | searched | no | Controllable | live request-by-request + echo compare | Recorded as config failure. |
| llama_cpp_arguments_override | Llama.cpp Arguments Override (GUI section) | any | not-verified | not-verified | not-verified | not searched | no | Unsupported | schema/CLI inspection | Not programmatically controlled. |
| reasoning_budget_message | Reasoning budget message | any | supported | not-verified | not-verified | not searched | no | Controllable | live request-by-request + echo compare | 400 invalid_type on wrong type. |
| speculative_draft_max_tokens | Speculative draft max tokens | any | supported | supported | schema-known | not searched | no | Controllable | live request-by-request + echo compare | Recorded as config failure. |
| echo_load_config | Echo applied config in load response | any | supported | not-verified | not-verified | not searched | no | Controllable | live request-by-request + echo compare | Falls back to GET instances echo. |
| gpu_ratio | GPU offload ratio (layers fraction) | any | unsupported | supported | schema-known | not searched | no | Controllable | live request-by-request + echo compare | Would 400 unrecognized_keys; filtered before send. |
| rope_freq_base | RoPE frequency base | any | unsupported | not-verified | schema-known | not searched | yes | Detected | safe 400-probe | Would 400; filtered. |
| rope_freq_scale | RoPE frequency scale | any | unsupported | not-verified | schema-known | not searched | yes | Detected | safe 400-probe | Would 400; filtered. |
| keep_model_in_memory | Keep model in memory | any | unsupported | not-verified | schema-known | not searched | no | Detected | safe 400-probe | Would 400; filtered. |
| seed | Seed (load) | any | unsupported | not-verified | schema-known | not searched | no | Detected | safe 400-probe | Would 400; filtered. |
| use_fp16_for_kv_cache | FP16 KV cache | any | unsupported | not-verified | schema-known | not searched | no | Detected | safe 400-probe | Would 400; filtered. |
| try_mmap | Memory-map weights | any | unsupported | not-verified | schema-known | not searched | no | Detected | safe 400-probe | Would 400; filtered. |
| llama_k_cache_quantization_type | K-cache quantization | any | unsupported | not-verified | schema-known | not searched | no | Detected | safe 400-probe | Would 400; filtered. |
| llama_v_cache_quantization_type | V-cache quantization | any | unsupported | not-verified | schema-known | not searched | no | Detected | safe 400-probe | Would 400; filtered. |
| n_threads | CPU thread count (load) | any | unsupported | not-verified | not-verified | not searched | no | Unsupported | safe 400-probe | Would 400; filtered. |
| unified_kv_cache | Unified KV cache | any | unsupported | not-verified | not-verified | not searched | no | Unsupported | safe 400-probe | Would 400; filtered. |
| ttl | Model TTL | any | unsupported | supported | not-verified | not searched | no | Controllable | live request-by-request + echo compare | Would 400 via REST; filtered. |
| auto_fit | Auto fit (load mode) | any | not-verified | not-verified | not-verified | not searched | no | Detected | schema/CLI inspection | Unverified control path. |
| auto_fit_min_context_length | Auto fit min context | any | not-verified | not-verified | not-verified | not searched | no | Detected | schema/CLI inspection | Unverified control path. |
| num_layers | Layer count override | any | unsupported | not-verified | not-verified | not searched | no | Unsupported | safe 400-probe | Would 400. |
| gpu | GPU selector (load) | any | unsupported | supported | not-verified | not searched | no | Controllable | live request-by-request + echo compare | Would 400 via REST; filtered. |
| gpu.offload_ratio | GPU offload ratio (SDK) | any | not-verified | supported | schema-known | not searched | no | Controllable | live request-by-request + echo compare | CLI errors surface with guidance. |
| gpu.main | Main GPU | any | not-verified | not-verified | schema-known | not searched | no | Detected | schema/CLI inspection | Single-GPU assumed unless exposed. |
| gpu.disabled | Disabled GPUs | any | not-verified | not-verified | schema-known | not searched | no | Detected | schema/CLI inspection | N/A. |
| gpu.split_strategy | GPU split strategy | any | not-verified | not-verified | schema-known | not searched | no | Detected | schema/CLI inspection | N/A. |
| cpu_thread_pool_size | CPU thread pool size (llama runtime) | any | not-verified | not-verified | schema-known | not searched | no | Detected | schema/CLI inspection | Unverified control path. |
| num_cpu_expert_layers_ratio | CPU MoE expert layer ratio | any | not-verified | not-verified | schema-known | not searched | yes | Detected | schema/CLI inspection | Unverified control path. |
| gpu_strict_vram_cap | Strict VRAM cap | any | not-verified | not-verified | schema-known | not searched | yes | Detected | schema/CLI inspection | Unverified control path. |
| try_direct_io | Direct IO | any | not-verified | not-verified | schema-known | not searched | no | Detected | schema/CLI inspection | Unverified control path. |
| speculative_draft_model | Speculative draft model | any | not-verified | supported | schema-known | not searched | no | Controllable | live request-by-request + echo compare | Without draft model: no effect. |
| speculative_draft_min_tokens | Speculative draft min tokens | any | supported | supported | schema-known | not searched | no | Controllable | live request-by-request + echo compare | Recorded as config failure. |
| speculative_draft_min_continue_probability | Speculative min continue probability | any | supported | supported | schema-known | not searched | no | Controllable | live request-by-request + echo compare | Recorded as config failure. |
| speculative_draft_mtp | Draft MTP speculative decoding | any | supported | supported | schema-known | not searched | no | Controllable | live request-by-request + echo compare | Recorded as config failure. |
| speculative_draft_simple | Draft simple speculative decoding | any | supported | supported | schema-known | not searched | no | Controllable | live request-by-request + echo compare | Recorded as config failure. |
| speculative_draft_dflash_sidecar | Draft DFlash sidecar | any | not-verified | not-verified | schema-known | not searched | yes | Detected | schema/CLI inspection | Unverified. |
| speculative_draft_dspark_sidecar | Draft DSpark sidecar | any | not-verified | not-verified | schema-known | not searched | yes | Detected | schema/CLI inspection | Unverified. |
| speculative_draft_mtp_sidecar | Draft MTP sidecar | any | not-verified | not-verified | schema-known | not searched | yes | Detected | schema/CLI inspection | Unverified. |
| temperature | Temperature | any | supported | not-verified | not-verified | not searched | no | Controllable | live request-by-request + echo compare | Case fails; recorded. |
| top_p | Top-p | any | supported | not-verified | not-verified | not searched | no | Controllable | live request-by-request + echo compare | Case fails; recorded. |
| top_k | Top-k | any | supported | not-verified | not-verified | not searched | no | Controllable | live request-by-request + echo compare | Case fails; recorded. |
| min_p | Min-p | any | supported | not-verified | not-verified | not searched | no | Controllable | live request-by-request + echo compare | Case fails; recorded. |
| presence_penalty | Presence penalty | any | supported | not-verified | not-verified | not searched | no | Controllable | live request-by-request + echo compare | Case fails; recorded. |
| reasoning | Reasoning effort | any | supported | not-verified | not-verified | not searched | no | Controllable | live request-by-request + echo compare | 400 invalid_value -> retry omitted (remembered). |
| max_output_tokens | Max output tokens | any | supported | not-verified | not-verified | not searched | no | Controllable | live request-by-request + echo compare | Truncation fails quality checks (by design). |
| repetition_penalty | Repetition penalty | any | unsupported | not-verified | not-verified | not searched | no | Unsupported | safe 400-probe | Would 400; filtered. |
| frequency_penalty | Frequency penalty | any | unsupported | not-verified | not-verified | not searched | no | Unsupported | safe 400-probe | Would 400; filtered. |
| typical_p | Typical sampling | any | unsupported | not-verified | not-verified | not searched | no | Unsupported | safe 400-probe | Would 400; filtered. |
| mirostat | Mirostat | any | unsupported | not-verified | not-verified | not searched | no | Unsupported | safe 400-probe | Would 400; filtered. |
| stop | Stop strings | any | unsupported | not-verified | not-verified | not searched | no | Unsupported | safe 400-probe | Would 400; filtered. |
| seed_chat | Seed (chat) | any | unsupported | not-verified | not-verified | not searched | no | Unsupported | safe 400-probe | Would 400; filtered. |
| cpu_threads | CPU threads (inference-time) | any | not-verified | not-verified | not-verified | not searched | no | Detected | schema/CLI inspection | Unverified control path; never injected blindly. |
| tool_call_stop_strings | Tool-call stop strings | any | not-verified | not-verified | not-verified | not searched | no | Detected | schema/CLI inspection | Unverified. |
| context_overflow_policy | Context overflow policy | any | not-verified | not-verified | not-verified | not searched | no | Detected | schema/CLI inspection | Unverified. |
| structured | Structured output mode | any | not-verified | not-verified | not-verified | not searched | no | Detected | schema/CLI inspection | Unverified. |
| repeat_penalty_xtc | XTC sampling | any | not-verified | not-verified | not-verified | not searched | no | Detected | schema/CLI inspection | Unverified. |
| tail_free_sampling | Tail-free sampling | any | not-verified | not-verified | not-verified | not searched | no | Detected | schema/CLI inspection | Unverified. |
| typical_sampling | Locally typical sampling | any | not-verified | not-verified | not-verified | not searched | no | Detected | schema/CLI inspection | Unverified. |
| logit_bias | Logit bias | any | not-verified | not-verified | not-verified | not searched | no | Detected | schema/CLI inspection | Unverified. |
| draft_model_inference | Draft model (inference) | any | not-verified | not-verified | not-verified | not searched | no | Detected | schema/CLI inspection | Unverified. |
| llama.threads | llama threads | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Detected | schema/CLI inspection | Not exposed; never injected. |
| llama.threads_batch | llama batch threads | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.ctx_size | llama ctx size | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.batch_size | llama batch size | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.ubatch_size | llama micro batch | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.flash_attn | llama flash attention | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.rope_scaling | llama rope scaling | llama.cpp | unsupported | not-verified | not-verified | not searched | yes | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.rope_scale | llama rope scale | llama.cpp | unsupported | not-verified | not-verified | not searched | yes | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.rope_freq_base | llama rope freq base | llama.cpp | unsupported | not-verified | not-verified | not searched | yes | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.rope_freq_scale | llama rope freq scale | llama.cpp | unsupported | not-verified | not-verified | not searched | yes | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.yarn_orig_ctx | YaRN original context | llama.cpp | unsupported | not-verified | not-verified | not searched | yes | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.yarn_ext_factor | YaRN extension factor | llama.cpp | unsupported | not-verified | not-verified | not searched | yes | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.yarn_attn_factor | YaRN attention factor | llama.cpp | unsupported | not-verified | not-verified | not searched | yes | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.yarn_beta_slow | YaRN beta slow | llama.cpp | unsupported | not-verified | not-verified | not searched | yes | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.yarn_beta_fast | YaRN beta fast | llama.cpp | unsupported | not-verified | not-verified | not searched | yes | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.kv_offload | llama KV offload | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.cache_type_k | llama K cache type | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.cache_type_v | llama V cache type | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.parallel | llama parallel sequences | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.mlock | llama mlock | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.mmap | llama mmap | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.numa | llama NUMA | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.device | llama device | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.cpu_moe | llama CPU MoE | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.n_cpu_moe | llama N CPU MoE | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.swa_full | llama SWA full | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.kv_unified | llama unified KV | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| llama.defrag_thold | llama defrag threshold | llama.cpp | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not exposed; never injected. |
| mlx.auto_fit | MLX auto fit | mlx | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not applicable to GGUF/llama.cpp. |
| mlx.disk_cache | MLX disk cache | mlx | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not applicable to GGUF/llama.cpp. |
| mlx.kv_cache_quantization | MLX KV cache quantization | mlx | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not applicable to GGUF/llama.cpp. |
| vllm.gpu_memory_utilization | vLLM GPU memory utilization | vllm | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not applicable to GGUF/llama.cpp. |
| vllm.reasoning_parser | vLLM reasoning parser | vllm | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not applicable to GGUF/llama.cpp. |
| vllm.tool_call_parser | vLLM tool call parser | vllm | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not applicable to GGUF/llama.cpp. |
| yuzu.auto_fit | Yuzu auto fit | yuzu | unsupported | not-verified | not-verified | not searched | no | Unsupported (backend-gated) | schema/CLI inspection | Not applicable to GGUF/llama.cpp. |
