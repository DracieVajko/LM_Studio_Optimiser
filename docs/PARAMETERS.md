# Parameter matrix: what the optimizer can and cannot set

Live-verified against the bundled LM Studio server (`lms` commit `69d945a`,
2026-09-19). Unknown keys return HTTP 400 `unrecognized_keys` **without loading
any model**, so capabilities are re-probed safely on every connect. A different
server version simply enables/disables rows below.

## Load parameters (`POST /api/v1/models/load`, flat JSON)

| Parameter | Set via REST | Meaning | Notes |
|---|---|---|---|
| `context_length` | yes | Max context window (tokens) | Main tuning dimension (2048-262144+). Larger ctx = more KV memory. |
| `flash_attention` | yes | Flash Attention kernels | **Required** with non-F16 KV cache quantization (server refuses load otherwise). Negligible speed delta on/off when allowed. |
| `offload_kv_cache_to_gpu` | yes | KV cache on GPU (`true`) or CPU (`false`) | ~3x tok/s difference (67 vs 22 on 4B/Q4). Biggest speed lever. |
| `eval_batch_size` | yes | Prompt-processing batch | Server default 2048. Tested 64-1024 in stage 4. |
| `physical_batch_size` | yes | Physical batch (memory path) | Server default 512. Stage-4 dimension, usually best at default. |
| `parallel` | yes | Max concurrency ("Max Concurrency" in GUI) | Server default 4. Stage-4 dimension. |
| `context_checkpoints` | yes | Context checkpointing granularity | Server default 32. Stage-4 dimension. |
| `reasoning_budget_message` | yes | Reasoning budget (string, e.g. low/medium/high) | Type is string; model-dependent values. |
| `num_experts` | yes | Active experts (MoE only) | Accepted generally; only meaningful for MoE architectures. |
| `speculative_draft_max_tokens` | yes | Speculative decoding draft length | Ineffective without a draft model configured (server default 3). Not benchmarked. |
| `gpu_ratio` | NO (CLI/GUI) | Fraction of model layers on GPU (0-1) | `lms load <model> --gpu 0.5\|max\|off`. REST knows only full on/off via KV placement. Stored in configs for documentation. |
| `rope_freq_base` / `rope_freq_scale` | NO | RoPE frequency overrides | Rejected via REST. Experimental everywhere; disabled in search. |
| `keep_model_in_memory` | NO | Keep weights resident after unload | Rejected via REST. |
| `try_mmap` | NO | Memory-map weight files | Rejected via REST. |
| `use_fp16_for_kv_cache`, `llama_k/v_cache_quantization_type` | NO (GUI) | KV cache precision | Set K/V cache to **Q4 starter** in LM Studio GUI (needs flash on). Biggest 6GB lever; REST cannot set it. |
| `n_threads` | NO (GUI) | CPU thread count | Set to maximum logical cores in LM Studio GUI. REST cannot set it. |
| `unified_kv_cache`, `ttl`, `num_layers`, `gpu`, `seed` | NO | Misc | Rejected via REST. |

Unload takes `{"instance_id": ...}` (returned by load as `instance_id`).

## Chat parameters (native `POST /api/v1/chat`, `{"model", "input": str}`)

| Parameter | Supported | Meaning |
|---|---|---|
| `temperature` | yes | Sampling temperature. Benchmarks: 0.3/0.3/0.3/0.1/0.0 per test; styles override coding/reasoning (precise 0.1) and summaries (creative 0.9). Precision probe tests 0.2. |
| `top_p` / `top_k` / `min_p` | yes | Nucleus/top-k/min-p sampling. Benchmarks leave server defaults; per-model recommendations in `.md` files. |
| `presence_penalty` | yes | Presence penalty. Server default in benchmarks. |
| `reasoning` | yes, values model-dependent | Thinking effort. Allowed set varies: `off`/`on` only on some models (verified live on qwen3.5-4b), full `off`/`low`/`medium`/`high`/`on` range on others; unsupported models reject the key (client retries without it). Benchmarks use `off` (score answers, not traces). |
| `max_output_tokens` | yes | Generation cap. Suite: 256/512-768/1024/512/256 per test, capped by `context/4`. |
| `repetition_penalty`, `frequency_penalty`, `typical_p`, `mirostat_*`, `stop`, `seed` | NO | Rejected via REST (400). Note: `stop` sequences cannot be enforced server-side. |

`/api/v1/chat/completions` (OpenAI-compatible) does **not** exist in this server version.

## Sourced generation defaults per lineage

See `lm_optimizer/services/generation_defaults.py` (every entry cites its source;
community fine-tunes inherit the closest official base):

- Qwen3.5 instruct: 0.7/0.8/20/0/pp1.5 (Qwen3.5 official)
- Qwen3 (incl. community qwen3.8/tongyi/marco): 0.7/0.8/20/0 non-thinking (Qwen3 official)
- DeepSeek-R1-Distill (fin-r1, deepseek-14b): 0.6/0.95, no system prompt (DeepSeek official)
- Ministral/Mistral: 0.0-0.7 range, 0.2 focused (Mistral official docs)
- Gemma (GGUF): 1.0/0.95/64 (Gemma team + Unsloth GGUF defaults)
- gpt-oss: 1.0/1.0, harmony format required (OpenAI official)
