# Reconstruction of LM Studio Optimizer on the host (2026-09-19)

Source: handover from the tablet (work done on the tablet, verified against LM Studio on this host).
Goal: bring the host copy (baseline 2026-08-29) to the state described in the handover,
verified: 61 tests, live REST probes (expected 200/400), benchmark numbers (~67/21/79 tok/s).

## 1. Verified contract vs LM Studio (live, to be re-probed in Task 1)

### Load `POST /api/v1/models/load` — parameters flat (no wrapper)
Supported (200): `context_length`, `flash_attention`, `offload_kv_cache_to_gpu`,
`eval_batch_size`, `physical_batch_size`, `parallel`, `context_checkpoints`,
`reasoning_budget_message`, `num_experts` (MoE only), `speculative_draft_max_tokens` (+ family).
Rejected (400 `unrecognized_keys`): `gpu_ratio`, `rope_freq_base`, `rope_freq_scale`,
`keep_model_in_memory`, `try_mmap`, `seed`, `use_fp16_for_kv_cache`,
`llama_k_cache_quantization_type`, `llama_v_cache_quantization_type`, `n_threads`,
`unified_kv_cache`, `ttl`, `num_layers`, `gpu`.

### Unload `POST /api/v1/models/unload`
Wants `{"instance_id": ...}` (not `identifier`).

### Chat — native `POST /api/v1/chat`
Input: `{"model": ..., "input": "<string>", ...}` (not `messages`).
Generation keys: `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
`reasoning`, `max_output_tokens`.
Rejected: `repetition_penalty`, `frequency_penalty`, `typical_p`, `mirostat_*`, `stop`, `seed`.
Native answer (`output[]` + `stats`) → convert to OpenAI format for the benchmark.

### Server state
`GET /api/v1/models` → `{"models": [...], "loaded_instances": [...]}`.
Before and after every test: `loaded_instances` must be empty.

## 2. Tasks and acceptance criteria

- T2 `domain/models.py`: `LoadConfiguration` + `physical_batch_size`, `parallel`,
  `context_checkpoints`, `reasoning_budget_message`, `speculative_draft_max_tokens`;
  `to_api_params()` returns ONLY supported keys; `ConfigurationResult.generation: dict|None`.
- T3 `services/lm_studio.py`: `list_models` parses `models[]`; capability probe =
  smallest model + unload in `finally` (or safe 400-probing without load);
  `load_model` sends parameters flat; `unload_model` via `instance_id`;
  `chat_completion` native (`input` + `max_output_tokens`, no `seed`/`stop`).
- T4 `api/client.py` + `api/schemas.py`: same contract (flat `LoadConfig`,
  `instance_id`, native chat, `LoadConfigSchema` + new fields, no `gpu_ratio`/`rope_*` in send-path).
- T5 `services/search_space.py`: stage-4 dimensions `physical_batch`/`parallel`/`checkpoints`
  (via `advanced_settings`), `estimate_size` counts them; 61 tests unchanged.
- T6 `services/benchmark.py`: native chat calls; `--style` temperatures
  (precise: coding/reasoning 0.1; creative: context/summary 0.9; balanced: suite default);
  `max_tokens = min(test-max, context/4)`; `result.generation` recorded; median aggregation.
- T7 `cli/main.py` + `services/model_recommendations.py` (new): `--style` for
  benchmark/optimize/recommend; benchmark `--physical-batch/--parallel/--checkpoints`;
  `recommend --vram 6` = manual checklist for GUI/host; ASCII output (no ✓/✗);
  `_display_benchmark_result` does not crash on `model_id`.
- T8 `hw_monitor.py` + `vram_matrix.py` (new, root): VRAM/RAM/swap + tok/s monitor
  (no arguments loads nothing); context×flash×KV matrix, 12 configurations,
  "Say hi in 5 words.", max 30 tokens; always unload; `loaded_instances` check.
- T9 `api/routes.py` + `README.md`: `_convert_config` + new fields; README = verified
  parameters, CLI flags, 6GB checklist.
- T10 Verification: `pytest` 61 passed; live smoke (connect/list/load/unload/chat +
  empty `loaded_instances`); `vram_matrix` reproduces ~67/21 tok/s;
  short benchmark qwen3.5-4b (ctx 4096, precise, reps 1) passed.

## 3. Live contract verification 2026-09-19 (probe_api.py, 36/36 MATCH)
- `GET /api/v1/models` → `{"models": [...]}`; every model has its own `loaded_instances: [{id, config}]`
  (no top-level `loaded_instances`). Model: `key`, `display_name`, `architecture`,
  `quantization{name,bits_per_weight}`, `size_bytes`, `max_context_length`, `capabilities`.
- Key validation precedes model resolution (fake model + bogus key → 400
  `unrecognized_keys`, nothing loads) → safe probing without load.
- Load 200: `{"type","instance_id","load_time_seconds","status"}` (no `success`, no echo).
  Full configuration echo is in `loaded_instances[].config` (server defaults:
  eval_batch_size=2048, physical_batch=512, parallel=4, checkpoints=32,
  reasoning_budget_message="", speculative max_tokens=3...).
- Unload: `{"instance_id"}` → 200 `{"instance_id"}`.
- Chat 200: `{"model_instance_id","output","response_id","stats"}`;
  `output[]` = `{type: reasoning|message, content}`; `stats` = `input_tokens`,
  `total_output_tokens`, `reasoning_output_tokens`, `tokens_per_second`,
  `time_to_first_token_seconds` (REAL!), `model_load_time_seconds`.
- `/api/v1/chat/completions` DOES NOT EXIST (404 Unexpected endpoint); neither does `/api/version`.
- `reasoning_budget_message` is a string (number → 400 invalid_type); ""/low/medium/high/off accepted.

## 4. Rulings on implementation
- R9: connect probe RED/GREEN via httpx MockTransport (old code: load models[0], no unload).
- R10: `reasoning_budget_message: str|None`, pass-through; values model-dependent ("" = default).
- R11: `estimated_ttft_ms` = real `time_to_first_token_seconds*1000`; fallback 10% heuristic.
- R12: `prompt_tok_s = input_tokens/ttft`, `generation_tok_s = stats.tokens_per_second`,
  `prompt_tokens = input_tokens`, `completion_tokens = total_output_tokens`.
- R13: `load_time_ms` = `stats.model_load_time_seconds*1000` if available, else client-measured.
- R14: `gpu_ratio`/`rope_*` stay in dataclasses (preset/test compat), but send-path filters them.
- R15: `capabilities.version = "v1"` after successful GET /api/v1/models (no version endpoint).
- R16: style temperatures: precise → coding+reasoning 0.1; creative → context(summary) 0.9;
  balanced → suite defaults; stored in `ConfigurationResult.generation`.

## 5. Phase 2 (2026-09-19/20, user request: full automation + night runs)
- R28: on "requires flash attention" error the optimizer prunes flash=False for that run
  (KV Q4 requires flash; REST cannot change it).
- R30: failure-guidance belongs in model_recommendations; no AI-decider needed
  (deterministic pipeline sufficed: 2/3 models tuned, 3rd blocked by model limits).
- R31: smoke KV default = GPU-if-available, else CPU (matrix compares both deliberately).
- R32: benchmark generates with reasoning="off" (answer, not thinking trace).
- R33: quality reads the longest successful output (best-of-N against single-sample flake).
- R34: suite max 512->768 for medium_reasoning + coding_task (derivations fit;
  ctx/4 cap still binds small contexts).
- R35: domain get_avg_* = median (outlier-robust).
- R36: on empty output exactly one retry with reasoning="low" (only if the model
  did not refuse reasoning).
- Checklist: KV Q4 starter + threads maximum (adapted to the user's Q4/8-thread preset).
- Night results: qwen3.5-4b 66.8 tok/s (ctx 2048, KV-GPU, q 0.983);
  ministral-3-3b 85.6 tok/s (ctx 8192, KV-GPU, q 1.000);
  glint blocked (strict-JSON model limitation, proven by 6 probes).
