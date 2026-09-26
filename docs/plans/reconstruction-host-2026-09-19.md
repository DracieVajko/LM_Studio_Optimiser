# Rekonstrukcia LM Studio Optimizera na hostovi (2026-09-19)

Zdroj: handover z tabletu (praca na tablete, overene voci LM Studio na tomto hostovi).
Ciel: uviest hostovu kopiю (baseline 2026-08-29) do stavu popisaneho v handovri,
overeneho: 61 testov, live REST sondy (ocakavane 200/400), benchmark cisla (~67/21/79 tok/s).

## 1. Overeny kontrakt voci LM Studio (live, bude preprobnuty v Task 1)

### Load `POST /api/v1/models/load` — parametre naplocho (bez wrappera)
Podporovane (200): `context_length`, `flash_attention`, `offload_kv_cache_to_gpu`,
`eval_batch_size`, `physical_batch_size`, `parallel`, `context_checkpoints`,
`reasoning_budget_message`, `num_experts` (len MoE), `speculative_draft_max_tokens` (+ rodina).
Zamietnute (400 `unrecognized_keys`): `gpu_ratio`, `rope_freq_base`, `rope_freq_scale`,
`keep_model_in_memory`, `try_mmap`, `seed`, `use_fp16_for_kv_cache`,
`llama_k_cache_quantization_type`, `llama_v_cache_quantization_type`, `n_threads`,
`unified_kv_cache`, `ttl`, `num_layers`, `gpu`.

### Unload `POST /api/v1/models/unload`
Chce `{"instance_id": ...}` (nie `identifier`).

### Chat — nativny `POST /api/v1/chat`
Vstup: `{"model": ..., "input": "<string>", ...}` (nie `messages`).
Generacne klucе: `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
`reasoning`, `max_output_tokens`.
Zamietnute: `repetition_penalty`, `frequency_penalty`, `typical_p`, `mirostat_*`, `stop`, `seed`.
Odpoved nativna (`output[]` + `stats`) → konvertovat do OpenAI formatu pre benchmark.

### Stav servera
`GET /api/v1/models` → `{"models": [...], "loaded_instances": [...]}`.
Pred aj po kazdom teste: `loaded_instances` musi byt prazdne.

## 2. Ulohy a akceptacne kriteria

- T2 `domain/models.py`: `LoadConfiguration` + `physical_batch_size`, `parallel`,
  `context_checkpoints`, `reasoning_budget_message`, `speculative_draft_max_tokens`;
  `to_api_params()` vracia LEN podporovane klucе; `ConfigurationResult.generation: dict|None`.
- T3 `services/lm_studio.py`: `list_models` parsuje `models[]`; sonda capabilities =
  najmensi model + unload vo `finally` (resp. bezpecne 400-sondovanie bez loadu);
  `load_model` posiela parametre naplocho; `unload_model` cez `instance_id`;
  `chat_completion` nativne (`input` + `max_output_tokens`, bez `seed`/`stop`).
- T4 `api/client.py` + `api/schemas.py`: rovnaky kontrakt (flat `LoadConfig`,
  `instance_id`, nativny chat, `LoadConfigSchema` + nove polia, bez `gpu_ratio`/`rope_*` v send-path).
- T5 `services/search_space.py`: stage-4 dimenzie `physical_batch`/`parallel`/`checkpoints`
  (cez `advanced_settings`), `estimate_size` ich zapocita; 61 testov bez zmeny.
- T6 `services/benchmark.py`: volanie nativneho chatu; `--style` teploty
  (precise: coding/reasoning 0.1; creative: context/sumarizacia 0.9; balanced: suite default);
  `max_tokens = min(test-max, kontext/4)`; `result.generation` zaznamenane; median agregacia.
- T7 `cli/main.py` + `services/model_recommendations.py` (novy): `--style` pre
  benchmark/optimize/recommend; benchmark `--physical-batch/--parallel/--checkpoints`;
  `recommend --vram 6` = manualny checklist pre GUI/hosta; ASCII vystup (ziadne ✓/✗);
  `_display_benchmark_result` nepadá na `model_id`.
- T8 `hw_monitor.py` + `vram_matrix.py` (nove, root): monitor VRAM/RAM/swap + tok/s
  (bez argumentov nic nenacitava); matica kontext×flash×KV, 12 konfiguracii,
  "Say hi in 5 words.", max 30 tokenov; vzdy unload; kontrola `loaded_instances`.
- T9 `api/routes.py` + `README.md`: `_convert_config` + nove polia; README = overene
  parametre, CLI flagy, 6GB checklist.
- T10 Verifikacia: `pytest` 61 passed; live smoke (connect/list/load/unload/chat +
  prazdne `loaded_instances`); `vram_matrix` reprodukuje ~67/21 tok/s;
  kratky benchmark qwen3.5-4b (ctx 4096, precise, reps 1) passed.

## 3. Live verifikacia kontraktu 2026-09-19 (probe_api.py, 36/36 MATCH)
- `GET /api/v1/models` → `{"models": [...]}`; kazdy model ma vlastne `loaded_instances: [{id, config}]`
  (ziadne top-level `loaded_instances`). Model: `key`, `display_name`, `architecture`,
  `quantization{name,bits_per_weight}`, `size_bytes`, `max_context_length`, `capabilities`.
- Key-validacia predchadza model-resolution (fake model + bogus key → 400
  `unrecognized_keys`, nic sa nenacita) → bezpecne sondovanie bez loadu.
- Load 200: `{"type","instance_id","load_time_seconds","status"}` (ziadne `success`, ziadne echo).
  Echo plnej konfiguracie je v `loaded_instances[].config` (server defaulty:
  eval_batch_size=2048, physical_batch=512, parallel=4, checkpoints=32,
  reasoning_budget_message="", speculative max_tokens=3...).
- Unload: `{"instance_id"}` → 200 `{"instance_id"}`.
- Chat 200: `{"model_instance_id","output","response_id","stats"}`;
  `output[]` = `{type: reasoning|message, content}`; `stats` = `input_tokens`,
  `total_output_tokens`, `reasoning_output_tokens`, `tokens_per_second`,
  `time_to_first_token_seconds` (REALNE!), `model_load_time_seconds`.
- `/api/v1/chat/completions` NEEXISTUJE (404 Unexpected endpoint); `/api/version` tiez nie.
- `reasoning_budget_message` je string (cislo → 400 invalid_type); ""/low/medium/high/off akceptovane.

## 4. Rulings k implementacii
- R9: connect-sonda RED/GREEN cez httpx MockTransport (stary kod: load models[0], ziaden unload).
- R10: `reasoning_budget_message: str|None`, pass-through; hodnoty model-dependent ("" = default).
- R11: `estimated_ttft_ms` = realne `time_to_first_token_seconds*1000`; fallback 10% heuristika.
- R12: `prompt_tok_s = input_tokens/ttft`, `generation_tok_s = stats.tokens_per_second`,
  `prompt_tokens = input_tokens`, `completion_tokens = total_output_tokens`.
- R13: `load_time_ms` = `stats.model_load_time_seconds*1000` ak dostupne, inak client-merane.
- R14: `gpu_ratio`/`rope_*` ostavaju v dataclassach (kompat presetov/testov), ale send-path ich filtruje.
- R15: `capabilities.version = "v1"` po uspesnom GET /api/v1/models (ziadny version endpoint).
- R16: style teploty: precise → coding+reasoning 0.1; creative → context(sumarizacia) 0.9;
  balanced → suite defaulty; ulozene v `ConfigurationResult.generation`.

## 5. Faza 2 (2026-09-19/20, poziadavka usera: plna automatizacia + nocne behy)
- R28: pri "requires flash attention" chybe optimizer pre daný beh oreze flash=False
  (KV Q4 vyzaduje flash; REST ho nevie zmenit).
- R30: failure-guidance patri do model_recommendations; ziadna AI-decider netreba
  (deterministicky pipeline stacil: 2/3 modelov naladenych, 3. blokovany limitom modelu).
- R31: smoke KV default = GPU-ak-je, inak CPU (matrix porovnava obe zamerne).
- R32: benchmark generuje s reasoning="off" (odpoved, nie thinking trace).
- R33: quality cita najdlhsi uspesny output (best-of-N proti single-sample flake).
- R34: suite max 512->768 pre medium_reasoning + coding_task (derivacie sa zmestia;
  ctx/4 cap stale viaze maly kontext).
- R35: domain get_avg_* = median (robustne voci outlierom).
- R36: pri prazdnom vystupe prave jeden retry s reasoning="low" (iba ak model
  reasoning neodmietol).
- Checklist: KV Q4 starter + threads maximum (prisposobene userovmu presetu Q4/8 threads).
- Nocne vysledky: qwen3.5-4b 66.8 tok/s (ctx 2048, KV-GPU, q 0.983);
  ministral-3-3b 85.6 tok/s (ctx 8192, KV-GPU, q 1.000);
  glint blokovany (strict-JSON model limitation, dokazane 6 sondami).
