# Deep Research Prompt: LM Studio + Ollama + llama.cpp Tuning & Configuration

> **Purpose:** Systematic research prompt for finding authoritative docs, forums (GitHub Issues, HuggingFace Discussions, Reddit r/LocalLLaMA, LM Studio Discord/Community), benchmarks, and configuration guides covering LM Studio, Ollama, and llama.cpp settings. Target: improve speed, accuracy, and stability for local MoE/dense LLM inference.

**Isolation note:** Research results must reference sources only (no downloads of unverified binaries); any configuration recommendations must stay within this repo's folder (`LM_STUDIO_DEFAULTS.md` + persistence loop rules) and never require external leaks.

---

## Section 1: LM Studio — What to Tune and Document

### 1.1 Default file structure (`~/.lmstudio/.internal/user-concrete-model-default-config/`)
- Confirm key names and legacy mappings: `llm.load.contextLength`, `llm.load.llama.flashAttention`, `llm.load.offloadKVCacheToGpu`, `llm.load.llama.acceleration.offloadRatio`, `llm.load.llama.evalBatchSize`, `llm.load.llama.physicalBatchSize`, `llm.load.numParallelSessions`, `llm.load.llama.ropeFrequencyBase` / `ropeFrequencyScale` (stored as `{"checked": true, "value": N}`), `llm.load.llama.speculativeDecoding.*` (current vs legacy `llm.load.llama.speculativeDecoding.*`), `llm.load.llama.cpuThreadPoolSize`, `llm.load.llama.vCacheQuantizationType`, `llm.load.llama.kCacheQuantizationType`, `llm.load.llama.tryMmap`.
- What does LM Studio's renderer bundle say about new keys? Any new fields in recent releases that this optimizer should map?

### 1.2 Load options: performance and accuracy tradeoffs
- Impact of `flashAttention` (verified true/false) on speed and VRAM usage for 8–32GB GPUs.
- Impact of `offloadRatio` (0.0 to 1.0): how does partial GPU offload affect token/s for dense (27B, 35B) vs MoE (8x7B, 3.5B active) models?
- `evalBatchSize` / `physicalBatchSize`: relationship to throughput and latency; best values for mobile (8GB) vs desktop (16–24GB) GPUs.
- `numParallelSessions`: does increasing this improve throughput for single-user interactive use, or only for multi-user server mode?
- KV quantization (`vCacheQuantizationType`, `kCacheQuantizationType`): tradeoffs between accuracy and VRAM savings; recommended quant levels (`q4_0`, `q8_0`, etc.) per GPU tier.
- Context length (`contextLength`): maximum practical context lengths per GPU tier (8GB, 16GB, 24GB) for dense vs MoE, considering KV cache overhead.
- Rope settings (`ropeFrequencyBase`, `ropeFrequencyScale`): when should `checked` be `false` vs `true`; impact on long-context accuracy.
- Speculative decoding flags (`draftSimple`, `draftMtp`, `draftDflashSidecar`, etc.): which combinations actually improve decode speed in LM Studio's current engine version?

### 1.3 Persistence loop behavior
- Confirm: file writes apply on next LM Studio load without restart; verify with live echo (`echo_load_config=true`).
- Confirm: legacy `.optimizer-backup` recognized by restore; timestamped backups (`optimizer-backup-<utc>`) keep newest 10.
- Confirm: restore validates JSON before writing; invalid/unreadable files never overwritten (HTTP 409 / CLI exit 1).
- Confirm: `verify-default` loads empty config, compares only `API_KEYS` fields (`context_length`, `flash_attention`, `gpu_ratio` excluded from verification), reports `gpu_ratio` and rope as `unverified`, always unloads in `finally`.
- Confirm: `diff-default` aligns legacy speculative keys correctly; reports added/changed/unchanged/preserved; identical restore is a no-op.

---

## Section 2: Ollama — Configuration and Integration

### 2.1 Endpoint and model loading patterns
- Default URL (`http://localhost:11434`); how does this optimizer connect (`lm_optimizer/cli/main.py`: `get_client()`, `OllamaClient`).
- `POST /api/generate` vs `POST /v1/chat/completions`: which endpoint is used by the benchmark service? What are the request/response schemas?
- How to handle `load_model` and `unload_model` through Ollama's API? Does Ollama expose per-model instance IDs like LM Studio (`identifier`)?

### 2.2 Performance settings
- `num_gpu_layers`: relationship to `llama.cpp` offload behavior; best practices for 8GB, 16GB, 24GB GPUs.
- `num_ctx`: context window tradeoffs.
- Temperature, sampling, and batch settings that affect benchmark reproducibility vs interactive use.
- Any Ollama-specific settings (e.g., `keep_alive`, `num_predict`) that affect speed/accuracy measurements.

### 2.3 Integration with optimizer's persistence loop
- Can Ollama defaults/config files be mapped to LM Studio's `user-concrete-model-default-config` format, or does the optimizer treat them as separate backends?
- If running both backends (LM Studio + Ollama), how does the optimizer decide which backend's results to use for recommendations?

---

## Section 3: llama.cpp — Build, Flags, and Performance

### 3.1 Build-time and runtime flags
- `-ngl 99`: full GPU layer offload; interaction with MoE (`--n-cpu-moe N`).
- `--n-cpu-moe`: how many expert weights to process on CPU vs offload to GPU; impact on 8GB GPU performance for 35B MoE.
- Context size (`-c`): memory overhead per layer; how to calculate safe max context for a given GPU VRAM.
- Batch settings: `--batch-size`, `--ubatch-size`, `--mlock` (lock model in RAM), `--no-mmap` (disable memory mapping).
- Thread settings: `--threads` relationship to CPU cores; impact when using `--n-cpu-moe` (CPU already busy with experts).
- Flash Attention (`--flash-attn`): performance gain; any accuracy impact?
- KV cache quantization (`--v-cache-quantization`, `--k-cache-quantization`): VRAM savings vs speed/accuracy tradeoff; recommended levels per GPU tier.

### 3.2 Benchmark/reproducibility
- How to run deterministic benchmarks (`seed`, fixed temperature/sample settings) across `llama.cpp` and LM Studio so that `benchmark_service` measurements are comparable?
- What metrics to collect: `generation_tok_s`, `estimated_ttft_ms`, `context_length`, `score`, `success` rate.
- How to handle `NO_CONFIG_MEETS_SPEED_FLOOR`: what diagnostics should the benchmark service report for the fastest candidate?

---

## Section 4: Benchmark & Sandbox — Automation, Cleanup, and Recommendations

### 4.1 Intelligent scheduling
- Ordering rules (ascending size → MoE before dense → dense last); VRAM floor (250–500 MB free); speed floor (5 tok/s default, configurable via `--speed-floor`).
- Partial/intermediate results: `status='partial'` saved before pause; resume from partial state without corrupting `best_config_id`.
- Pause behavior: when `speed_floor` is not met, unload model (free VRAM), save partial, wait for user input (`--interactive` CLI / web prompt), then resume with shorter tests first, full tests after confirmation.

### 4.2 Cleanup automation
- `clean_tests` command (`--dry-run`, `--failed-count` default 2): remove artifacts and DB entries with `status='failed'` after consecutive failures; only within project folder; never delete outside directory.
- What artifacts to clean: temporary benchmark output files, failed run DB entries, backup files older than retention limit (`BACKUP_KEEP` = 10)? Should cleanup respect timestamped backups or delete them too?

### 4.3 Recommendation logic (`recommend_model`)
- Input: `model_id` + DB history (`run_repo.get`, `ConfigurationResult` list, `metrics_json`).
- Output: best `LoadConfiguration` (context_length, gpu_ratio, flash_attention, batch settings) with justification based on measured `generation_tok_s`, `score`, `success` rate.
- How to handle missing measurements for a model: fall back to catalog defaults or hardware fingerprint estimates (`vram_matrix.py`, `recommender.py`)?

---

## Section 5: Security, Isolation, and Web Access

### 5.1 API key isolation
- `LM_OPTIMIZER_ASSISTANT_URL` / `LM_OPTIMIZER_ASSISTANT_KEY`: read from `.env` (prefix `LM_OPTIMIZER_`); never written to repo/code; never exposed in logs or test output; never used unless user explicitly sets `.env` and passes `--use-assistant`.

### 5.2 Folder isolation
- All file operations (`defaults` write, backup creation, cleanup) restricted to `LM_OPTIMIZER_LMSTUDIO_HOME` (default: `~/.lmstudio`) and project directory (`tests/test_persistence_loop_cli.py` fixtures, `docs/`, `lm_optimizer/`).
- No writes to system directories; no `os.system` or subprocess calls to external commands (except `nvidia-smi` for VRAM detection, which is read-only and safe).

### 5.3 Web access
- By default: no web search or external HTTP requests (only LM Studio `127.0.0.1:1234` and optional `.env` endpoint).
- If user configures `LM_OPTIMIZER_ASSISTANT_URL` (e.g., external Mistral endpoint), requests should use `urllib.request` or `httpx` with timeouts, retry limits (`max_retries` = 3 from config), and no cookie/token leakage.

### 5.4 No live LLM load requirement
- Design specifies no live `load_model` needed for scheduling/recommendation; `verify-default` uses mock client (`FakeClient`) unless user explicitly confirms live mode.
- Any future live mode must be gated behind explicit user confirmation (`typer.confirm` or UI checkbox) and must unload model in `finally`.

---

## Section 6: Output Format Requirements

- Design document saved to `docs/specs/YYYY-MM-DD-<topic>-design.md`.
- Implementation plan saved to `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`.
- All CLI/API interfaces documented with exact signatures (parameters, return types).
- All tests include failing test → verification → passing test cycle; commit after each task.
- Self-review checklist: no placeholders (`TBD` removed), no contradictions (ordering rules match VRAM floor, isolation rules match cleanup scope), scope focused (single repo integration), types consistent (`TestManager` signature used consistently across CLI/service/test).
