# Changelog

All notable changes to this project will be documented in this file.

## [1.5.0b1] - 2026-10-03 - BETA

Ollama backend is beta (implemented, mocked-test evidence only, no live
server verification yet). llama.cpp backend is not implemented (explicit
phase-2 stub). LM Studio path is stable and unchanged by default.

### Added
- **Backend switch**: `--backend lm-studio|ollama|llama-cpp` (default
  `lm-studio`, byte-identical behavior), `BackendClient` seam.
- **Ollama (beta)**: options sweep (`num_ctx`, `num_batch`, `num_thread`,
  `num_gpu`, sampling), `ollama-export` / `ollama-apply` with editable
  `<model>-best.modelfile.md`, new-tag-only confirmed `/api/create`.

## [1.4.0] - 2026-10-02

### Added
- **Stage-4 grid unification**: eval batch default `[64, 128, 256, 512, 1024, 2048]`,
  parallels `[1, 2, 4, 8]` throughput-gated (interactive collapses to `[1]`);
  physical/checkpoints extended values opt-in only via `advanced_settings`.
- **MoE minimal gating**: max 3 derived expert variations (`default, /2, /4`),
  only when `is_moe`; `n-cpu-moe` documented manual (no verified channel).
- **Experimental opt-ins** (default OFF): `--enable-rope` / `--enable-cpu-moe` /
  `--enable-speculative` on `optimize` + `auto`, tty-gated prompt, run marked
  `is_experimental`.
- **Draft discovery**: extended hints + `RECOMMENDED_DRAFTS` (names only, never
  downloaded), `find_local_draft()`, skip-with-reason when absent.
- **Manual memory duel**: new `manual-memory-duel --model --stage mmap|keep` CLI —
  guided re-measure of the auto best after a manual GUI toggle, keep/revert
  verdict at +5% gen tok/s (TTFT tiebreak), fail-closed unload before+after.

## [1.3.0] - 2026-10-02

### Fixed
- **Settings layout**: `space-y-*` removed from grid containers (it offset form
  rows); Host/Port and settings pairs align in clean 2-col grids.
- **README**: screenshot slots for Profiles + CMD menu, trimmed success-report
  example, `docs/screenshots/` dir.
- **macOS CI**: `get_cpu_info()` no longer crashes where `psutil.cpu_freq` is
  missing (AttributeError on some runners) — frequency degrades to None,
  detection never fails. Regression tests included.

### Added
- **Denser speed grid**: eval batch {64, 128, 256, 1024, 2048}, MoE experts {2, 4, 8},
  parallel {1, 2, 8} (~+6 probes per model, still under the 25-probe cap).
- **Speculative token sweep**: `speculative_draft_max_tokens` {3, 8} with MTP whenever
  a draft model is discovered.
- **Sampling sweep**: opt-in `sample-sweep` CLI command — top_p/top_k ×
  precision/chat/creative from source-tagged publisher values, full suite +
  quality re-validation per combo, best-profile verdict.
- **BenchmarkCase top_p/top_k**: threaded from case definition through all
  generation paths (incl. reasoning retry and JSON nudge); defaults stay None
  (server defaults, comparability preserved).

### Fixed
- **CI red**: `GET /api/models/{id}` returns JSON 503 when LM Studio is unreachable
  instead of raising; error-shape test no longer needs a live server.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-27

### Added
- **Core Architecture**
  - Modular domain-driven design with clean separation (domain, services, database, API, UI, CLI)
  - Pydantic-based configuration management with `.env` support
  - Structured JSON logging with structlog

- **LM Studio Integration**
  - Async HTTP client with retry logic and capability discovery
  - Automatic detection of supported load parameters
  - Model listing, loading, unloading, and chat completion

- **Hardware Detection**
  - Cross-platform GPU detection (NVIDIA via GPUtil/nvidia-smi, AMD via rocm-smi, Apple Metal)
  - CPU, RAM, VRAM detection
  - Configurable hardware overrides

- **Benchmark Suite**
  - 5 deterministic tests: Short Instruction, Medium Reasoning, Long Context, Coding, Structured Output
  - Configurable repetitions with warmup runs
  - Median aggregation for robustness
  - Streaming-aware TTFT estimation

- **Quality Evaluation**
  - Multi-dimensional scoring (task completion, factual consistency, format compliance, coding correctness, truncation detection, malformed detection)
  - Category-specific evaluators (JSON, code, general text)
  - Configurable quality threshold (default 0.97)
  - Repetition detection

- **Optimization Engine**
  - 5-stage adaptive search: Discovery → Coarse Search → Refinement → Batch Optimization → Validation
  - Dynamic search space generation based on hardware, model, and LM Studio capabilities
  - Profile-based scoring (Speed, Balanced, Context, Quality, Custom)
  - Pareto frontier calculation
  - Checkpointing and resume capability
  - Pause/cancel/resume control

- **Database**
  - SQLite with versioned migrations
  - Repositories for hardware, models, runs, configurations, benchmarks, quality, presets, settings
  - Full run history with model-specific views

- **Web UI**
  - FastAPI + Jinja2 templates
  - Tailwind CSS for styling
  - Chart.js for visualizations
  - WebSocket for real-time progress updates
  - Dashboard, History, Results, Settings pages
  - Interactive charts (generation vs context, VRAM vs context, speed vs GPU ratio, quality vs speed, Pareto frontier)

- **CLI**
  - Typer-based commands: status, models, inspect, benchmark, optimize, apply, runs, presets, restore
  - Rich terminal output with tables and progress bars
  - Dry-run support for all commands

- **Preset Management**
  - Save/load/apply/delete presets
  - Export/import JSON
  - Model-specific preset history

- **Safety Features**
  - Automatic model unload between tests
  - Checkpointing every 5 configurations
  - Graceful pause/cancel/resume
  - Restore previous configuration
  - Timeout handling

### Security
- Default binding to localhost (127.0.0.1:8080)
- CORS restricted to localhost
- No credentials stored
- Input validation on all API endpoints

### Documentation
- Comprehensive README with installation, usage, and architecture
- CONTRIBUTING.md with development guidelines
- SECURITY.md with threat model and best practices
- CHANGELOG.md

### Tests
- Unit tests for config, hardware, benchmark suite, profiles, model capabilities
- Pytest with asyncio support
- Type checking with mypy (strict)
- Linting with ruff

### Configuration
- `.env.example` with all settings
- Pydantic Settings with validation
- Environment variable overrides for all settings

## [1.0.0-beta] - 2026-08-29

First public beta release.

### Highlights

- Hardware-aware optimization
- Adaptive benchmark/search pipeline
- Speed/Balanced/Context/Quality/Custom profiles
- LM Studio capability discovery
- Baseline comparison
- Correctness heuristics
- Pareto analysis
- SQLite history
- Presets
- Web UI
- CLI
- Configurable local/remote LM Studio endpoint

### Beta Status

Automated tests and static validation pass.

Physical end-to-end LM Studio validation is still pending.

> ⚠️ This is an untested beta. The developer machine is unavailable for repair, so no complete real-world optimization run has been performed for this release. Use with caution and verify generated configurations.

### Changed - Optimization Correctness (v0.1.1)
- **Hardware-agnostic scoring**: Removed hardcoded `50 tok/s`, `1000 prompt tok/s`, `2000ms TTFT`, `32768 context`, `6GB VRAM` assumptions. Speed/TTFT now run-relative, context model-relative, memory hardware-relative (15% headroom).
- **Memory scoring**: No longer “less VRAM is always better”; 5GB/30tok and 8GB/50tok both score 1.0 on 24GB GPU, OOM on 6GB.
- **Quality terminology**: Renamed to “Correctness / Quality (heuristic checks)”, shows `29/30 checks passed`, not false precision.
- **Profile thresholds**: Documented intentional differences (Speed 0.95, Balanced/Context 0.97, Quality 0.99) and validation that quality weight drives selection.
- **TTFT**: Renamed to `estimated_ttft_ms`, labeled “Est. TTFT (estimated, no streaming)” everywhere.
- **Coarse search**: Deterministic intelligent sampling covering low/high context, GPU, KV, Flash, batch (no longer first N).
- **Refinement**: Adaptive around top-3 Pareto candidates, interaction tests, avoids failed regions.
- **RoPE**: Experimental, disabled by default, requires explicit enable and stricter quality.
- **Baseline**: Captures actual current config, benchmarks it, shows % changes from real measurements.
- **Explainability**: Every winning config exposes `score_breakdown` per component.
- **Determinism**: Fixed prompts/seed 42, deterministic candidate generation, `benchmark_params` stored in DB.
- **Tests**: 46 new tests for arbitrary VRAM (4/6/12/24/48+GB), CPU-only, zero-range, OOM, Pareto, determinism.

### Changed - v1.0.0-beta Release Readiness
- **LM Studio URL**: Default changed from developer-specific `http://100.101.20.64:1234` to universal `http://127.0.0.1:1234`. Fully configurable via `.env`, Web UI Settings, and `lm-optimizer status --url <url>` (CLI override not persisted). Validated with user-friendly errors.
- **Developer data removed**: Eliminated personal IP, RTX 3060, 6GB hardcodes from source/docs; replaced with neutral `127.0.0.1` / `192.168.1.100` examples.
- **Configuration**: Storage moved to `data/` (`data/optimizer.db`, `data/reports/`, `data/presets/`, `data/logs/`). Added `DATABASE_PATH` to `config.py`. Legacy `config/optimizer.db` auto-migrated.
- **Repository hygiene**: Added `.env.example` and comprehensive `.gitignore` (covers `data/`, `*.db`, `__pycache__/`, `*.log` etc.). Removed generated `optimizer.db`/logs from tracking.
- **First-run**: App no longer crashes if LM Studio offline; shows “⚠ LM Studio is not available — Open Settings to configure”.
- **Model selection**: UI now requires explicit single model selection, shows metadata (arch, params, quant, context, MoE, size), search filter, and disables Start until selected. Different quantizations treated as separate targets.
- **Profile selection**: Explicit cards with descriptions; Custom validates weights total 100%.
- **Advanced settings**: Collapsed by default with “Normal users do not need to change these”; only supported params shown.
- **Pre-run review**: Confirmation screen shows model, profile, search space, estimated configs; large runs (>50) warn “Estimated tests: 384 — Continue?”.
- **Results**: Distinguishes baseline, best speed/balanced/context/quality, Pareto frontier with actual measured values only.
- **Apply**: Shows diff `Current → New`, verifies actual LM Studio config after load, restores on failure.
- **Presets**: First-class with model identity, hardware snapshot, metrics, version, timestamp; save/rename/apply/delete/export/import with compatibility check.
- **History**: Persists across restarts, shows date/model/quant/profile/speed/context/score/duration, deletable without accidental data loss.
- **Cleanup**: Consolidated duplicate hardware/benchmark/quality implementations; deprecated legacy `optimizer/engine.py` and `ui/web.py` (kept as wrappers). Single authoritative `hardware/`, `benchmark/suite.py`, `scoring/evaluator.py`, `services/lm_studio.py`.
- **Security**: Default `127.0.0.1`, CORS restricted to localhost, URL validation prevents SSRF, no secrets in logs.
- **Dependencies**: Audited `pyproject.toml`, pinned minimum versions, separated runtime/dev.
- **Installation**: Fresh `pip install -e .` in clean venv verified.
- **Documentation**: Rewrote README for public GitHub with beta status, problem statement, architecture, features, hardware support, limitations, safety; fixed outdated claims (no “objectively best”).
- **Version**: Bumped `0.1.0 → 1.0.0-beta` (PEP 440 `1.0.0b0` in package) across `pyproject.toml`, `__init__.py`, `api/main.py`, `domain/models.py`.
- **Tests & CI**: Added smoke tests for URL, disconnected LM Studio, model/profile selection, preset compatibility, DB init; `pytest 61 passed`, `ruff check`, `mypy`, `compileall` clean.

## [1.2.0] - 2026-10-01

### Added
- **Unload guard (`services/unload_guard.py`)**: fail-closed `assert_unloaded()` on every
  model boundary (auto/fit/ctx/matrix loops, optimize CLI, sandbox duels, config compare,
  web optimize start). A stuck model stops the run with a Keep-OFF hint instead of
  silently contaminating the next model's speed measurements.
- **Structured-output nudge**: one follow-up call (`Reply with ONLY the JSON object`)
  when a reasoning model thinks aloud without emitting JSON; original text kept as thinking.
- **Failed-run reports**: `results/<model>-failed-<stamp>.md` with per-config table,
  per-test breakdown of the closest config, full failed-test outputs and guidance.
- **Generation profiles**: measured benchmark settings + publisher recommendations
  (Precision / Chat / Creative) per architecture family, every value source-tagged;
  `GET /api/generation-profiles` + Results section.
- **Optimize-all**: dashboard queue (multi-model sequential runs with skip terms and
  stop-after-current), CMD menu item `[3]`, `run-model.bat ALL`, Linux passthrough docs.
- **Installers**: `install.bat` / `install.sh` (venv + `pip install -e .[dev]` + `.env`
  from example) and `.env.example`.
- **CI matrix**: GitHub Actions on `ubuntu-latest` + `macos-latest` (pytest + JS syntax).
- **Apple Silicon honesty**: unified-memory entry instead of fake VRAM errors; hardened
  snapshot formatting.

### Changed
- **Speed context capped at 4096** (was: full user cap up to 32768): cheaper loads/KV,
  quality phase still runs at the user cap.
- **Progress bar**: planned probe units instead of the Cartesian estimate; recovery
  budget planned upfront so the bar never dips mid-run.
- **KV-quant flash lesson persists** across runs (preskips known-impossible flash-off
  probes; self-heals when a flash-off load serves).
- **JSON fallback**: think-aloud-before-JSON extracts with `format 0.5` +
  `thinking_outside_json` flag and UI warning instead of auto-0; stray `</think>`
  penalized via `no_malformed`.
- **README rewritten** for v1.1.0+ reality (sandbox, duels, dark mode, per-OS launchers,
  screenshot slots, platform matrix); Troubleshooting §11 (two-models-loaded); Linux
  `.sh` launchers; `TROUBLESHOOTING.md` Keep-OFF prerequisite.

### Fixed
- **HTTP 500 on Start Optimization**: run+state prepared synchronously (`prepare_run()`
  / `execute()` split); background task continues from the prepared run.
- Menu batch fall-through in `run-model.bat`; launcher test updated for the new item.
- Stale `running` rows in `cleanup_test_runs` are listed, never auto-deleted.

## [1.1.0] - 2026-09-28

### Added
- **Model sandbox (`/sandbox`)**: text / HTML / 3D-scene duels model-vs-model (sequential with unload, server defaults), isolated output folders, iframe preview + file links, preset prompts per kind, model dropdowns.
- **Duel history**: `duels` table (migration 10), `GET /api/duels`, per-duel view, `DELETE` incl. files; History page section with bulk cleanup.
- **Run lifecycle**: `DELETE /api/runs/{id}` (guarded), `POST /api/runs/{id}/abandon` (dead runs → INTERRUPTED), `config_count` + `stale` flags, per-row Abandon/Delete buttons.
- **Results transparency**: per-test Prompt/Thinking/Output blocks with token counts, `thinking_text` plumbing (client → benchmark → DB → API → UI), full config detail + standalone JSON page, truthful zero-pass reasons incl. quality_rejected.
- **Dark mode**: OS-following with toggle, chart theming, contrast-checked overrides.
- **Nav unification**: identical header (logo, links, theme, status badge) on all pages; shared status poller.

### Changed
- Read-only API endpoints are load-free (`echo_probe=False`): `/api/status` ~0.5 s instead of timeouts; capability probing cached per server + warmed at startup.
- `/api/runs` lists skip corrupt rows (counted); scoreless configs serialize (`score: null`) instead of killing the endpoint.
- Apply is validated (404 unknown model), load-free, with one-shot confirmation.

### Fixed
- `websocket.js` class/comma SyntaxError whitescreening dashboard+results; JS syntax regression test over all UI scripts.
- Recommended card and table reading config from wrong level (always OFF/CPU/Auto); comparison modal X button (module-scope handler).
- Double-prefix route `/api/api/runs/...`; SPA nav hijack swallowing new pages; asset cache-busting versions.

## [Unreleased]

### Planned
- Docker support with GPU access
- LLM-as-judge quality evaluation (optional)
- Multi-GPU optimization support
- Export to CSV/HTML reports
- Webhook notifications for long runs
- Comparison view between runs
- Scheduled optimization runs
- Model quantization recommendations