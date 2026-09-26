# Changelog

All notable changes to this project will be documented in this file.

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