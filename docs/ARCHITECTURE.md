# LM Studio Auto Optimizer - Architecture Documentation

## Overview

This document provides a comprehensive technical overview of the LM Studio Auto Optimizer architecture for frontier AI systems to understand, extend, and maintain the codebase.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LM Studio Auto Optimizer                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   CLI        │  │   Web UI     │  │   API        │  │  Services  │ │
│  │  (Typer)     │  │  (FastAPI+   │  │  (FastAPI)   │  │  (Core)    │ │
│  │              │  │   Tailwind)  │  │              │  │            │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬─────┘ │
│         │                 │                 │                 │       │
│         └─────────────────┼─────────────────┼─────────────────┘       │
│                           ▼                 ▼                         │
│              ┌─────────────────────────────────────────────┐         │
│              │           Domain Layer (Pydantic)           │         │
│              │  Models, Profiles, Configs, Results, Runs   │         │
│              └─────────────────────────────────────────────┘         │
│                           │                                           │
│         ┌─────────────────┼─────────────────┐                        │
│         ▼                 ▼                 ▼                        │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐                   │
│  │ Database   │   │  Storage   │   │  Backends  │                   │
│  │ (SQLite)   │   │ Checkpoints│   │ LM Studio  │                   │
│  │            │   │ (JSON)     │   │  Ollama    │                   │
│  └────────────┘   └────────────┘   └────────────┘                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Domain Models (`lm_optimizer/domain/models.py`)

All data structures use Pydantic v2 for validation and serialization.

Key models:
- `OptimizationRun` - Complete optimization run with metadata
- `ConfigurationResult` - Single configuration benchmark result
- `ModelIdentity` - Model metadata (id, name, architecture, etc.)
- `HardwareInfo` - Detected hardware (CPU, GPU, RAM, VRAM)
- `LoadConfiguration` - LM Studio load parameters
- `RunStatus`, `ConfigurationStatus`, `OptimizationStage` - State enums
- `RunStatus`: `pending`, `running`, `resumed`, `paused`, `partial_success`, `success`, `failed`, `cancelled`, `interrupted`

### 2. Services Layer

#### Benchmark Service (`lm_optimizer/services/benchmark.py`)
- Executes benchmark cases against loaded models
- Handles preheat/warmup, streaming, metrics collection
- `BenchmarkService` - main class
- `BenchmarkCase` - single test definition

#### Quality Evaluator (`lm_optimizer/services/quality.py`)
- Multi-dimensional quality scoring
- Checks: task completion, factual consistency, format compliance, coding correctness, truncation, malformed
- `QualityEvaluator` with `QualityConfig`

#### Optimizer (`lm_optimizer/services/optimizer.py`)
- `AdaptiveOptimizer` - main orchestration class
- Phase-A pipeline: Prepare → Speed → Frontier → Quality → Recovery → Final Validation
- Checkpointing at every phase boundary
- Resume from checkpoint with `resume_from_checkpoint()`

#### Search Space Generator (`lm_optimizer/services/search_space.py`)
- Hardware-aware configuration space generation
- GPU-specific candidates (flash, KV, batch, GPU ratio)
- MoE expert gating (max 3 variations)
- Speculative draft discovery

#### Host Guard (`lm_optimizer/services/hostguard.py`)
- Exclusive model access enforcement
- Waits for loaded models to unload
- Cross-process coordination

#### Job Lock (`lm_optimizer/services/joblock.py`)
- Single optimizer job at a time
- PID-based lock with stale detection
- Automatic recovery from crashed processes

### 3. Database Layer (`lm_optimizer/database/`)

SQLite with repositories pattern:
- `run_repo` - OptimizationRun CRUD
- `config_repo` - ConfigurationResult CRUD
- `model_repo` - ModelIdentity CRUD
- `preset_repo` - User presets
- `settings_repo` - User settings
- `duel_repo` - Sandbox duel history

### 4. API Layer (`lm_optimizer/api/`)

FastAPI with:
- `/api/runs` - Run management (list, get, configs, pareto, export, abandon, delete)
- `/api/optimize` - Start/pause/resume/cancel optimization
- `/api/models` - Model listing and details
- `/api/duels` - Sandbox duel history
- `/api/settings` - User settings
- WebSocket `/api/ws/{run_id}` - Live progress updates

### 5. Web UI (`lm_optimizer/ui/`)

FastAPI + Tailwind CSS + vanilla JS modules:
- `templates/` - Jinja2 templates
- `static/js/` - ES6 modules (api.js, ui.js, results.js, history.js, charts.js, websocket.js)
- Dark mode via CSS variables + localStorage

### 6. CLI (`lm_optimizer/cli/main.py`)

Typer-based commands:
- `optimize` - Single model optimization
- `auto` - All models sequentially
- `resume` / `resume-menu` - Resume from checkpoint
- `pause` - Request graceful pause
- `checkpoints` - List checkpoints
- `judge` - AI-assisted model selection
- `context-sweep` - Max context sweep
- `sample-sweep` - Generation sampling sweep
- `manual-memory-duel` - Manual keep/revert test
- `ollama-export` / `ollama-apply` - Ollama Modelfile workflow

### 7. Storage (`lm_optimizer/storage/run_checkpoint.py`)

JSON-based checkpoints with atomic writes:
- Full optimizer state serialization
- Completed candidate tracking (no re-runs on resume)
- Phase state preservation
- Recovery log

## Key Flows

### Optimization Flow

```
1. User calls `optimize` or `auto` (CLI) or POST /api/optimize (Web)
2. Acquire job lock (fail if another job running)
3. Connect to LM Studio, verify empty state
4. Detect hardware
5. Generate search space
6. Create AdaptiveOptimizer
7. Run Phase-A pipeline:
   a. PREPARE: unload, snapshot free resources
   b. SPEED: cheap probes at frozen context
   c. FRONTIER: 5% band + contrarian
   d. QUALITY: full suite on finalists at frozen quality ctx
   e. RECOVERY: bounded rollback (max 3 probes)
   f. FINAL_VALIDATION: adaptive reps, median wins
8. Optional: CONTEXT_SWEEP (Phase B)
9. Save best report, update DB
10. Release lock
```

### Pause/Resume Flow

```
Pause (Web UI or CLI):
1. User clicks Pause or runs `lm-optimizer pause <run_id>`
2. Creates pause flag file at `checkpoint_dir/pause_{run_id}.flag`
3. Optimizer checks flag at safe boundaries (after each candidate/suite)
4. On flag: checkpoint_now(reason="user-pause"), update status to "paused"

Resume (Web UI or CLI):
1. User clicks Resume or runs `lm-optimizer resume <run_id>` or `resume-menu`
2. Load checkpoint from storage
3. Create new AdaptiveOptimizer
4. Call `resume_from_checkpoint(run_id, revalidate=False)`
5. Optimizer skips completed candidates, continues from next
6. Live progress via WebSocket (Web UI)
```

### Checkpoint Structure

```json
{
  "run_id": "uuid",
  "model_id": "model-name",
  "stage": "quality",
  "search_space": [...],
  "completed_candidate_ids": [1, 2, 5, 8],
  "phase_state": {
    "speed": {...},
    "frontier": {...},
    "quality": {...},
    "recovery": {...}
  },
  "optimizer_state": {...},
  "saved_at": "2026-10-07T12:00:00Z"
}
```

## Configuration

### Profiles (`lm_optimizer/domain/models.py` - `OptimizationProfile`)
- `SPEED` - maximize generation tok/s
- `BALANCED` - speed × quality balance
- `CONTEXT` - maximize context length
- `QUALITY` - maximize quality score
- `CUSTOM` - user-defined weights

### Advanced Settings (`lm_optimizer/api/schemas.py` - `AdvancedSettingsSchema`)
- `max_context` - cap context sweep
- `min_gpu_ratio` / `max_gpu_ratio`
- `eval_batch_sizes` - custom batch values
- `parallel_values` - parallel candidates
- `enable_phase_b` - context sweep opt-in
- `enable_rope` / `enable_cpu_moe` / `enable_speculative` - experimental flags

## Testing

543 tests in `lm_optimizer/tests/`:
- Unit tests for each service
- Integration tests for API endpoints
- CLI command tests
- Pause/resume durability tests
- Phase-A pipeline correctness
- Web UI rendering tests
- Database isolation tests

Run: `python -m pytest lm_optimizer/tests/ -v`

## Extending the System

### Adding a New Backend

1. Implement `BackendClient` protocol in `lm_optimizer/backends/`
2. Add to `_VALID_BACKENDS` in `cli/main.py`
3. Add client factory in `get_backend_client()`
4. Register capabilities in `parameter_registry.py`

### Adding a New Benchmark Test

1. Add case to `BenchmarkSuite` in `services/benchmark.py`
2. Add quality evaluator if needed in `services/quality.py`
3. Update search space if test requires specific params

### Adding a New Optimizer Phase

1. Add phase to `OptimizationStage` enum
2. Implement phase logic in `AdaptiveOptimizer`
3. Add checkpoint serialization
4. Add resume logic

## Error Handling Principles

1. **Fail fast on configuration errors** - validate early
2. **Graceful degradation** - Web UI read endpoints never raise on LM Studio down
3. **Checkpoint first** - always checkpoint before risky operations
4. **Explicit status codes** - `RunStatus` enum covers all states
5. **User-facing messages** - never expose stack traces or secrets

## Security Considerations

- No credentials in logs (URL sanitization in `_handle_connection_error`)
- Input validation via Pydantic models
- SQL injection prevention via parameterized queries
- Path traversal protection in file serving
- WebSocket origin validation (same-origin)

## Performance Characteristics

- **Speed phase**: ~1-5 min per model (cheap probes)
- **Quality phase**: ~5-20 min per finalist (full suite)
- **Memory**: SQLite + JSON checkpoints, ~10-50 MB per run
- **Concurrency**: Single job lock, sequential model processing

## Future Extension Points

1. **Multi-backend optimization** - compare LM Studio vs Ollama vs llama.cpp
2. **Distributed optimization** - multi-GPU, multi-machine
3. **Continuous optimization** - watch for model updates, re-optimize
4. **Model judge AI** - full LLM-based configuration analysis
5. **Cloud backend** - offload benchmarking to cloud GPUs