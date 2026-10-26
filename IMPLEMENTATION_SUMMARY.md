# Implementation Summary: LM Studio Auto Optimizer Enhancements

## Overview
Implemented the comprehensive optimizer enhancements as specified in the design document, including pre-run configuration, speed floor enforcement, presets, stage selection, context strategies, and final sandbox validation.

## Changes Made

### 1. Domain Models (`lm_optimizer/domain/models.py`)
- Added new fields to `OptimizationRun`:
  - **Speed floor**: `min_generation_tok_s` (default 2.0), `enforce_min_generation_speed` (default True)
  - **Preset & stage selection**: `run_preset`, `selected_stages`, `max_speed_probes`, `max_finalists`, `benchmark_repetitions_override`, `validation_repetitions_min`, `validation_repetitions_max`, `recovery_budget`
  - **Context strategies**: `context_strategy_modes`
  - **Final sandbox**: `final_sandbox_mode`, `final_sandbox_preset_ids`, `final_sandbox_custom_prompts`, `sandbox_compare_baseline`
  - **Reporting**: `resolved_preset`

### 2. API Schemas (`lm_optimizer/api/schemas.py`)
- Extended `AdvancedSettingsSchema` with all new fields with validation
- Extended `OptimizationRequest` with top-level speed floor and preset fields

### 3. Optimizer Core (`lm_optimizer/services/optimizer.py`)
- **Preset Definitions**: Added `PRESET_DEFINITIONS` for quick/standard/thorough/custom with documented budgets
- **Preset Resolution**: `resolve_preset()` function with override support and warnings
- **Stage Dependency Validation**: `validate_stage_dependencies()` with clear rules
- **Prepare Run**: Updated to resolve preset, apply speed floor, compose phase skips, store resolved budgets
- **Speed Floor Enforcement**: In `_finalize()`, checks `best_config.get_avg_generation_tok_s() >= min_generation_tok_s`; if enforced and not met, sets status to `COMPLETED_NO_WINNER`
- **Context Strategies**: Added `_run_context_strategies()` and `_build_strategy_config()` supporting:
  - `winner`: frozen runtime from Phase A
  - `gpu_resident`: GPU-resident weights + KV on GPU
  - `model_first_kv_cpu`: model on GPU, KV on CPU
  - `system_spillover`: partial GPU offload (CLI channel gated)
- **Final Sandbox**: Added `_run_final_sandbox()` running tasks against actual winner config with optional baseline comparison

### 4. CLI (`lm_optimizer/cli/main.py`)
- Updated `_optimize_advanced()` to accept all new parameters
- Updated `_preflight_optimize()` with validation for:
  - Speed floor (>= 0)
  - Preset (quick/standard/thorough/custom)
  - Stage selection (valid stage names)
  - Context strategies (valid strategy names)
  - Sandbox mode (off/one/all/custom)
- Added all new CLI options to `optimize` command

### 5. API Routes (`lm_optimizer/api/routes.py`)
- No code changes needed - schemas automatically pass new fields through to optimizer

### 6. Documentation
- Created `docs/specs/2026-10-23-optimizer-enhancements-design.md`

## Key Features Implemented

### Speed Floor (Default 2.0 tok/s)
- Default minimum generation speed: 2.0 tok/s (configurable)
- Enforcement flag: `--enforce-min-generation-speed` / `--no-enforce-min-generation-speed`
- Low S0 does NOT abort search; explores high-impact toggles first
- Below-floor candidates cannot become recommended winner when enforced
- Returns explicit `COMPLETED_NO_WINNER` if all candidates below floor
- Failed/empty/timeout samples never become valid 0 tok/s contenders

### Presets
| Preset | Speed Probes | Finalists | Bench Reps | Val Reps (min/max) | Recovery | Context Strategies | Sandbox |
|--------|--------------|-----------|------------|-------------------|----------|-------------------|---------|
| quick  | 12           | 2         | 1          | 2/2               | 1        | winner            | off     |
| standard| 25          | 4         | 3          | 2/5               | 3        | winner, model_first_kv_cpu | off |
| thorough| 40          | 6         | 3          | 3/5               | 5        | winner, gpu_resident, model_first_kv_cpu, system_spillover | all |

### Stage Selection (Custom Mode)
- `--stage` repeatable option: preflight, speed, quality, recovery, validation, context, sandbox, experimental
- Dependency resolution prevents misleading "validated" outcomes
- Quality requires speed (or user-selected config)
- Recovery requires quality failure
- Validation requires quality-passing candidate
- Context requires selected runtime config
- Sandbox requires validated config

### Context Strategies
- Multiple strategies tested independently on winner config
- Each produces separate results with explicit configuration used
- Gated through capability registry
- Results stored in `phase_b_result.context_strategies`

### Final Sandbox
- Runs against actual optimizer winner configuration (not defaults)
- Modes: off, one, all, custom
- Reuses existing sandbox presets (web, research, blender)
- Custom prompts with validation
- Optional baseline comparison
- Results stored separately from optimizer scores

## Testing
- **1285 tests pass** (`pytest tests/ lm_optimizer/tests/`), `python -m compileall lm_optimizer` clean
- `ruff check --select E9,F63,F7,F82,F821` clean on all modified modules
- New CLI options visible in `--help`; invalid choices exit 2 without connecting
- Test files added:
  - `tests/test_speed_floor.py` — defaults, override, disable-vs-zero, below-floor rejection + diagnostic, prompt-vs-generation tok/s, invalid samples, low-S0 non-abort
  - `tests/test_presets_stages.py` — documented budgets, custom overrides (incl. zero budget), dependency warnings, CLI rejection, `prepare_run` freezing of skips/budgets
  - `tests/test_context_strategies.py` — strategy names, KV-on-CPU vs GPU-resident, capability gating, CLI-only spillover
  - `tests/test_final_sandbox.py` — custom-prompt schema/limits, manual-review scores, exact-config load/unload, built-in registry single source, winner-config results isolated from scores/Pareto
  - `tests/test_fingerprint_cache.py` — fingerprint differentiation, hardware signature, revalidation policy, candidate identity for every load parameter, duplicate recognition
  - `tests/test_resume_settings.py` — frozen `resolved_settings` round-trip, legacy runs keep defaults, partial/speed-only runs cannot be validated winners
  - `tests/test_sandbox_isolation.py` — artifact path-traversal containment, unload on generation error, sequential registry-ordered execution, sandbox-off no-op
  - `tests/test_context_capacity_proof.py` — verdict classification per floor, missing telemetry, capacity vs usable at a strict floor, retrieval requirement, sweep refinement/OOM/precheck estimates

## Bugs Found and Fixed During Test Pass
- `resolve_preset("custom", ...)` rejected every override as "unknown" (custom has no base keys) — fixed.
- Custom `--stage` selection was ignored by `prepare_run` (fell back to full campaign) — fixed; explicit `--optimize-context` now also wins over a preset without the context stage.
- `prepare_run` did not persist `run_preset` / ordered `selected_stages` — fixed.
- `--min-generation-tok-s` etc. were not passed from the `optimize` command into `_preflight_optimize`, so CLI validation and the run used defaults — fixed.
- `_run_final_sandbox` imported a non-existent `BUILTIN_TASKS`/`get_task_by_id`/`run_sandbox_task`; winner-config sandbox execution was never runnable — fixed by adding a real registry alias, custom-prompt validator and a load/verify/generate/unload runner.
- Pre-existing `NameError: min_tps` in `lm-optimizer ctx` (used in the sweep call but never declared as an option) — fixed by adding `--min-tps`.
- Speed-floor no-winner runs now record `no_winner_reason="NO_CONFIG_MEETS_SPEED_FLOOR"` and keep the fastest candidate visible in `search_space["speed_floor"]` (it was silently erased).
- Frozen settings were not persisted for resume: new fields fell back to defaults after a restart (a 3.5 tok/s run resumed with 2.0) — fixed with `benchmark_params["resolved_settings"]` + `apply_persisted_run_settings()` on resume.
- `final_recommendation()` could promote a scoreless speed probe when `best_config_id` pointed at it — now requires a scored PASSED candidate.
- Sandbox artifact names could escape the per-task workdir (`../` or absolute paths) — now contained, with a safe fallback to `output.txt`.
- `max_speed_probes`, `validation_repetitions_min/max` and preset `benchmark_repetitions` were stored but never applied — now wired into `_phase_speed`, `_phase_final_validation` and `BenchmarkService.config` respectively.
- Empty CLI lists (`selected_stages=[]`, `context_strategy_modes=[]`) overrode preset defaults, so `--preset quick` ran with no stage plan and `standard` lost its context strategies — empty lists now mean "unset".
- `no_winner_reason` was memory-only (no DB column) — now persisted in `benchmark_params` so reports/resume keep the explicit no-winner state.
- Unload verification treated a stale inventory entry as a hard failure after two instant attempts (cancelled a live run on 2026-10-24) — now three attempts with 1 s / 2.5 s settle delays.

## Files Modified
- `lm_optimizer/domain/models.py`
- `lm_optimizer/api/schemas.py`
- `lm_optimizer/services/optimizer.py`
- `lm_optimizer/services/sandbox_lab.py`
- `lm_optimizer/services/sandbox_runner.py`
- `lm_optimizer/services/run_summary.py`
- `lm_optimizer/cli/main.py`
- `docs/specs/2026-10-23-optimizer-enhancements-design.md` (new)

## Not Yet Implemented (Future Work)
- UI pre-run configuration screen (dashboard.html + app.js): backend/CLI/API are wired, the web form is not.
- Dedicated markdown report sections for speed-floor outcome, context strategies and sandbox results (data is persisted in `search_space` / `benchmark_params`, not yet rendered).
- Hardware validation on a real LM Studio/GPU (all new tests are mocked; no live model loads were performed).