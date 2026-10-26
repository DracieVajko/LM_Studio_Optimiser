# LM Studio Auto Optimizer - Enhancements Design Document

## Overview
This document describes the implementation of enhanced optimization workflows including pre-run configuration, speed floor enforcement, presets, stage selection, context strategies, and final sandbox validation.

## 1. Domain Model Changes

### 1.1 OptimizationRun - New Fields
```python
# Speed floor settings (persisted with run)
min_generation_tok_s: float = 2.0
enforce_min_generation_speed: bool = True

# Preset and stage selection
run_preset: str = "standard"  # quick, standard, thorough, custom
selected_stages: list[str] = []  # e.g., ["speed", "quality", "validation"]
max_speed_probes: int | None = None
max_finalists: int | None = None
benchmark_repetitions: int = 3
validation_repetitions_min: int = 2
validation_repetitions_max: int = 5
recovery_budget: int = 3
quality_threshold: float = 0.97
max_context: int | None = None
optimize_context: bool = False
context_strategy_modes: list[str] = []  # e.g., ["winner", "model_first_kv_cpu"]
final_sandbox_mode: str = "off"  # off, one, all, custom
final_sandbox_preset_ids: list[str] = []
final_sandbox_custom_prompts: list[dict] = []
sandbox_compare_baseline: bool = False
```

### 1.2 AdvancedSettingsSchema - New Fields (API)
Add all the above fields to `AdvancedSettingsSchema` in `api/schemas.py`

### 1.3 Preset Definitions
```python
PRESET_DEFINITIONS = {
    "quick": {
        "max_speed_probes": 12,
        "max_finalists": 2,
        "benchmark_repetitions": 1,
        "validation_repetitions_min": 2,
        "validation_repetitions_max": 2,
        "recovery_budget": 1,
        "selected_stages": ["speed", "quality", "validation"],
        "context_strategy_modes": ["winner"],
        "final_sandbox_mode": "off",
    },
    "standard": {
        "max_speed_probes": 25,
        "max_finalists": 4,
        "benchmark_repetitions": 3,
        "validation_repetitions_min": 2,
        "validation_repetitions_max": 5,
        "recovery_budget": 3,
        "selected_stages": ["speed", "quality", "recovery", "validation"],
        "context_strategy_modes": ["winner", "model_first_kv_cpu"],
        "final_sandbox_mode": "off",
    },
    "thorough": {
        "max_speed_probes": 40,
        "max_finalists": 6,
        "benchmark_repetitions": 3,
        "validation_repetitions_min": 3,
        "validation_repetitions_max": 5,
        "recovery_budget": 5,
        "selected_stages": ["speed", "quality", "recovery", "validation", "context"],
        "context_strategy_modes": ["winner", "gpu_resident", "model_first_kv_cpu", "system_spillover"],
        "final_sandbox_mode": "all",
    },
    "custom": {
        # User-defined values
    }
}
```

## 2. CLI Changes

### 2.1 New Options for `optimize` command
```python
# Speed floor
min_generation_tok_s: float = 2.0
enforce_min_generation_speed: bool = True
no_enforce_min_generation_speed: bool = False  # mutually exclusive flag

# Preset
run_preset: str = "standard"  # quick, standard, thorough, custom

# Stage selection (custom mode)
selected_stages: list[str] = []

# Budget controls
max_speed_probes: int | None = None
max_finalists: int | None = None
benchmark_repetitions: int = 3
validation_repetitions_min: int = 2
validation_repetitions_max: int = 5
recovery_budget: int = 3

# Context strategies
context_strategy_modes: list[str] = []

# Sandbox
final_sandbox_mode: str = "off"
final_sandbox_preset_ids: list[str] = []
final_sandbox_custom_prompts: list[str] = []  # JSON strings
sandbox_compare_baseline: bool = False
```

## 3. API Changes

### 3.1 AdvancedSettingsSchema Updates
Add all new fields with validation.

### 3.2 OptimizationRequest Updates
Add top-level fields for speed floor and preset.

## 4. Optimizer Core Changes

### 4.1 Speed Floor Enforcement
- In `_finalize()`: Check `best_config.get_avg_generation_tok_s() >= min_generation_tok_s`
- If not met and `enforce_min_generation_speed`: set status to `NO_CONFIG_MEETS_SPEED_FLOOR`
- Track rejection reason per candidate

### 4.2 Preset Resolution
- In `prepare_run()`: Resolve preset to concrete budgets
- Merge with user overrides (custom mode)
- Store resolved values in `benchmark_params`

### 4.3 Stage Dependency Resolution
- Validate stage combinations before execution
- Show resolved plan to user
- Prevent misleading "validated" outcomes

### 4.4 Context Strategies
- Extend `_phase_context_optional()` to test multiple strategies
- Each strategy produces separate configuration/results
- Gate modes through capability registry

### 4.5 Final Sandbox Integration
- After `_finalize()`, if `final_sandbox_mode != "off"`:
  - Load winner config
  - Run selected sandbox tasks
  - Record results separately from optimizer scores

## 5. UI Changes

### 5.1 Pre-run Configuration Screen
- Model selection
- Hardware detection display
- Preset selector (quick/standard/thorough/custom)
- Speed floor input (with disable checkbox)
- Stage checkboxes (custom mode)
- Sandbox mode selector
- Cost estimate display

## 6. Database & Persistence

### 6.1 Migration
- Add columns for new fields if needed (or use JSON `benchmark_params`)

### 6.2 Checkpoint
- Store planned stages, completed stage/probe IDs
- Store resolved preset values
- Store context strategy state
- Store pending sandbox tasks

## 7. Reporting

### 7.1 Markdown Report Updates
- Show speed floor and enforcement status
- Show preset and stage plan
- Separate physical max context from usable context
- Show sandbox results separately

## 8. Tests

### 8.1 Unit Tests
- Speed floor behavior (default 2.0, override, disable)
- Preset budget resolution
- Stage dependency resolver
- Context strategy gating
- Sandbox loading of winner config

### 8.2 Integration Tests
- Full quick/standard/thorough runs (mocked)
- Custom stage selection
- Partial runs not marked complete
- Checkpoint/resume with new fields

---

## Implementation Priority

1. **Domain models & API schemas** (foundation)
2. **CLI options & preset resolution** (user-facing)
3. **Optimizer core: speed floor, presets, stage deps** (core logic)
4. **Context strategies** (Phase B enhancement)
5. **Final sandbox integration** (post-optimization)
6. **UI pre-run screen** (frontend)
7. **Reporting updates** (output)
8. **Tests** (validation)

---

## Files to Modify

### Backend
- `lm_optimizer/domain/models.py` - Add fields to OptimizationRun
- `lm_optimizer/api/schemas.py` - Add fields to AdvancedSettingsSchema, OptimizationRequest
- `lm_optimizer/services/optimizer.py` - Core logic changes
- `lm_optimizer/services/context_sweep.py` - Context strategy support
- `lm_optimizer/services/sandbox_lab.py` - Sandbox task loading
- `lm_optimizer/cli/main.py` - CLI options and preflight
- `lm_optimizer/api/routes.py` - API endpoint updates

### Frontend
- `lm_optimizer/ui/templates/dashboard.html` - Pre-run screen
- `lm_optimizer/ui/static/js/app.js` - Pre-run screen logic

### Tests
- `tests/test_speed_floor.py` - New test file
- `tests/test_presets.py` - New test file
- `tests/test_stage_selection.py` - New test file
- `tests/test_context_strategies.py` - New test file
- `tests/test_sandbox_integration.py` - New test file