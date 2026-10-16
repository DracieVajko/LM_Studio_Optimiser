# Auto-Run & Smart Test Completion Design

**Date**: 2026-10-15
**Status**: Approved for implementation

---

## 1. Overview

Add a new CLI command `lm-optimizer auto-run` that completes all incomplete model optimizations, re-runs models when conditions change (LM Studio version, hardware, benchmark suite), and skips models that are already complete under current conditions. Also add a separate `lm-optimizer sandbox-duel` command for model-vs-model comparisons.

---

## 2. Fingerprint Schema

### 2.1 Storage
- Primary: `run.benchmark_params.fingerprint` (JSON string in existing `benchmark_params_json` column)
- Computed on-the-fly for backward compatibility with runs lacking fingerprint

### 2.2 Components (SHA256 hex digest)

| Field | Source | Example |
|-------|--------|---------|
| `model_id` | ModelIdentity | `qwen3-32b-q4_k_m` |
| `quantization` | ModelIdentity | `q4_k_m` |
| `architecture` | ModelIdentity | `qwen3` |
| `lm_studio_version` | `lms_cli.lms_version()` | `0.3.12` |
| `hardware_fingerprint` | HardwareInfo (GPU names+VRAM, CPU cores, RAM) | `a1b2c3d4e5f67890` |
| `benchmark_suite_version` | Date of benchmark case definitions | `2026-10-15` |

**Hardware fingerprint sub-schema**:
```json
{
  "gpus": [{"name": "RTX 3060", "vram_gb": 6.0}],
  "cpu_cores_physical": 8,
  "cpu_cores_logical": 16,
  "total_ram_gb": 32.0
}
```

---

## 3. Auto-Run Command Interface

### 3.1 CLI Definition
```bash
lm-optimizer auto-run [MODELS...] [OPTIONS]
```

### 3.2 Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `MODELS` | args | all discovered | Model IDs or substrings |
| `--profile` | str | "balanced" | Optimization profile |
| `--quality` | float | 0.97 | Minimum quality threshold |
| `--style` | str | "balanced" | Benchmark style |
| `--max-context` | int | 32768 | Max context length |
| `--repetitions` | int | 3 | Benchmark repetitions |
| `--skip` | str | None | Comma-separated substrings to skip |
| `--max-size-gb` | float | None | Only models ≤ this size |
| `--revalidate-fingerprint` | bool | False | Force re-check even if fingerprint matches |
| `--dry-run` | bool | False | Show plan without executing |
| `--output` | Path | "results" | Report output directory |
| `--workload` | str | "interactive" | Workload type |
| `--yes` | bool | False | Skip confirmation |
| `--non-interactive` | bool | False | Never prompt |

### 3.3 Execution Flow

1. Discover models from LM Studio (or use MODELS args)
2. Filter by `--skip`, `--max-size-gb`
3. For each model (sorted by size ascending):
   - Load latest run from DB
   - Compute current fingerprint
   - Apply decision matrix (Section 4)
   - Execute action
4. Generate markdown summary report

---

## 4. Per-Model Decision Logic

### 4.1 Decision Matrix

| Run State | Fingerprint Match | Action |
|-----------|-------------------|--------|
| No run | N/A | **FULL** — fresh `optimize` |
| Complete (SUCCESS/COMPLETED) | ✅ Match | **SKIP** |
| Complete | ❌ Mismatch | **RE-RUN** — full `optimize` |
| Incomplete (RUNNING/PAUSED/INTERRUPTED/RESUMED) | ✅ Match | **RESUME** — `resume_from_checkpoint` |
| Incomplete | ❌ Mismatch | **RESUME + REVALIDATE** |
| FAILED | Any | **FULL** |

### 4.2 Fingerprint Mismatch → Revalidation Scope

| Changed Field | Revalidation Scope |
|---------------|-------------------|
| `lm_studio_version` | All completed configs |
| `hardware_fingerprint` | All completed configs |
| `benchmark_suite_version` | Only new benchmark cases |
| `quantization` / `architecture` | Full re-run |
| `model_id` | N/A (different model) |

### 4.3 Implementation

Extend `resume_from_checkpoint`:
```python
async def resume_from_checkpoint(
    self,
    run_id: Any,
    revalidate: bool = False,
    revalidate_fingerprint: bool = False,  # NEW
    fingerprint_diff: list[str] | None = None,  # NEW
    style: str = "balanced",
) -> OptimizationRun:
```

---

## 5. Markdown Report Format

### 5.1 Output File
`results/auto-run-YYYYMMDD-HHMMSS.md`

### 5.2 Sections

1. **Summary Table** — counts of skipped/resumed/full/re-run/failed
2. **Per-Model Results** — one subsection per model with:
   - Previous run ID & date
   - Fingerprint match/mismatch
   - Action taken
   - Best config & score
3. **Failed Models** (if any)
4. **Fingerprint Changes Table** — what changed per model
5. **Environment Notes** — hardware, LM Studio version, benchmark suite version

### 5.3 Console Live Output
```
Auto-run started: 10 models
  qwen3-32b        ⏭️  SKIP (fingerprint match)
  llama3.1-8b      🔄  RESUME [=======>    ] 7/15
  mistral-7b-v0.3  🆕  FULL   [========>   ] 12/18
  ...
Completed in 2h 34m
Report: results/auto-run-20261015-203000.md
```

---

## 6. Sandbox/Duels — Separate Command

### 6.1 CLI Definition
```bash
lm-optimizer sandbox-duel [OPTIONS]
```

### 6.2 Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model-a` | str | required | First model ID |
| `--model-b` | str | required | Second model ID |
| `--prompt` | str | required | Prompt to test |
| `--config-a` | str | "auto-best" | Config for A |
| `--config-b` | str | "auto-best" | Config for B |
| `--repetitions` | int | 3 | Runs per model |
| `--style` | str | "balanced" | Benchmark style |
| `--output` | Path | "results/duels" | Output directory |

### 6.3 Flow

1. Load best config for each model (from latest SUCCESS run or specified preset)
2. For each model: load → run prompt N times → measure TTFT/tok/s/quality → unload
3. Compare side-by-side
4. Generate duel report

### 6.4 Report Format
`results/duels/duel-modelA-vs-modelB-YYYYMMDD-HHMMSS.md` with:
- Configuration table
- Prompt
- Per-run results table per model
- Median comparison
- Verdict table

---

## 7. Implementation Approach

### 7.1 Files to Modify/Create

**New files**:
- `lm_optimizer/services/fingerprint.py` — fingerprint computation & comparison
- `lm_optimizer/cli/auto_run.py` — auto-run command logic
- `lm_optimizer/cli/sandbox_duel.py` — duel command logic

**Modified files**:
- `lm_optimizer/services/optimizer.py` — add `revalidate_fingerprint` + `fingerprint_diff` to `resume_from_checkpoint`
- `lm_optimizer/cli/main.py` — register new commands
- `lm_optimizer/domain/models.py` — add fingerprint field to `OptimizationRun` (via benchmark_params)

### 7.2 Dependencies

- Uses existing: `resume_from_checkpoint`, `AdaptiveOptimizer`, `BenchmarkService`, `run_repo`
- New: `hashlib` for SHA256, `lms_cli` for version detection

---

## 8. Testing Strategy

### 8.1 Unit Tests
- Fingerprint computation (deterministic, same inputs = same output)
- Fingerprint diff detection
- Decision matrix logic
- Report generation

### 8.2 Integration Tests
- Auto-run dry-run on test models
- Resume + revalidate on fingerprint mismatch
- Sandbox duel with mocked models

### 8.3 Regression
- Existing resume/checkpoint tests must pass
- Existing optimize tests must pass

---

## 9. Backward Compatibility

- Runs without fingerprint: treated as "unknown" → auto-run computes and stores fingerprint on next run
- No schema migration needed (uses existing `benchmark_params_json`)
- All existing CLI commands unchanged

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Fingerprint collision | SHA256 → negligible |
| LM Studio version undetectable | Fallback to "unknown" → forces revalidation |
| Hardware detection fails | Use cached hardware from run + warn |
| Benchmark suite version drift | Date-based version; manual bump when cases change |

---

## Appendix: Spec Self-Review Checklist

- [x] No TBD/TODO placeholders
- [x] Internal consistency: fingerprint fields match decision matrix
- [x] Scope: single command + separate duel command (not a platform)
- [x] No ambiguity: all fields, options, flows explicitly defined
- [x] Backward compatibility addressed
- [x] Testing strategy included