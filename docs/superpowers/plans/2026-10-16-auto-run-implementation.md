# Auto-Run & Smart Test Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `lm-optimizer auto-run` command to complete incomplete optimizations, re-run on fingerprint changes, skip completed models, and generate markdown reports. Add separate `lm-optimizer sandbox-duel` command for model-vs-model comparisons.

**Architecture:** Fingerprint-based decision engine extends existing resume/checkpoint infrastructure. New fingerprint service computes SHA256 of (model, quantization, architecture, LM Studio version, hardware, benchmark suite). Auto-run command iterates models, applies decision matrix, calls existing `optimize`/`resume` with revalidate flags. Sandbox duel is a standalone command using existing benchmark service.

**Tech Stack:** Python 3.10+, SQLite (existing), hashlib (SHA256), typer/rich (CLI), existing optimizer/resume pipeline.

**Spec:** `docs/specs/2026-10-15-auto-run-design.md`

## Global Constraints

- Python 3.10+ (existing)
- No schema migrations (fingerprint stored in existing `benchmark_params_json`)
- Backward compatible: runs without fingerprint treated as "unknown" → revalidated
- All existing tests must pass
- TDD: write failing test first, then implementation
- One task = one testable deliverable with its own test cycle

## Review Focus

| Input/Failure Mode | Expected Behavior | Pinned to Task |
|--------------------|-------------------|----------------|
| LM Studio version undetectable | Fingerprint uses "unknown", forces revalidation | Task 2 |
| Hardware detection fails mid-run | Use cached hardware from run + warn | Task 2 |
| Benchmark suite version drift | Date-based version; manual bump when cases change | Task 2 |
| Partial run with fingerprint mismatch | Resume + revalidate only affected configs | Task 4 |
| Model quantization/architecture change | Full re-run (fundamentally different) | Task 4 |
| Concurrent auto-run invocations | Job lock prevents (existing `acquire_lock`) | Task 6 |

---

### Task 1: Fingerprint Service (`fingerprint.py`)

**Files:**
- Create: `lm_optimizer/services/fingerprint.py`
- Test: `lm_optimizer/tests/test_fingerprint.py`

**Interfaces:**
- Consumes: `ModelIdentity`, `HardwareInfo`, `lms_cli.lms_version()`, benchmark suite version constant
- Produces: `compute_fingerprint(...) -> str`, `fingerprint_diff(old: dict, new: dict) -> list[str]`, `FINGERPRINT_FIELDS` constant

- [ ] **Step 1: Write failing test for fingerprint computation**

```python
def test_fingerprint_deterministic():
    """Same inputs always produce same fingerprint."""
    fp1 = compute_fingerprint(model_id="m1", quantization="q4", architecture="arch",
                              lm_studio_version="0.3.12", hardware_fp="abc123",
                              benchmark_version="2026-10-15")
    fp2 = compute_fingerprint(model_id="m1", quantization="q4", architecture="arch",
                              lm_studio_version="0.3.12", hardware_fp="abc123",
                              benchmark_version="2026-10-15")
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA256 hex

def test_fingerprint_changes_on_input_change():
    """Any field change produces different fingerprint."""
    base = dict(model_id="m1", quantization="q4", architecture="arch",
                lm_studio_version="0.3.12", hardware_fp="abc123",
                benchmark_version="2026-10-15")
    fp1 = compute_fingerprint(**base)
    fp2 = compute_fingerprint(**{**base, "lm_studio_version": "0.3.13"})
    assert fp1 != fp2

def test_hardware_fingerprint():
    """Hardware fingerprint from GPU names, VRAM, CPU, RAM."""
    hw = HardwareInfo(os="T", cpu_name="C", cpu_cores_physical=8, cpu_cores_logical=16,
                      total_ram_gb=32.0, gpu_count=1,
                      gpus=[GPUInfo(index=0, name="RTX 3060", vram_gb=6.0, vendor="NVIDIA")])
    fp = hardware_fingerprint(hw)
    assert len(fp) == 16  # prefix
    # Same hardware = same fingerprint
    assert fp == hardware_fingerprint(hw)

def test_fingerprint_diff():
    """Diff returns list of changed field names."""
    old = {"lm_studio_version": "0.3.10", "hardware_fingerprint": "abc"}
    new = {"lm_studio_version": "0.3.12", "hardware_fingerprint": "abc"}
    assert fingerprint_diff(old, new) == ["lm_studio_version"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest lm_optimizer/tests/test_fingerprint.py -v
```
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `fingerprint.py`**

- `FINGERPRINT_FIELDS = ["model_id", "quantization", "architecture", "lm_studio_version", "hardware_fingerprint", "benchmark_suite_version"]`
- `compute_fingerprint(...)` → concatenate with `|` separator → SHA256 hex
- `hardware_fingerprint(HardwareInfo)` → JSON with sorted keys → SHA256 → first 16 chars
- `fingerprint_diff(old, new)` → list of keys where values differ
- `parse_fingerprint(fp: str) -> dict` → reverse lookup for diff (store components in benchmark_params)

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest lm_optimizer/tests/test_fingerprint.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/fingerprint.py lm_optimizer/tests/test_fingerprint.py
git commit -m "feat: add fingerprint service for model/run identification"
```

---

### Task 2: Fingerprint Integration in Optimizer

**Files:**
- Modify: `lm_optimizer/services/optimizer.py` (resume_from_checkpoint, _phase_quality, _finalize)
- Modify: `lm_optimizer/domain/models.py` (add fingerprint property helper)
- Test: `lm_optimizer/tests/test_fingerprint_integration.py`

**Interfaces:**
- Consumes: `fingerprint.compute_fingerprint`, `fingerprint.fingerprint_diff`
- Produces: `resume_from_checkpoint` accepts `revalidate_fingerprint`, `fingerprint_diff`; stores fingerprint in `run.benchmark_params.fingerprint` on first run

- [ ] **Step 1: Write failing test for fingerprint storage on new run**

```python
def test_new_run_stores_fingerprint():
    """First run computes and stores fingerprint in benchmark_params."""
    run = run_repo.get(run_id)
    assert "fingerprint" in run.benchmark_params
    assert run.benchmark_params["fingerprint"] == expected_fp
```

- [ ] **Step 2: Write failing test for resume with fingerprint mismatch revalidation**

```python
def test_resume_revalidates_on_fingerprint_mismatch():
    """When fingerprint differs, completed configs revalidated per diff scope."""
    # Setup: run with old fingerprint, completed configs
    # Trigger: resume with revalidate_fingerprint=True, diff=["hardware_fingerprint"]
    # Assert: _test_config called for completed configs
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest lm_optimizer/tests/test_fingerprint_integration.py -v
```
Expected: FAIL

- [ ] **Step 4: Implement in `optimizer.py`**

1. In `prepare_run`: compute fingerprint → store in `run.benchmark_params["fingerprint"]` and `run.benchmark_params["fingerprint_components"]`
2. In `resume_from_checkpoint`: add `revalidate_fingerprint`, `fingerprint_diff` parameters
3. In `_restore_tested_state`: if `revalidate_fingerprint` and `fingerprint_diff`:
   - For each completed config: check if any diff field affects it
   - If affected: remove from `tested_configs` so it gets re-run
4. In `_phase_quality`: when building `quality_done`, respect revalidated configs
5. In `_finalize`: update stored fingerprint to current

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest lm_optimizer/tests/test_fingerprint_integration.py -v
```
Expected: PASS

- [ ] **Step 6: Run existing resume/checkpoint tests**

```bash
pytest lm_optimizer/tests/test_resume_reconcile.py lm_optimizer/tests/test_pause_resume.py -v
```
Expected: PASS (no regressions)

- [ ] **Step 7: Commit**

```bash
git add lm_optimizer/services/optimizer.py lm_optimizer/domain/models.py lm_optimizer/tests/test_fingerprint_integration.py
git commit -m "feat: integrate fingerprint into optimizer resume/revalidate logic"
```

---

### Task 3: Auto-Run CLI Command

**Files:**
- Create: `lm_optimizer/cli/auto_run.py`
- Modify: `lm_optimizer/cli/main.py` (register command)
- Test: `lm_optimizer/tests/test_auto_run_cli.py`

**Interfaces:**
- Consumes: `run_repo.list_all`, `run_repo.get_by_model`, `fingerprint.compute_fingerprint`, `AdaptiveOptimizer.optimize`, `AdaptiveOptimizer.resume_from_checkpoint`
- Produces: `auto_run(models, options) -> list[AutoRunResult]`, markdown report generation

- [ ] **Step 1: Write failing test for decision matrix**

```python
def test_decision_matrix():
    """Each run state + fingerprint combo produces correct action."""
    cases = [
        (None, None, "FULL"),
        (RunStatus.SUCCESS, True, "SKIP"),
        (RunStatus.SUCCESS, False, "RE-RUN"),
        (RunStatus.INTERRUPTED, True, "RESUME"),
        (RunStatus.INTERRUPTED, False, "RESUME_REVALIDATE"),
        (RunStatus.FAILED, True, "FULL"),
        (RunStatus.FAILED, False, "FULL"),
    ]
    for run_status, fp_match, expected in cases:
        assert decide_action(run_status, fp_match) == expected
```

- [ ] **Step 2: Write failing test for markdown report generation**

```python
def test_report_generation():
    """Report contains summary, per-model sections, fingerprint changes."""
    results = [
        AutoRunResult(model="m1", action="SKIP", fingerprint_match=True, ...),
        AutoRunResult(model="m2", action="RESUME", fingerprint_match=True, ...),
        AutoRunResult(model="m3", action="FULL", fingerprint_match=False, ...),
    ]
    md = generate_report(results, env_info={...})
    assert "## Summary" in md
    assert "m1" in md and "SKIP" in md
    assert "m2" in md and "RESUME" in md
    assert "m3" in md and "FULL" in md
    assert "Fingerprint Changes" in md
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest lm_optimizer/tests/test_auto_run_cli.py -v
```
Expected: FAIL

- [ ] **Step 4: Implement `auto_run.py`**

- `decide_action(run_status, fingerprint_match) -> Action` (enum: FULL, SKIP, RESUME, RESUME_REVALIDATE, RE_RUN)
- `plan_auto_run(models, options) -> list[PlannedAction]` — builds plan with dry-run support
- `execute_auto_run(plan, options) -> list[AutoRunResult]` — sequential execution, progress display
- `generate_report(results, env_info) -> str` — markdown per spec
- `get_env_info() -> dict` — hardware, LM Studio version, benchmark suite version
- Main `auto_run` command function with typer options

- [ ] **Step 5: Register in `main.py`**

```python
from lm_optimizer.cli.auto_run import auto_run
app.command()(auto_run)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest lm_optimizer/tests/test_auto_run_cli.py -v
```
Expected: PASS

- [ ] **Step 7: Integration test with mocked optimizer**

```bash
pytest lm_optimizer/tests/test_auto_run_cli.py::test_full_flow -v
```
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add lm_optimizer/cli/auto_run.py lm_optimizer/cli/main.py lm_optimizer/tests/test_auto_run_cli.py
git commit -m "feat: add auto-run CLI command with fingerprint-aware decisions"
```

---

### Task 4: Sandbox Duel CLI Command

**Files:**
- Create: `lm_optimizer/cli/sandbox_duel.py`
- Modify: `lm_optimizer/cli/main.py` (register command)
- Test: `lm_optimizer/tests/test_sandbox_duel.py`

**Interfaces:**
- Consumes: `run_repo.get_by_model`, `BenchmarkService.run_benchmark`, `QualityEvaluator`
- Produces: `sandbox_duel(model_a, model_b, prompt, options) -> DuelResult`, markdown duel report

- [ ] **Step 1: Write failing test for best config loading**

```python
def test_load_best_config_from_latest_run():
    """Loads auto-best config from latest SUCCESS run."""
    run = create_mock_run(status=RunStatus.SUCCESS, best_config_id=config_id)
    run_repo.save(run)
    config = load_best_config("model-a")
    assert config.context_length == 8192
```

- [ ] **Step 2: Write failing test for duel execution**

```python
def test_duel_runs_prompt_on_both_models():
    """Runs same prompt on both models with their configs."""
    results = run_duel("model-a", "model-b", "test prompt", repetitions=2)
    assert len(results.model_a_runs) == 2
    assert len(results.model_b_runs) == 2
    assert all(r.prompt == "test prompt" for r in results.model_a_runs)
```

- [ ] **Step 3: Write failing test for duel report**

```python
def test_duel_report():
    """Generates markdown with config table, per-run table, verdict."""
    result = DuelResult(...)
    md = generate_duel_report(result)
    assert "## Configurations" in md
    assert "## Verdict" in md
    assert "TTFT" in md
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
pytest lm_optimizer/tests/test_sandbox_duel.py -v
```
Expected: FAIL

- [ ] **Step 5: Implement `sandbox_duel.py`**

- `load_best_config(model_id) -> LoadConfiguration` — from latest SUCCESS run or preset
- `run_duel(model_a, model_b, prompt, options) -> DuelResult` — sequential: load A → run N times → unload → load B → run N times → unload
- `generate_duel_report(result) -> str` — markdown per spec
- Main `sandbox_duel` command with typer options

- [ ] **Step 6: Register in `main.py`**

```python
from lm_optimizer.cli.sandbox_duel import sandbox_duel
app.command()(sandbox_duel)
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest lm_optimizer/tests/test_sandbox_duel.py -v
```
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add lm_optimizer/cli/sandbox_duel.py lm_optimizer/cli/main.py lm_optimizer/tests/test_sandbox_duel.py
git commit -m "feat: add sandbox-duel CLI command for model-vs-model comparison"
```

---

### Task 5: Benchmark Suite Version Constant

**Files:**
- Create: `lm_optimizer/benchmark/suite_version.py`
- Modify: `lm_optimizer/benchmark/suite.py` (export version)
- Test: `lm_optimizer/tests/test_suite_version.py`

**Interfaces:**
- Consumes: benchmark case definitions
- Produces: `BENCHMARK_SUITE_VERSION = "2026-10-15"` (date string)

- [ ] **Step 1: Write failing test**

```python
def test_suite_version_format():
    """Version is YYYY-MM-DD string."""
    from lm_optimizer.benchmark.suite_version import BENCHMARK_SUITE_VERSION
    assert re.match(r"\d{4}-\d{2}-\d{2}", BENCHMARK_SUITE_VERSION)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest lm_optimizer/tests/test_suite_version.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# lm_optimizer/benchmark/suite_version.py
BENCHMARK_SUITE_VERSION = "2026-10-15"  # Update when benchmark cases change
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest lm_optimizer/tests/test_suite_version.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/benchmark/suite_version.py lm_optimizer/tests/test_suite_version.py
git commit -m "feat: add benchmark suite version for fingerprint"
```

---

### Task 6: Job Lock Integration & End-to-End Test

**Files:**
- Modify: `lm_optimizer/cli/auto_run.py` (add job lock)
- Test: `lm_optimizer/tests/test_auto_run_e2e.py`

**Interfaces:**
- Consumes: `acquire_lock`, `release_lock` from `joblock`
- Produces: Auto-run acquires lock for entire run; blocks concurrent runs

- [ ] **Step 1: Write failing test for job lock**

```python
def test_auto_run_acquires_job_lock():
    """Auto-run acquires lock, second invocation exits with message."""
    with acquire_lock("auto-run:test"):
        # Second call should detect lock
        result = run_auto_run(["model1"], non_interactive=True)
        assert "Another optimizer job is already running" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest lm_optimizer/tests/test_auto_run_e2e.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement lock in `auto_run.py`**

```python
def auto_run(...):
    with acquire_lock("auto-run"):
        # entire auto-run flow
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest lm_optimizer/tests/test_auto_run_e2e.py -v
```
Expected: PASS

- [ ] **Step 5: Full integration test (mocked LM Studio)**

```bash
pytest lm_optimizer/tests/test_auto_run_e2e.py::test_full_flow_mocked -v
```
Expected: PASS

- [ ] **Step 6: Run full test suite for regressions**

```bash
pytest lm_optimizer/tests/test_resume_reconcile.py lm_optimizer/tests/test_pause_resume.py lm_optimizer/tests/test_bugfix_pass.py -v
```
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add lm_optimizer/cli/auto_run.py lm_optimizer/tests/test_auto_run_e2e.py
git commit -m "feat: add job lock to auto-run; full integration test"
```

---

### Task 7: Documentation & CLI Help

**Files:**
- Modify: `lm_optimizer/cli/auto_run.py` (help strings)
- Modify: `lm_optimizer/cli/sandbox_duel.py` (help strings)
- Create: `docs/cli/auto-run.md`
- Create: `docs/cli/sandbox-duel.md`

- [ ] **Step 1: Verify all typer options have `help=` strings**

- [ ] **Step 2: Write CLI docs**

```markdown
# auto-run
Complete all incomplete model optimizations...

# sandbox-duel
Compare two models on a prompt...
```

- [ ] **Step 3: Commit**

```bash
git add docs/cli/auto-run.md docs/cli/sandbox-duel.md
git commit -m "docs: add CLI documentation for auto-run and sandbox-duel"
```

---

## Execution Order

```
Task 1 (fingerprint) → Task 2 (optimizer integration) → Task 5 (suite version)
                                                              ↓
Task 3 (auto-run CLI) ← Task 4 (sandbox duel) ← Task 6 (lock + e2e) → Task 7 (docs)
```

Tasks 1, 2, 5 are foundational. 3 and 4 can run in parallel after 1+2. 6 depends on 3+4. 7 last.

---

## Self-Review Checklist

- [x] Spec coverage: All sections (1-7) mapped to tasks
- [x] Step scan: Each step is one test→implement→verify cycle
- [x] Type consistency: `fingerprint_diff` returns `list[str]`, `compute_fingerprint` returns `str`
- [x] Review focus: All 5 failure modes pinned to tasks
- [x] Proportion: Plan is concise; no code bodies except algorithms