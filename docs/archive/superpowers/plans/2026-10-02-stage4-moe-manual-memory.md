# Stage-4 Grid + MoE + Manual Memory Duel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify stage-4 sweep values, gate MoE minimally, run experimental tail opt-in at the end, and add a manual mmap/keep duel with keep/revert verdict.

**Architecture:** Keep the staged hierarchical search (no Cartesian products); extend only allow-listed REST keys; experimental levers (RoPE, CPU-MoE, speculative) run only after the frozen winner as opt-in; mmap/mlock/keep stay MANUAL_ONLY with a guided CLI duel that re-measures and compares against the auto best.

**Tech Stack:** Python, pytest, typer CLI, REST `LoadConfiguration.API_KEYS`, `lms` CLI read-only paths.

**Spec:** User decisions 2026-10-02 (6 points: manual mmap/keep duel, MoE-only minimal, RoPE experimental prompt, draft list optional at end, overrides read-only, concurrency decided here) + `docs/OPTIMIZATION_METHOD.md` + live registry `lm_optimizer/services/parameter_registry.py`.

## Global Constraints

- 445/445 pytest must stay green; run `pytest` before every commit.
- Run `node --check` equivalent for touched JS (`websocket.js`) when UI changes; no UI changes in this plan.
- Never commit `results/` or `*.db` (gitignored, local only).
- Never inject undocumented REST keys; `LoadConfiguration.to_api_params()` is the only sender.
- Fail-closed unload guard on every load path (`assert_unloaded` before AND after).
- TDD: failing test first for every task.
- `MAX_PLAN_PROBES = 25` in `lm_optimizer/services/speed_search.py:18` must hold.
- `try_mmap` / `keep_model_in_memory` / `n_threads` stay MANUAL_ONLY, never auto-probed.
- `llama_cpp_arguments_override` JSON is read-only introspection, never written programmatically.
- RoPE / CPU-MoE / speculative runs are marked `is_experimental = True`.
- Keep/revert threshold: manual keeps only on >= +5% generation tok/s; TTFT is tiebreak.

## Review Focus

- 35B MoE at ctx 262144 with eval 2048 OOMs the host; a reasonable person expects the run to record OOM and continue, never hang.
- `interactive` workload with `parallel=8` requested silently degrades latency; expected behavior is parallel stays 1 unless explicit override or throughput workload.
- Manual duel run when the user did NOT actually toggle mmap in the GUI; expected behavior is the verdict says "no measurable difference, revert guidance stands" rather than claiming a tuning win.
- Draft model listed as recommended but not downloaded locally; expected behavior is skip with reason, never auto-download.
- Best `.md` missing or from another model; expected behavior is the duel refuses with "no matching auto best" instead of comparing across models.

---

### Task 1: Stage-4 grid unification

**Files:**
- Modify: `lm_optimizer/services/search_space.py:221-274`
- Modify: `lm_optimizer/services/speed_search.py:119-140` (no-op if S2/S3 alts already match — do not touch cosmetically)
- Modify: `lm_optimizer/tests/test_workload.py` only if it asserts the old generator default (stale assertion update)
- Test: `lm_optimizer/tests/test_stage4_grid.py` (create)

**Interfaces:**
- Consumes: `LoadConfiguration`, `LMStudioCapabilities`, `apply_to_space(parallels, workload, explicit_override)` from `lm_optimizer/services/workload.py:47`.
- Produces: `_generate_batch_sizes() -> list[int]` including 2048; `_generate_parallels()` including 8 (throughput/explicit only); `build_speed_plan()` still `<= 25` probes.

- [ ] **Step 1: Write the failing test**

```python
def test_eval_grid_includes_2048():
    space = SearchSpaceGenerator(client_with(caps_eval=True)).generate(model, hw, profile)
    assert 2048 in space.batch_sizes

def test_parallel_8_throughput_only():
    assert 8 in SearchSpaceGenerator(c).generate(m, h, p, {"workload_type": "throughput"}).parallels
    assert SearchSpaceGenerator(c).generate(m, h, p, {"workload_type": "interactive"}).parallels == [1]

def test_extended_values_opt_in_only():
    space = SearchSpaceGenerator(c).generate(m, h, p, {"physical_batches": [128, 256, 512, 1024, 2048]})
    assert 128 in space.physical_batch_sizes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_stage4_grid.py -v`
Expected: FAIL with "function not defined" or assertion on missing 2048/8.

- [ ] **Step 3: Implement unified grid in `lm_optimizer/services/search_space.py`**

Unify `_generate_batch_sizes` default to `[64, 128, 256, 512, 1024, 2048]`; `_generate_parallels` default `[1, 2, 4, 8]` filtered through `apply_to_space`; `_generate_physical_batch_sizes` / `_generate_context_checkpoints` accept `physical_batches` / `checkpoints` overrides, defaults stay `[256, 512, 1024]` / `[16, 32, 64]`; keep `build_speed_plan` S2/S3 alts (`eval 64,128,256,1024,2048`, `parallel 1,2,8`, `physical 256,1024`, `checkpoints 16,64`) and `MAX_PLAN_PROBES` bound.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest lm_optimizer/tests/test_stage4_grid.py lm_optimizer/tests/test_speed_search.py lm_optimizer/tests/test_workload.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/search_space.py lm_optimizer/tests/test_stage4_grid.py
git commit -m "feat: unify stage-4 grid with 2048 eval and throughput-gated parallel 8"
```

### Task 2: MoE minimal gated

**Files:**
- Modify: `lm_optimizer/services/search_space.py:276-296`
- Modify: `lm_optimizer/services/speed_search.py:101-103`
- Test: `lm_optimizer/tests/test_moe_minimal.py` (create)

**Interfaces:**
- Consumes: `ModelIdentity.is_moe`, `num_experts`, `SpeedCaps(is_moe, rest_keys)` from Task 1.
- Produces: expert probe set of max 3 values (`default, default//2, default//4`, deduped, anchor excluded); `n-cpu-moe` stays manual unless a verified channel exists.

- [ ] **Step 1: Write the failing test**

```python
def test_non_moe_probes_no_experts():
    probes, _ = build_speed_plan(anchor(), 2048, caps(is_moe=False))
    assert all(p.config.num_experts is None for p in probes)

def test_moe_max_three_expert_probes():
    # NOTE: _clone propagates the anchor's num_experts to every probe (S0 == anchor),
    # so count only the S1 expert-variation set (anchor value excluded).
    probes, _ = build_speed_plan(anchor(num_experts=8), 2048, caps(is_moe=True))
    varied = [p for p in probes if "MoE expert count" in p.reason and p.config.num_experts != 8]
    assert len(varied) <= 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_moe_minimal.py -v`
Expected: FAIL (too many or ungated expert probes).

- [ ] **Step 3: Implement MoE gate in `lm_optimizer/services/speed_search.py` and `lm_optimizer/services/search_space.py`**

Gate `_generate_expert_counts` and the S1 expert loop on `model.is_moe and caps.supports_num_experts`; cap at 3 probes; document `num_cpu_expert_layers_ratio` as experimental/manual until a REST/CLI path verifies (no blind injection).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_moe_minimal.py lm_optimizer/tests/test_speed_search.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/search_space.py lm_optimizer/services/speed_search.py lm_optimizer/tests/test_moe_minimal.py
git commit -m "feat: gate MoE probes on is_moe with max-3 expert set"
```

### Task 3: Experimental tail opt-ins

**Files:**
- Modify: `lm_optimizer/cli/main.py` (auto/optimize entry: add `--enable-rope/--enable-cpu-moe/--enable-speculative` default False + interactive prompt)
- Modify: `lm_optimizer/services/optimizer.py` (thread flags into `advanced_settings`, set `run.is_experimental`)
- Test: `lm_optimizer/tests/test_experimental_optins.py` (create)

**Interfaces:**
- Consumes: Task 1-2 space/plan builders.
- Produces: `parse_experimental_flags(enable_rope: bool, enable_cpu_moe: bool, enable_speculative: bool) -> dict` with all-False defaults; `apply_experimental_mark(run, flags) -> run` setting `is_experimental`; experimental tail runs only after FINAL winner.

- [ ] **Step 1: Write the failing test**

```python
def test_experimental_defaults_off():
    assert parse_experimental_flags(False, False, False) == {"enable_rope": False, "enable_cpu_moe": False, "enable_speculative": False}

def test_experimental_marks_run():
    run = apply_experimental_mark(run_fixture(), {"enable_rope": True})
    assert run.is_experimental is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_experimental_optins.py -v`
Expected: FAIL with "function not defined".

- [ ] **Step 3: Implement `parse_experimental_flags()` and run marking**

Prompt text: `"Test experimental RoPE / CPU-MoE / speculative?"` (typer appends the `[no]` default); any True sets `run.is_experimental = True` with reason string; tail stages execute only post-FINAL.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_experimental_optins.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/cli/main.py lm_optimizer/services/optimizer.py lm_optimizer/tests/test_experimental_optins.py
git commit -m "feat: add experimental opt-in prompts defaulting to off"
```

### Task 4: Draft discovery + recommended list

**Files:**
- Modify: `lm_optimizer/services/speculative.py:18-32`
- Test: `lm_optimizer/tests/test_speculative_drafts.py` (create)

**Interfaces:**
- Consumes: model list from `list_models()`.
- Produces: `RECOMMENDED_DRAFTS: dict[str, list[str]]` per family; `discover_drafts(models) -> list` extended; `find_local_draft(models, target_id) -> str | None` returning None when absent (never downloads).

- [ ] **Step 1: Write the failing test**

```python
def test_discovers_suffixed_draft():
    assert discover_drafts([m("Qwen3-0.6B"), m("Qwen3-32B")]) != []

def test_missing_draft_returns_none():
    assert find_local_draft([m("Qwen3-32B")], "Qwen3-32B") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_speculative_drafts.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement extended hints and `find_local_draft()` in `lm_optimizer/services/speculative.py`**

Extend `DRAFT_HINTS` with small-model hints (`0.6b, 0.5b, 1b, tiny, draft`); add `RECOMMENDED_DRAFTS = {"qwen": ["Qwen3-0.6B"], "default": []}` (names only, no download); `ab_compare` runs only when `find_local_draft` returns a key, else returns `{"ran": False, "reason": "no local draft"}`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_speculative_drafts.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/speculative.py lm_optimizer/tests/test_speculative_drafts.py
git commit -m "feat: extend draft discovery with recommended list, skip when absent"
```

### Task 5: Manual memory duel CLI

**Files:**
- Create: `lm_optimizer/services/manual_memory_duel.py`
- Modify: `lm_optimizer/cli/main.py` (add `manual-memory-duel` command)
- Test: `lm_optimizer/tests/test_manual_memory_duel.py` (create)

**Interfaces:**
- Consumes: `save_best_report()` path `results/<model>-best.md`, `BenchmarkService.run_benchmark()`.
- Produces: `verdict(auto_gen: float, manual_gen: float, auto_ttft: float, manual_ttft: float, threshold: float = 0.05) -> dict` with `{decision: "keep" | "revert", reason: str}`; CLI `manual-memory-duel --model <id> --stage mmap|keep`.

- [ ] **Step 1: Write the failing test**

```python
def test_keep_on_6pct_gain():
    assert verdict(10.0, 10.6, 500.0, 490.0)["decision"] == "keep"

def test_revert_on_noise():
    assert verdict(10.0, 10.2, 500.0, 510.0)["decision"] == "revert"

def test_model_mismatch_refuses():
    with pytest.raises(ValueError):
        load_auto_best("other-model-best.md", "this-model")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_manual_memory_duel.py -v`
Expected: FAIL with "function not defined".

- [ ] **Step 3: Implement `verdict()` + `load_auto_best()` in `lm_optimizer/services/manual_memory_duel.py` and wire CLI**

CLI flow: print auto best summary → pause `"Switch mmap OFF in LM Studio GUI (default ON), then press Enter"` → re-measure same `LoadConfiguration` → `verdict()` → print `"Keep OFF"` or `"Revert to ON"` → one lever per invocation + printed hint to run the other stage; fail-closed unload before/after; model-id must match `.md`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_manual_memory_duel.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/manual_memory_duel.py lm_optimizer/cli/main.py lm_optimizer/tests/test_manual_memory_duel.py
git commit -m "feat: add manual-memory-duel with keep/revert verdict"
```

### Task 6: Docs + reporting notes

**Files:**
- Modify: `lm_optimizer/services/model_recommendations.py:18-37`
- Modify: `lm_optimizer/services/reporting.py:64-86`
- Modify: `docs/OPTIMIZATION_METHOD.md`
- Run: `scripts/generate_parameter_matrix.py`
- Test: extend `lm_optimizer/tests/test_reporting_transparency.py`

**Interfaces:**
- Consumes: outputs of Tasks 1-5.
- Produces: updated `manual_checklist()` (mmap/keep steps), `_prune_notes()` (stage-4 + experimental wording), regenerated `docs/LM_STUDIO_PARAMETER_MATRIX.md`.

- [ ] **Step 1: Write the failing test**

```python
def test_prune_notes_mention_manual_duel():
    assert any("manual-memory-duel" in n or "mmap" in n for n in _prune_notes(run, configs))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_reporting_transparency.py -v`
Expected: FAIL on missing wording.

- [ ] **Step 3: Update checklist, prune notes, and method docs**

Add mmap OFF + keep-in-memory duel steps to `manual_checklist`; clarify S2/S3 tried-values wording and experimental tail in `reporting.py`; document new flags and duel flow in `OPTIMIZATION_METHOD.md`; regenerate the parameter matrix script.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_reporting_transparency.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/model_recommendations.py lm_optimizer/services/reporting.py docs/OPTIMIZATION_METHOD.md docs/LM_STUDIO_PARAMETER_MATRIX.md
git commit -m "docs: document stage-4 grid, experimental tail and manual duel"
```
