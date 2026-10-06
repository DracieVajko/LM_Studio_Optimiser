# Context-Sweep Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** New `context-sweep` test escalating context on the stored best config to its absolute maximum, with speed-floor and needle-recall stop conditions plus full per-try visibility.

**Architecture:** New `services/context_probe.py` reusing `context_sweep.sweep()` (geometric + bisect) with a probe that loads the best config at ctx N, runs one short benchmark + one 90%-fill needle recall; CLI `context-sweep` command; report reuses full-visibility helpers; `optimize` gains `--skip-context` (default OFF).

**Tech Stack:** Python, pytest, typer CLI.

**Spec:** Conversation 2026-10-06 (decisions: new command, absolute tok/s floor default 1.0, moderate recall default 0.8, hybrid powers-of-2 then 50% bisect, --skip-context default OFF).

## Global Constraints

- TDD: failing test first for every task.
- Full pytest green before every commit; `node --check` if UI JS touched (no UI changes expected).
- Never commit `results/` or `*.db`.
- No best config in DB → refuse with reason, never silent default.
- Fail-closed unload + verified-empty between probes (`assert_unloaded`).
- ASCII-only in reports.
- Do not touch unrelated dirty working-tree files; commit only task files.

## Review Focus

- 232k-context needle probe on a 6GB-VRAM host; expected behavior is OOM recorded as failed boundary, never a hang (probe timeouts apply).
- Recall denominator when the model truncates the 90% filler; expected behavior is recall computed over asked facts only, with fill-ratio stated.
- `--min-speed` breach on the FIRST probe; expected behavior is immediate stop with reason, not zero-length escalation.
- Backfill-incompatible old runs (no best config); expected behavior is refusal naming the model.
- Filler size estimation error (chars-to-tokens heuristic); expected behavior is measured prompt_tokens reported alongside the 90% target, never claimed exact.

---

### Task 1: Needle probe service

**Files:**
- Create: `lm_optimizer/services/context_probe.py`
- Test: `lm_optimizer/tests/test_context_probe.py` (create)

**Interfaces:**
- Consumes: `context_sweep.sweep/probe contract {ctx, ok, tok_s, error}`, `BenchmarkService` public surface, `deep_suite.RECALL_FACTS` pattern for planted facts.
- Produces: `build_filler(target_chars: int) -> str` (scaled synthetic doc with K planted facts); `run_needle(client, model_id, load_config, ctx, fill_ratio=0.9) -> dict` with `{recall, asked, fill_chars, prompt_tokens}`; `probe_context(client, model_id, load_config, ctx, min_speed, min_recall) -> dict` with `{ctx, ok, tok_s, prompt_tok_s, ttft_ms, recall, error}`.

- [ ] **Step 1: Write the failing test**

```python
async def test_probe_stops_below_speed_floor(mock_client_slow):
    out = await probe_context(mock_client_slow, "m", cfg(), 8192, min_speed=1.0, min_recall=0.8)
    assert out["ok"] is False and "speed" in out["error"]
```

NOTE: probe functions are async (the sweep() contract needs an awaitable); `asyncio_mode = "auto"` is set, so plain `async def test_` works.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_context_probe.py -v`
Expected: FAIL with "function not defined".

- [ ] **Step 3: Implement filler + needle + probe in `lm_optimizer/services/context_probe.py`**

One load per probe, unload in finally; recall = hits/asked via case-insensitive substring; ok requires load ok AND tok_s >= min_speed AND recall >= min_recall.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_context_probe.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/context_probe.py lm_optimizer/tests/test_context_probe.py
git commit -m "feat: add context needle probe service"
```

### Task 2: CLI context-sweep command

**Files:**
- Modify: `lm_optimizer/cli/main.py` (new `context-sweep` command)
- Test: extend `lm_optimizer/tests/test_context_probe.py`

**Interfaces:**
- Consumes: `probe_context` from Task 1, `resolve_best_config` pattern from `services/deep.py`, `context_sweep.sweep` + `format_report`.
- Produces: `context-sweep --model <id> --max-context N [--min-speed 1.0] [--min-recall 0.8] [--skip-fill-test] [--output results]` printing per-try table live + saving report.

- [ ] **Step 1: Write the failing test**

```python
def test_sweep_refuses_without_best_config(mock_client_no_best):
    with pytest.raises(NoBestConfigError):
        resolve_best_for_sweep(mock_client_no_best, "ghost-model")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_context_probe.py -v -k refuse`
Expected: FAIL.

- [ ] **Step 3: Implement command following `deep` command patterns (prepare_host, verified_empty, per-try table, summary, post assert_unloaded)**

Geometric ladder from best-config ctx to --max-context, then bisect refinement via sweep(); stop early on floor breaches per probe contract.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_context_probe.py -v`
Expected: PASS, plus full suite green.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/cli/main.py lm_optimizer/tests/test_context_probe.py
git commit -m "feat: add context-sweep CLI command"
```

### Task 3: Sweep report with per-try table + full outputs

**Files:**
- Modify: `lm_optimizer/services/reporting.py` (new `save_context_report`), reuse `_all_configs_rows`/`_config_outputs` style helpers
- Test: extend `lm_optimizer/tests/test_context_probe.py`

**Interfaces:**
- Consumes: sweep result dict + per-try probe dicts from Task 2.
- Produces: `save_context_report(model_id, probes, trio, out_dir) -> Path` writing `<model>-context-<stamp>.md` with per-try table (ctx | gen tok/s | prompt tok/s | TTFT | recall | status | error), trio verdicts (max stable / perf-optimal / balanced), fill-ratio notes, full verbatim outputs appendix.
- Base branch is `main` (no dependency on unmerged full-reports/deep branches): if `_all_configs_rows`/`_config_outputs` helpers are absent there, implement the needed minimal local rendering in the report function instead of importing them.

- [ ] **Step 1: Write the failing test**

```python
def test_context_report_lists_every_try():
    path = save_context_report("m", [probe_ok(4096), probe_fail(8192)], trio(), tmp_path)
    text = path.read_text()
    assert "4096" in text and "8192" in text and "Maximum stable context" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_context_probe.py -v -k report`
Expected: FAIL.

- [ ] **Step 3: Implement `save_context_report()`**

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_context_probe.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/reporting.py lm_optimizer/tests/test_context_probe.py
git commit -m "feat: add context-sweep report with per-try table"
```

### Task 4: --skip-context flag + docs

**Files:**
- Modify: `lm_optimizer/cli/main.py` (`optimize --skip-context`, default OFF, skips ctx-dimension candidates)
- Modify: `docs/OPTIMIZATION_METHOD.md` (context-sweep section: usage, defaults, two-phase flow)
- Test: extend `lm_optimizer/tests/test_context_probe.py` (flag constrains generated contexts)

**Interfaces:**
- Consumes: search-space context candidate generation.
- Produces: identical results with flag OFF (byte-identical default path); with flag ON, context candidates collapse to the fixed small set.

- [ ] **Step 1: Write the failing test**

```python
def test_skip_context_flag_collapses_candidates():
    space = generate_space(skip_context=True)
    assert space == sorted(set(space)) and len(space) <= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_context_probe.py -v -k skip`
Expected: FAIL.

- [ ] **Step 3: Implement flag + docs section**

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_context_probe.py -v`
Expected: PASS, plus full suite green.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/cli/main.py docs/OPTIMIZATION_METHOD.md lm_optimizer/tests/test_context_probe.py
git commit -m "feat: add --skip-context flag and context-sweep docs"
```
