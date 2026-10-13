# Full Report Visibility + Deep Research Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full report visibility (all configs with full outputs in .md and web UI) and Deep Research benchmark (separate CLI command + UI tab with default/custom prompts)

**Architecture:** Two independent but related features: (1) Extend `reporting.py` to include all-configs table + full outputs in .md reports, add all-configs table to Results page via existing API; (2) New `deep_research.py` service with built-in + YAML-overridable prompts, `deep-research` CLI command, extend `/deep` page with new tab for running prompts sequentially per model, per-model preview with Prompt/Thinking/Output tabs and pagination.

**Tech Stack:** Python (FastAPI, Typer, PyYAML), vanilla JS (ES modules), Jinja2 templates, pytest

**Spec:** `C:\Users\DracieVajko\.opencode\plan\2025-10-05-full-reports-deep-research-design.md`

## Global Constraints

- TDD: failing test first for every task
- ASCII-only in reports (existing convention)
- Never commit `results/` or `*.db`
- Fail-closed unload between model runs (existing `assert_unloaded`)
- Backward compatibility: existing reports render unchanged; existing CLI commands unchanged
- No new REST endpoints — reuse existing `/api/runs/{id}/configurations` and `/api/deep/runs*`
- Never touch unrelated dirty working-tree files (README.md, run-web.bat, docs/* deletions)

## Review Focus

1. **Large .md files** — 50+ configs × 5 tests × full outputs = multi-MB .md; memory-safe streaming writes
2. **Web UI performance** — 200-row cap on all-configs table; lazy-load per-test outputs on expand
3. **Deep Research YAML parsing** — graceful fallback when override file missing/malformed
4. **Sequential model runs** — fail-closed unload between models; batch never aborts early
5. **Prompt/Thinking/Output escaping** — markdown fences must not break (reuse `_fence_block`)
6. **Backfill CLI** — `re-report` must not modify DB; skip corrupt rows with warning
6. **Deep Research .md parsing** — API parses .md for leaderboard/preview; tripwire tests catch format drift
7. **Unicode/escaping** — ASCII-only structure; verbatim model text preserved

---

### Task 1: All-configs table in .md reports

**Files:**
- Modify: `lm_optimizer/services/reporting.py` (add `_all_configs_rows`, wire into `save_best_report`)
- Create: `lm_optimizer/tests/test_full_reports.py`

**Interfaces:**
- Consumes: `ConfigurationResult` list with `config`, `status`, `score`, `metrics`, `error`, `context_length`
- Produces: `_all_configs_rows(configs: list, best_id) -> list[str]` — markdown table rows

- [ ] **Step 1: Write the failing test**

```python
def test_all_configs_table_lists_every_config():
    rows = _all_configs_rows([passed_cfg(9.1), failed_cfg(), passed_cfg(7.7)], best_id=passed_cfg(9.1).id)
    text = "\n".join(rows)
    assert text.count("|") > 0
    assert text.index("9.1") < text.index("7.7") < text.index("load_failed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_full_reports.py -v`
Expected: FAIL with "function not defined"

- [ ] **Step 3: Implement `_all_configs_rows()` in `lm_optimizer/services/reporting.py`**

Columns: `# | Ctx | Flash | KV | Eval | Phys | Parallel | Checkpoints | Experts | Status | Score | Gen tok/s | Quality | Error`
- Score-descending for passed, failed at bottom
- `n/a` for missing metrics; error trimmed to 120 chars
- Pipe-escape via `_cell`; reuse `_fmt` for display

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_full_reports.py lm_optimizer/tests/test_reporting_transparency.py -v`
Expected: PASS

- [ ] **Step 5: Wire into `save_best_report` after winner sections**

Insert after "All tried configurations" header (around line 840)

- [ ] **Step 6: Commit**

```bash
git add lm_optimizer/services/reporting.py lm_optimizer/tests/test_full_reports.py
git commit -m "feat: add all-tried-configs table to best report"
```

---

### Task 2: Full verbatim outputs appendix in .md reports

**Files:**
- Modify: `lm_optimizer/services/reporting.py` (add `_full_outputs_section`, wire into `save_best_report` + `save_failed_report`)
- Test: extend `lm_optimizer/tests/test_full_reports.py`

**Interfaces:**
- Consumes: `BenchmarkMetrics` (test_name, prompt, thinking_text, output_text)
- Produces: `_full_outputs_section(configs) -> list[str]` — per-config `### Config {id}` sections with per-test Prompt/Thinking/Output verbatim

- [ ] **Step 1: Write the failing test**

```python
def test_outputs_appendix_includes_thinking_and_output():
    lines = _full_outputs_section([cfg_with_thinking()])
    text = "\n".join(lines)
    assert "Thinking" in text and "test thinking trace" in text
    assert "test output text" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_full_reports.py::test_outputs_appendix_includes_thinking_and_output -v`
Expected: FAIL

- [ ] **Step 3: Implement `_full_outputs_section()` and `_config_outputs()`**

Reuse `_fence_block` (escapes inner ``` only); empty thinking → `n/a (non-reasoning or unrecorded)`
No metrics → `no measurements recorded (status)` line

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_full_reports.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/reporting.py lm_optimizer/tests/test_full_reports.py
git commit -m "feat: append full per-config outputs to reports"
```

---

### Task 3: All-configs table on web Results page

**Files:**
- Modify: `lm_optimizer/ui/static/js/results.js` (add `renderAllConfigs(runId)`)
- Test: `node --check lm_optimizer/ui/static/js/results.js` + extend `lm_optimizer/tests/test_dashboard_api.py` if needed

**Interfaces:**
- Consumes: existing `GET /api/runs/{id}/configurations` response
- Produces: rendered table with same columns as .md; failed rows show status+error; 200-row cap + "showing X of Y" note; JSON detail links

- [ ] **Step 1: Write the failing check**

```python
def test_configurations_endpoint_lists_all_statuses(client, seeded_run):
    data = client.get(f"/api/runs/{seeded_run}/configurations").json()
    assert {c["status"] for c in data["configurations"]} >= {"passed", "load_failed"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_dashboard_api.py -v -k configurations`
Expected: FAIL (or JS function missing)

- [ ] **Step 3: Implement `renderAllConfigs(runId)` in `results.js`**

Reuse existing styles; add expandable rows with per-test Prompt/Thinking/Output (lazy-load from existing `/results/{run_id}/configs/{config_id}` JSON detail); 200-row cap

- [ ] **Step 4: Run checks to verify they pass**

Run: `node --check lm_optimizer/ui/static/js/results.js` and `pytest lm_optimizer/tests/test_dashboard_api.py -q`
Expected: PASS, exit 0

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/ui/static/js/results.js lm_optimizer/tests/test_dashboard_api.py
git commit -m "feat: show all tried configs on web results page"
```

---

### Task 4: Backfill CLI command (`re-report`)

**Files:**
- Modify: `lm_optimizer/cli/main.py` (add `re-report` command)
- Test: extend `lm_optimizer/tests/test_full_reports.py`

**Interfaces:**
- Consumes: `run_repo` listing, `save_best_report`/`save_failed_report` from Tasks 1-2
- Produces: `re-report [--model MODEL] [--all]` regenerating reports from DB

- [ ] **Step 1: Write the failing test**

```python
def test_rereport_regenerates_from_stored_run(stored_run):
    out = invoke_rereport(model=stored_run.model.id)
    assert "regenerated 1" in out and report_file_exists(stored_run)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_full_reports.py -v -k rereport`
Expected: FAIL

- [ ] **Step 3: Implement `re-report` command**

Iterate runs via `run_repo`; call `save_best_report` if winner exists else `save_failed_report`; print summary `regenerated N, skipped M (reason)`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_full_reports.py -v` + full suite
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/cli/main.py lm_optimizer/tests/test_full_reports.py
git commit -m "feat: add re-report backfill command"
```

---

### Task 5: Deep Research suite + metrics

**Files:**
- Create: `lm_optimizer/benchmark/deep_suite.py`
- Test: `lm_optimizer/tests/test_deep_suite.py`

**Interfaces:**
- Consumes: `BenchmarkCase` shape from `lm_optimizer/domain/models.py`
- Produces: `DEEP_RESEARCH_CASES: list[BenchmarkCase]` (5 fixed tasks), `deep_metrics(metrics) -> dict`

- [ ] **Step 1: Write the failing test**

```python
def test_deep_suite_has_five_fixed_tasks():
    assert [c.name for c in DEEP_RESEARCH_CASES] == ["long_context_recall", "multi_hop", "json_discipline", "coding_precision", "instruction_follow"]
    assert all(c.max_tokens >= 1024 for c in DEEP_RESEARCH_CASES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_deep_suite.py -v`
Expected: FAIL with "module not found"

- [ ] **Step 3: Implement `DEEP_RESEARCH_CASES` + `deep_metrics()`**

Follow `suite.py` conventions; recall task embeds synthetic document with planted facts

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_deep_suite.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/benchmark/deep_suite.py lm_optimizer/tests/test_deep_suite.py
git commit -m "feat: add fixed deep-research suite and metrics"
```

---

### Task 6: Deep Research service + single-model runner + CLI

**Files:**
- Create: `lm_optimizer/services/deep_research.py`
- Modify: `lm_optimizer/cli/main.py` (add `deep-research` command)
- Test: `lm_optimizer/tests/test_deep_research.py`

**Interfaces:**
- Consumes: `BenchmarkService.run_benchmark`, `DEEP_RESEARCH_CASES`, best-config lookup via `run_repo`
- Produces: `run_deep_research(client, model_id, prompts, out_dir) -> dict` with per-task metrics + thinking + scores; `.md` under `results/deep/`

- [ ] **Step 1: Write the failing test**

```python
def test_deep_research_uses_best_config_and_logs_thinking(mock_client_with_best):
    out = run_deep_research(mock_client_with_best, "m", [])
    assert out["tasks"][0]["thinking_chars"] >= 0
    assert "deep" in out["report_path"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_deep_research.py -v`
Expected: FAIL with "function not defined"

- [ ] **Step 3: Implement service + CLI command**

- `DEEP_RESEARCH_PROMPTS` built-in list (5 prompts covering recall, reasoning, JSON, coding, instruction)
- Load YAML override from `deep_prompts.yaml` (CWD → config dir)
- `resolve_best_config(model_id)` via `run_repo.get_by_model` + `get_best_config()`
- `run_deep_research_async(client, model_id, prompts, out_dir, context)` — single load, run all cases, unload, fail-closed
- CLI: `deep-research --models a,b,c [--prompts-file deep_prompts.yaml] [--out-dir results/deep] [--context N]`
- Refuse with reason when no best config (never silent default)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_deep_research.py lm_optimizer/tests/test_deep_suite.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/deep_research.py lm_optimizer/cli/main.py lm_optimizer/tests/test_deep_research.py
git commit -m "feat: add deep-research service and CLI"
```

---

### Task 7: Deep Research multi-model batch + resume

**Files:**
- Modify: `lm_optimizer/services/deep_research.py` (add `run_deep_batch`), `lm_optimizer/cli/main.py` (extend `deep-research`)
- Test: extend `lm_optimizer/tests/test_deep_research.py`

**Interfaces:**
- Consumes: `run_deep_research_async` from Task 6
- Produces: `run_deep_batch(client, model_ids, prompts, out_dir)` — sequential, unload guard, skip-done resume, batch summary `.md`

- [ ] **Step 1: Write the failing test**

```python
def test_batch_continues_after_model_failure(mock_client_one_fail):
    out = run_deep_batch(mock_client_one_fail, ["m1", "m2"], prompts)
    assert out["models"]["m1"]["status"] == "failed"
    assert out["models"]["m2"]["status"] == "completed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_deep_research.py -v -k batch`
Expected: FAIL

- [ ] **Step 3: Implement batch loop**

Load once per model → run all prompts → unload → next model
Skip models with existing `-deep-` report (resume)
`--all-optimized` resolves models with stored best config
UnloadNotClean per model recorded; dirty host skips remaining models

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_deep_research.py -v` + full suite
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/deep_research.py lm_optimizer/cli/main.py lm_optimizer/tests/test_deep_research.py
git commit -m "feat: add multi-model deep-research batch with resume"
```

---

### Task 8: Deep Research leaderboard + preview UI + API

**Files:**
- Modify: `lm_optimizer/api/routes.py` (add `/api/deep/research/runs*`), `lm_optimizer/api/main.py` (add `/deep` page route if needed)
- Create: `lm_optimizer/ui/templates/deep_research.html`, `lm_optimizer/ui/static/js/deep_research.js`
- Test: extend `lm_optimizer/tests/test_dashboard_api.py`

**Interfaces:**
- Consumes: batch summary + per-model reports from Task 7
- Produces: leaderboard sorted by (score, gen tok/s, elapsed, model id); per-model preview with Prompt/Thinking/Output tabs, 50k-char pagination; export combined `.md`

- [ ] **Step 1: Write the failing test**

```python
def test_deep_research_runs_endpoint_lists_completed_batches(client, seeded_deep_batch):
    data = client.get("/api/deep/research/runs").json()
    assert seeded_deep_batch in [r["id"] for r in data["runs"]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_dashboard_api.py -v -k deep`
Expected: FAIL

- [ ] **Step 3: Implement endpoints + page**

- `GET /api/deep/research/runs` — list batch stamps from `results/deep/deep-batch-*.md`
- `GET /api/deep/research/runs/{id}` — deterministic sort (score, gen tok/s, elapsed, model id); parse per-model `.md` for preview
- `GET /api/deep/research/runs/{id}/export` — combined `.md` as `text/markdown`
- Page reuses `results.html` patterns (spinner, error card, History link)
- Preview paginates >50k chars with note; HTML-escaped model text

- [ ] **Step 4: Run checks to verify they pass**

Run: `node --check lm_optimizer/ui/static/js/deep_research.js` and `pytest lm_optimizer/tests/test_dashboard_api.py -q`
Expected: PASS, exit 0

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/api/routes.py lm_optimizer/api/main.py lm_optimizer/ui/templates/deep_research.html lm_optimizer/ui/static/js/deep_research.js lm_optimizer/tests/test_dashboard_api.py
git commit -m "feat: add deep-research leaderboard and preview UI"
```

---

### Task 9: Deep Research default prompts YAML + example

**Files:**
- Create: `deep_prompts.yaml.example` (in repo root)
- Modify: `lm_optimizer/services/deep_research.py` (load override)

**Interfaces:**
- Consumes: YAML file at CWD → `config/` → bundled defaults
- Produces: merged prompt list (override replaces by name)

- [ ] **Step 1: Create `deep_prompts.yaml.example`**

```yaml
prompts:
  - name: "Custom Recall Test"
    prompt: "Read this document and answer: ..."
    category: "recall"
    max_tokens: 8192
```

- [ ] **Step 2: Implement loader**

Search order: `./deep_prompts.yaml` → `config/deep_prompts.yaml` → bundled defaults
Override by `name`; log which source used

- [ ] **Step 3: Verify**

Run: `pytest lm_optimizer/tests/test_deep_research.py -v -k prompts`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add deep_prompts.yaml.example lm_optimizer/services/deep_research.py
git commit -m "feat: add deep-research default prompts with YAML override"
```

---

### Task 10: Final integration + cleanup

**Files:**
- Run full suite, verify no regressions
- Update `docs/OPTIMIZATION_METHOD.md` with Deep Research usage

**Interfaces:**
- Consumes: all previous tasks
- Produces: working feature set

- [ ] **Step 1: Full test suite**

Run: `python -m pytest lm_optimizer/tests -q`
Expected: 550+ passed

- [ ] **Step 2: UI sanity check**

Run: `node --check` on all `lm_optimizer/ui/static/js/*.js`
Expected: exit 0

- [ ] **Step 3: Manual smoke test**

`deep-research --models <model> --out-dir /tmp/test-deep` (mocked client in CI, real for local)

- [ ] **Step 4: Commit docs update**

```bash
git add docs/OPTIMIZATION_METHOD.md
git commit -m "docs: add Deep Research usage to OPTIMIZATION_METHOD"
```

- [ ] **Step 5: Final commit + tag**

```bash
git commit -m "feat: full report visibility + deep research benchmark"
git tag v1.5.0-beta.1
```

---

## Execution Handoff

This plan has 10 tasks with clear dependencies (1→2, 1→3, 5→6, 6→7, 7→8). Tasks 1-4 (Full Reports) are independent of 5-9 (Deep Research) and can run in parallel. Recommend **Subagent-driven** for thoroughness — each task gets fresh context and review. The stacked branches (`feat/full-reports` already merged to main, `feat/deep-research` pending) mean deep research tasks depend on full-reports helpers (`_all_configs_rows`, `_config_outputs`), so implement in order or rebase onto updated main.