# Full Report Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every report shows everything tried — all configs with scores, speeds, outputs, and errors — in `.md` and web UI, backfilled for old runs.

**Architecture:** One shared renderer (`_all_configs_rows` + `_config_outputs`) used by best and failed reports; web results page consumes the existing `GET /runs/{run_id}/configurations` endpoint; a `re-report` CLI command regenerates reports from stored DB runs.

**Tech Stack:** Python, pytest, typer CLI, vanilla JS (results.js).

**Spec:** Conversation 2026-10-05 (decisions: FULL outputs in `.md`, table in `.md` + web UI, backfill old runs from DB).

## Global Constraints

- TDD: failing test first for every task.
- Full pytest green before every commit; `node --check` on touched JS.
- Never commit `results/` or `*.db`.
- Failed-at-load rows show status + error only (no metrics exist — state it, never invent).
- Full outputs included verbatim for every metric of every config that has them; report header states the size.
- Do not touch unrelated dirty working-tree files (README.md, run-web.bat, test_web_resilience.py, docs deletions/archive); commit only task files.
- ASCII-only in reports (existing convention).

## Review Focus

- Run with 50+ configs producing a multi-MB `.md`; a reasonable person expects the file to still open and the table to render, not a crash.
- Config whose metrics contain undecodable/control characters; expected behavior is sanitized output, not broken markdown tables (pipe-escaping).
- Backfill on a run whose DB rows are corrupt/missing; expected behavior is skip-with-count, not abort.
- Web results page for a run with 100+ configs; expected behavior is a rendered table (paginated or capped with a note), not a frozen browser.
- Failed report for a run where the closest config has zero metrics; expected behavior is status/error rows, not an exception.

---

### Task 1: All-configs table in best report

**Files:**
- Modify: `lm_optimizer/services/reporting.py` (new `_all_configs_rows(configs, best_id)`, wire into `save_best_report`)
- Test: `lm_optimizer/tests/test_full_reports.py` (create)

**Interfaces:**
- Consumes: `ConfigurationResult` fields (config, status, score, get_avg_generation_tok_s, quality_score, error, context_length).
- Produces: `_all_configs_rows(configs: list, best_id) -> list[str]` — markdown table `| # | Ctx | Flash | KV | Eval | Phys | Parallel | Checkpoints | Experts | Status | Score | Gen tok/s | Quality | Error |`, score-descending, failed rows bottom with status + trimmed error;ISSING metrics render as `n/a`.

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
Expected: FAIL with "function not defined".

- [ ] **Step 3: Implement `_all_configs_rows()` in `lm_optimizer/services/reporting.py` and call it from `save_best_report` after the winner parameter table**

Pipe-escape cell text; error trimmed to 120 chars; score formatted `%.3f` or `n/a`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_full_reports.py lm_optimizer/tests/test_reporting_transparency.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/reporting.py lm_optimizer/tests/test_full_reports.py
git commit -m "feat: add all-tried-configs table to best report"
```

### Task 2: Full outputs appendix in reports

**Files:**
- Modify: `lm_optimizer/services/reporting.py` (new `_config_outputs(config)`, wire into `save_best_report` + `save_failed_report`)
- Test: extend `lm_optimizer/tests/test_full_reports.py`

**Interfaces:**
- Consumes: `BenchmarkMetrics` (test_name, prompt, output_text, thinking_text, success, generation_tok_s) from Task 1 context.
- Produces: `_config_outputs(config) -> list[str]` — per-test `###` sections with Prompt / Thinking / Output verbatim; empty thinking renders `n/a (non-reasoning or unrecorded)`; configs with no metrics render one `no measurements recorded (status)` line.

- [ ] **Step 1: Write the failing test**

```python
def test_outputs_appendix_includes_thinking_and_output():
    lines = _config_outputs(cfg_with_thinking())
    text = "\n".join(lines)
    assert "Thinking" in text and "test thinking trace" in text
    assert "test output text" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_full_reports.py::test_outputs_appendix_includes_thinking_and_output -v`
Expected: FAIL.

- [ ] **Step 3: Implement `_config_outputs()` and append per-config sections after the table in both report savers**

Fence outputs in code blocks; escape inner fences.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_full_reports.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/reporting.py lm_optimizer/tests/test_full_reports.py
git commit -m "feat: append full per-config outputs to reports"
```

### Task 3: All-configs table on web results page

**Files:**
- Modify: `lm_optimizer/ui/static/js/results.js` (run view: fetch `GET /runs/{id}/configurations`, render table, link each row to existing `/results/{run_id}/configs/{config_id}` JSON detail)
- Test: `node --check lm_optimizer/ui/static/js/results.js` + extend `lm_optimizer/tests/test_dashboard_api.py` only if the endpoint needs changes (it exists — prefer no backend change)

**Interfaces:**
- Consumes: existing `GET /api/runs/{run_id}/configurations` response shape from Task 1 data.
- Produces: rendered table with identical columns to the `.md` table; rows for failed configs show status + error; empty run shows "no configurations recorded".

- [ ] **Step 1: Write the failing check**

```python
def test_configurations_endpoint_lists_all_statuses(client, seeded_run):
    data = client.get(f"/api/runs/{seeded_run}/configurations").json()
    assert {c["status"] for c in data["configurations"]} >= {"passed", "load_failed"}
```

(If the endpoint already satisfies this, assert the JS function name exists instead — no backend change.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_dashboard_api.py -v -k configurations`
Expected: FAIL (or JS function missing — document which).

- [ ] **Step 3: Implement `renderAllConfigs(runId)` in `results.js` wired into the run view, reusing existing styles and the JSON detail links**

Cap initial render at 200 rows with a "showing X of Y" note when larger.

- [ ] **Step 4: Run checks to verify they pass**

Run: `node --check lm_optimizer/ui/static/js/results.js` and `pytest lm_optimizer/tests/test_dashboard_api.py -q`
Expected: PASS, exit 0.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/ui/static/js/results.js lm_optimizer/tests/test_dashboard_api.py
git commit -m "feat: show all tried configs on web results page"
```

### Task 4: Backfill CLI command

**Files:**
- Modify: `lm_optimizer/cli/main.py` (new `re-report [--model] [--all]` command)
- Test: extend `lm_optimizer/tests/test_full_reports.py`

**Interfaces:**
- Consumes: `run_repo` listing, `save_best_report` / `save_failed_report` from Tasks 1-2.
- Produces: `re-report` regenerating reports for stored runs (winner runs → best report, winnerless → failed report); prints `regenerated N, skipped M (reason)`; never modifies the DB.

- [ ] **Step 1: Write the failing test**

```python
def test_rereport_regenerates_from_stored_run(stored_run):
    out = invoke_rereport(model=stored_run.model.id)
    assert "regenerated 1" in out and report_file_exists(stored_run)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_full_reports.py -v -k rereport`
Expected: FAIL.

- [ ] **Step 3: Implement `re-report` following existing CLI command patterns (run lookup, per-run saver dispatch, summary line)**

Corrupt/missing rows skip with count, never abort.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_full_reports.py -v`
Expected: PASS, plus full suite green.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/cli/main.py lm_optimizer/tests/test_full_reports.py
git commit -m "feat: add re-report backfill command"
```
