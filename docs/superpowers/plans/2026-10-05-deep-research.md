# Deep Research Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Post-optimization long benchmark across user-selected models (each on its best config) with full logs and a leaderboard + output preview.

**Architecture:** New `benchmark/deep_suite.py` (fixed long tasks) + `services/deep.py` (single-model runner, batch runner, leaderboard builder) reusing `BenchmarkService`, `BenchmarkMetrics` (incl. thinking_text), and the full-visibility report helpers; CLI `deep` command; `GET /api/deep/runs` + web deep page reusing results-page patterns.

**Tech Stack:** Python, pytest, typer CLI, FastAPI, vanilla JS.

**Spec:** Conversation 2026-10-05 (decisions: each model on its best config, fixed deep suite, leaderboard + preview, thinking best-effort).

## Global Constraints

- TDD: failing test first for every task.
- Full pytest green before every commit; `node --check` on touched JS.
- Never commit `results/` or `*.db`.
- Each model runs on its stored best config (DB best report/DB lookup); models without a best are skipped with a reason, never defaulted silently.
- Fail-closed unload + verified-empty server between models (`assert_unloaded` before AND after each model).
- Thinking best-effort: `n/a` where absent, never a failure.
- Per-model token/time budget caps (defaults: max 8192 output tokens per task, abort model after 3 consecutive errors).
- Do not touch unrelated dirty working-tree files; commit only task files.
- ASCII-only in reports.

## Review Focus

- Model whose best config no longer loads (server/GUI changed since tuning); expected behavior is skip-with-reason, not a crash of the whole batch.
- 232k-context task on a small-VRAM host; expected behavior is OOM recorded per task, batch continues.
- Batch interrupted overnight (power/process kill); expected behavior is resume skipping completed models on rerun.
- Two models with identical scores; expected behavior is a deterministic tiebreak (gen tok/s, then TTFT), not random order.
- Thinking trace of 100k+ chars; expected behavior is logged fully (user chose full outputs), page preview paginates.

---

### Task 1: Deep suite definition + metrics

**Files:**
- Create: `lm_optimizer/benchmark/deep_suite.py`
- Test: `lm_optimizer/tests/test_deep_suite.py` (create)

**Interfaces:**
- Consumes: `BenchmarkCase` shape from `lm_optimizer/domain/models.py`.
- Produces: `DEEP_CASES: list[BenchmarkCase]` (5 fixed tasks: long-context recall, multi-hop reasoning, JSON discipline, coding precision, instruction following; each with documented prompt, max_tokens, temperature); `deep_metrics(metrics) -> dict` with `recall_accuracy`, `thinking_chars`, `elapsed_s`, `gen_tok_s`.

- [ ] **Step 1: Write the failing test**

```python
def test_deep_suite_has_five_fixed_tasks():
    assert [c.name for c in DEEP_CASES] == ["long_context_recall", "multi_hop", "json_discipline", "coding_precision", "instruction_follow"]
    assert all(c.max_tokens >= 1024 for c in DEEP_CASES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_deep_suite.py -v`
Expected: FAIL with "module not found".

- [ ] **Step 3: Implement `DEEP_CASES` + `deep_metrics()` in `lm_optimizer/benchmark/deep_suite.py`**

Follow existing suite.py case conventions; recall task embeds a synthetic document with planted facts.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_deep_suite.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/benchmark/deep_suite.py lm_optimizer/tests/test_deep_suite.py
git commit -m "feat: add fixed deep-research suite and metrics"
```

### Task 2: Single-model deep runner + CLI

**Files:**
- Create: `lm_optimizer/services/deep.py` (`run_deep_model`, per-model `.md` under `results/deep/`)
- Modify: `lm_optimizer/cli/main.py` (`deep --model <id>` command)
- Test: `lm_optimizer/tests/test_deep_run.py` (create, mocked client)

**Interfaces:**
- Consumes: `BenchmarkService.run_benchmark`, `DEEP_CASES` from Task 1, best-config lookup via `run_repo`/best report.
- Produces: `run_deep_model(client, model_id, load_config) -> dict` with per-task metrics + thinking + scores; report file `results/deep/<model>-deep-<stamp>.md` reusing full-visibility helpers.

- [ ] **Step 1: Write the failing test**

```python
def test_deep_run_uses_best_config_and_logs_thinking(mock_client_with_best):
    out = run_deep_model(mock_client_with_best, "m")
    assert out["tasks"][0]["thinking_chars"] >= 0
    assert "deep" in out["report_path"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_deep_run.py -v`
Expected: FAIL with "function not defined".

- [ ] **Step 3: Implement runner + CLI command following `benchmark` command patterns (prepare_host, unload guard, BenchmarkService)**

No best config → refuse with reason (never silent default).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_deep_run.py lm_optimizer/tests/test_deep_suite.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/deep.py lm_optimizer/cli/main.py lm_optimizer/tests/test_deep_run.py
git commit -m "feat: add single-model deep runner and CLI"
```

### Task 3: Multi-model batch with resume

**Files:**
- Modify: `lm_optimizer/services/deep.py` (`run_deep_batch`), `lm_optimizer/cli/main.py` (`deep --models a b` / `--all-optimized`)
- Test: extend `lm_optimizer/tests/test_deep_run.py`

**Interfaces:**
- Consumes: `run_deep_model` from Task 2.
- Produces: `run_deep_batch(client, model_ids) -> dict` sequential with unload guard between models, per-model budget caps, skip-done resume (completed model reports on disk are reused), batch summary `.md`.

- [ ] **Step 1: Write the failing test**

```python
def test_batch_continues_after_model_failure(mock_client_one_fail):
    out = run_deep_batch(mock_client_one_fail, ["m1", "m2"])
    assert out["models"]["m1"]["status"] == "failed"
    assert out["models"]["m2"]["status"] == "completed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_deep_run.py -v -k batch`
Expected: FAIL.

- [ ] **Step 3: Implement batch loop (resolve best per model → run → unload-verify → next; failures recorded, never abort batch)**

`--all-optimized` resolves models having a stored best config.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_deep_run.py -v`
Expected: PASS, plus full suite green.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/deep.py lm_optimizer/cli/main.py lm_optimizer/tests/test_deep_run.py
git commit -m "feat: add multi-model deep batch with resume"
```

### Task 4: Leaderboard + preview UI and API

**Files:**
- Modify: `lm_optimizer/api/routes.py` (or main.py where JSON APIs live: `GET /api/deep/runs`, `GET /api/deep/runs/{id}`) + `lm_optimizer/api/main.py` (`GET /deep` page route) + `lm_optimizer/ui/static/js/api.js` (client methods) + nav links in templates
- Create: `lm_optimizer/ui/templates/deep.html`, `lm_optimizer/ui/static/js/deep.js`
- Test: extend `lm_optimizer/tests/test_dashboard_api.py`

**Interfaces:**
- Consumes: batch summary + per-model reports from Task 3.
- Produces: leaderboard table (model | gen tok/s | quality | thinking chars | elapsed | verdict) sorted deterministically (score desc, gen tok/s desc, elapsed asc — deep reports record elapsed per task, the honest stand-in for TTFT — then model id); per-model preview (prompt/output/thinking tabs); export link to combined `.md`.

- [ ] **Step 1: Write the failing test**

```python
def test_deep_runs_endpoint_lists_completed_batches(client, seeded_deep):
    data = client.get("/api/deep/runs").json()
    assert seeded_deep in [r["id"] for r in data["runs"]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_dashboard_api.py -v -k deep`
Expected: FAIL.

- [ ] **Step 3: Implement endpoints + page reusing results-page patterns (spinner, error card, History link)**

Preview paginates outputs over 50k chars with a note.

- [ ] **Step 4: Run checks to verify they pass**

Run: `node --check lm_optimizer/ui/static/js/deep.js` and `pytest lm_optimizer/tests/test_dashboard_api.py -q`
Expected: PASS, exit 0.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/api/routes.py lm_optimizer/ui/templates/deep.html lm_optimizer/ui/static/js/deep.js lm_optimizer/tests/test_dashboard_api.py
git commit -m "feat: add deep leaderboard and preview UI"
```
