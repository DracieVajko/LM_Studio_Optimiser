# LLM Judge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an AI-powered "second opinion" feature that evaluates optimization results and recommends per-model winners, overall winner, task-specific guidance, and avoid-list — with LOCAL (LM Studio), API (OpenRouter), and HEURISTIC backends.

**Architecture:** Three-layer: API endpoints → JudgeService (orchestrates data loading, prompt building, backend call, DB persistence) → ModelJudge backends (LocalJudge, APIJudge, HeuristicJudge). PromptBuilder constructs summary table for ALL configs + full detail for top-N + optional reasoning matrix.

**Tech Stack:** Python 3.11+, httpx, sqlite3, json, vanilla JS frontend. No new dependencies.

**Spec:** `docs/specs/2026-10-19-llm-judge-design.md`

## Global Constraints

- No new Python dependencies (uses existing `httpx`, `sqlite3`, `json`)
- Frontend: vanilla JS, no new libraries
- LM Studio must be running for LOCAL backend
- OpenRouter API key for API backend (user-provided)
- All DB writes use existing `run_repo` / `config_repo` patterns
- Follow existing code style: type hints, dataclasses, async/await
- Unit tests write artifacts only to `tmp_path`
- Commits only with explicit user approval

## Review Focus

| Input / Condition | Expected Behavior | Test Location |
|---|---|---|
| LM Studio not running (LOCAL backend) | Clear error, suggest starting LM Studio or switching backend | Task 4 (LocalJudge) |
| Prompt exceeds model context (8k tokens) | Auto-reduce `top_n_detail` until fits, warn user | Task 3 (PromptBuilder) |
| Judge returns invalid JSON | Retry once with stricter prompt, then fallback to heuristic | Task 4 (JudgeService) |
| No configurations in run | Return error "No configurations to evaluate" | Task 2 (JudgeService data loading) |
| Cross-model with mixed reasoning sweeps | Build matrix per model, include in prompt | Task 3 (PromptBuilder) |

---

### Task 1: Database Migration — `judge_verdicts` Table

**Files:**
- Create: `lm_optimizer/database/migrations/008_judge_verdicts.py`
- Modify: `lm_optimizer/database/manager.py` (add migration to list)
- Test: `tests/test_db_migration_judge.py`

**Interfaces:**
- Produces: `judge_verdicts` table with schema per spec §6

- [ ] **Step 1: Write failing test**

```python
def test_judge_verdicts_table_exists(tmp_path):
    from lm_optimizer.database.manager import DatabaseManager
    db = DatabaseManager(tmp_path / "test.db")
    db.initialize()
    con = db.connect()
    cols = [row[1] for row in con.execute("PRAGMA table_info(judge_verdicts)")]
    expected = {"id", "run_ids_json", "config_ids_json", "use_case", "custom_prompt",
                "priority", "backend", "judge_model", "verdict_json", "prompt_tokens",
                "completion_tokens", "latency_ms", "status", "error_message", "created_at"}
    assert expected.issubset(set(cols))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db_migration_judge.py::test_judge_verdicts_table_exists -v`
Expected: FAIL (table doesn't exist)

- [ ] **Step 3: Implement migration in `lm_optimizer/database/migrations/008_judge_verdicts.py`**

```python
def migrate(con):
    con.execute("""
        CREATE TABLE judge_verdicts (
            id TEXT PRIMARY KEY,
            run_ids_json TEXT NOT NULL,
            config_ids_json TEXT,
            use_case TEXT NOT NULL,
            custom_prompt TEXT,
            priority TEXT NOT NULL,
            backend TEXT NOT NULL,
            judge_model TEXT NOT NULL,
            verdict_json TEXT NOT NULL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            latency_ms INTEGER,
            status TEXT NOT NULL DEFAULT 'completed',
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.execute("CREATE INDEX idx_judge_verdicts_created ON judge_verdicts(created_at DESC)")
    con.execute("CREATE INDEX idx_judge_verdicts_run ON judge_verdicts(run_ids_json)")
```

Register in `manager.py` migrations list.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_db_migration_judge.py::test_judge_verdicts_table_exists -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/database/migrations/008_judge_verdicts.py lm_optimizer/database/manager.py tests/test_db_migration_judge.py
git commit -m "feat: add judge_verdicts table migration"
```

---

### Task 2: JudgeService — Data Loading & Orchestration

**Files:**
- Create: `lm_optimizer/services/judge_service.py`
- Test: `tests/test_judge_service.py`

**Interfaces:**
- Consumes: `run_repo`, `config_repo` (existing repos)
- Produces: `async def run_judge(request: JudgeRequest) -> JudgeVerdict`

- [ ] **Step 1: Write failing tests**

```python
def test_load_run_data_returns_configs_with_metrics(tmp_path):
    from lm_optimizer.services.judge_service import JudgeService
    # setup: create run + 3 configs with metrics_json containing prompt_text/output_text
    svc = JudgeService(db_path=tmp_path / "test.db")
    data = await svc._load_run_data(["run-uuid-1"])
    assert len(data) == 1
    assert len(data[0]["configs"]) == 3
    for cfg in data[0]["configs"]:
        assert "metrics_json" in cfg
        assert "quality_json" in cfg
        assert "config_json" in cfg

def test_load_run_data_filters_by_config_ids(tmp_path):
    # when config_ids provided, only those configs returned
    ...

def test_load_run_data_empty_run_raises(tmp_path):
    # run with no configs → ValueError
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_judge_service.py -v`
Expected: FAIL (module/class not found)

- [ ] **Step 3: Implement `JudgeService` in `lm_optimizer/services/judge_service.py`**

```python
class JudgeService:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    async def _load_run_data(self, run_ids: list[str], config_ids: list[str] | None = None) -> list[dict]:
        # 1. For each run_id: get run row (model, quality_threshold, started_at)
        # 2. Query configurations where run_id IN (...) AND status IN ('passed','quality_failed','load_failed')
        # 3. If config_ids: filter to those
        # 4. For each config: parse config_json, metrics_json, quality_json
        # 5. Return list of {run_id, model_id, quality_threshold, configs: [...]}
        pass

    async def run_judge(self, request: JudgeRequest) -> JudgeVerdict:
        # 1. Load data via _load_run_data
        # 2. Build prompt via PromptBuilder (Task 3)
        # 3. Select backend (LocalJudge/APIJudge/HeuristicJudge) per request.backend
        # 4. Call backend.evaluate(comparison)
        # 5. Persist verdict to judge_verdicts table
        # 6. Return verdict
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_judge_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/judge_service.py tests/test_judge_service.py
git commit -m "feat: add JudgeService with data loading"
```

---

### Task 3: PromptBuilder — Summary + Top-N Detail + Reasoning Matrix

**Files:**
- Create: `lm_optimizer/services/judge_prompt.py`
- Test: `tests/test_judge_prompt.py`

**Interfaces:**
- Consumes: List of run data from `JudgeService._load_run_data`
- Produces: `str` (full prompt) via `PromptBuilder.build(run_data, request) -> str`

- [ ] **Step 1: Write failing tests**

```python
def test_build_summary_table_all_configs():
    from lm_optimizer.services.judge_prompt import PromptBuilder
    run_data = [{"model_id": "m1", "configs": [
        {"id": "c1", "config_json": {"context_length": 4096, "gpu_ratio": 0.8, ...},
         "metrics_json": [{"generation_tok_s": 10.5, "prompt_tok_s": 100, "estimated_ttft_ms": 50}],
         "quality_json": {"overall": 0.95, "checks_passed": 28, "checks_total": 30,
                          "by_test": {"coding_task": 0.9, "structured_output": 0.95, "medium_reasoning": 0.98}},
         "stability_score": 0.9, "peak_vram_gb": 5.2, "score": 0.8}
    ]}]
    req = JudgeRequest(top_n_detail=2, include_reasoning_matrix=False)
    prompt = PromptBuilder.build(run_data, req)
    assert "SUMMARY TABLE (ALL configs)" in prompt
    assert "m1" in prompt
    assert "10.5" in prompt
    assert "0.95" in prompt

def test_build_top_n_detail_includes_prompt_output():
    # top_n_detail=1 → includes prompt_text[:500] and output_text[:500] for top config
    ...

def test_build_reasoning_matrix_groups_by_base_runtime():
    # configs with same base runtime but different reasoning=off/low/medium/high/xhigh
    # → matrix table with 5 rows per base config
    ...

def test_build_auto_reduces_top_n_when_prompt_too_long():
    # mock tokenizer: if prompt > 8000 tokens, reduce top_n_detail until fits
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_judge_prompt.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `PromptBuilder` in `lm_optimizer/services/judge_prompt.py`**

```python
class PromptBuilder:
    @staticmethod
    def build(run_data: list[dict], request: JudgeRequest) -> str:
        # 1. Flatten all configs across runs
        # 2. If fastest_context_preset: keep only fastest passed config at max ctx per model
        # 3. Build summary table (ALL configs): Model | Config | ctx | tok/s | prompt tok/s | TTFT | Quality | VRAM | Stability | Key Tests
        # 4. Sort by score desc, take top_n_detail for full detail
        # 5. For each top config: runtime params, avg metrics, quality breakdown, stability, VRAM, sample prompt/output
        # 6. If include_reasoning_matrix: group by (model, base_runtime) where base_runtime = config_json minus generation.reasoning
        #    Build matrix table per group
        # 7. Format system + hardware + tables + task instruction
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_judge_prompt.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/judge_prompt.py tests/test_judge_prompt.py
git commit -m "feat: add PromptBuilder with reasoning matrix support"
```

---

### Task 4: ModelJudge Backends — LocalJudge, APIJudge, HeuristicJudge

**Files:**
- Modify: `lm_optimizer/services/model_judge.py` (extend existing)
- Test: `tests/test_model_judge.py`

**Interfaces:**
- Consumes: `ModelComparison` (existing dataclass), `JudgeConfig` (existing dataclass)
- Produces: `JudgeVerdict` (existing dataclass) via `async evaluate(comparison) -> JudgeVerdict`

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_heuristic_judge_returns_valid_verdict():
    from lm_optimizer.services.model_judge import HeuristicJudge, ModelComparison
    judge = HeuristicJudge()
    comparison = ModelComparison(...)  # minimal valid fixture
    verdict = await judge.evaluate(comparison)
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.confidence == 0.6
    assert "Heuristic" in verdict.reasoning
    await judge.close()

@pytest.mark.asyncio
async def test_local_judge_calls_lm_studio_chat_completions(httpx_mock):
    from lm_optimizer.services.model_judge import LocalJudge
    httpx_mock.add_response(json={"choices": [{"message": {"content": '{"recommended_config_id":"c1","reasoning":"test","confidence":0.8,"pros":[],"cons":[],"alternative_config_ids":[],"warnings":[]}'}}]})
    judge = LocalJudge(JudgeConfig(backend=JudgeBackend.LOCAL, model="test-model"))
    verdict = await judge.evaluate(comparison_fixture)
    assert verdict.recommended_config_id == "c1"
    # verify POST to http://127.0.0.1:1234/v1/chat/completions with correct payload
    await judge.close()

@pytest.mark.asyncio
async def test_api_judge_calls_openrouter(httpx_mock):
    # similar, with Authorization header and api_base
    ...

@pytest.mark.asyncio
async def test_local_judge_fallback_on_json_parse_error(httpx_mock):
    # first response: invalid JSON → retry with stricter prompt → second response: valid
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model_judge.py -v`
Expected: FAIL

- [ ] **Step 3: Implement backends in `lm_optimizer/services/model_judge.py`**

```python
class LocalJudge(ModelJudge):
    def __init__(self, config: JudgeConfig):
        self.config = config
        self.client = httpx.AsyncClient(base_url="http://127.0.0.1:1234/v1", timeout=60.0)

    async def evaluate(self, comparison: ModelComparison) -> JudgeVerdict:
        # 1. Build prompt via PromptBuilder (pass comparison data + request params)
        # 2. POST /chat/completions with response_format={"type": "json_object"}
        # 3. Parse JSON → JudgeVerdict
        # 4. On JSON decode error: retry once with "Return ONLY valid JSON" appended
        # 5. Return verdict
        pass

    async def close(self):
        await self.client.aclose()

class APIJudge(ModelJudge):
    def __init__(self, config: JudgeConfig):
        self.config = config
        base = config.api_base or "https://openrouter.ai/api/v1"
        self.client = httpx.AsyncClient(base_url=base, timeout=60.0,
            headers={"Authorization": f"Bearer {config.api_key}"})

    async def evaluate(self, comparison: ModelComparison) -> JudgeVerdict:
        # Same as LocalJudge but with API headers
        pass

class HeuristicJudge(ModelJudge):
    async def evaluate(self, comparison: ModelComparison) -> JudgeVerdict:
        # 1. For each config: normalize tok/s, quality, stability to 0-1
        # 2. Weight by priority (speed/quality/balanced/context)
        # 3. Pick highest score as winner
        # 4. Build per-model winners, overall winner, task breakdown
        # 5. Return JudgeVerdict with confidence=0.6, reasoning="Heuristic scoring: ..."
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_judge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/model_judge.py tests/test_model_judge.py
git commit -m "feat: implement LocalJudge, APIJudge, HeuristicJudge backends"
```

---

### Task 5: API Endpoints — POST /api/judge, GET /api/judge/{id}, GET /api/judge/history

**Files:**
- Modify: `lm_optimizer/ui/web.py` (add routes)
- Test: `tests/test_judge_api.py`

**Interfaces:**
- Consumes: `JudgeService.run_judge` (Task 2)
- Produces: JSON responses per spec §5

- [ ] **Step 1: Write failing tests**

```python
def test_post_judge_returns_judge_id(client, tmp_path):
    # setup: db with 1 run + configs
    resp = client.post("/api/judge", json={
        "run_ids": ["run-uuid"],
        "use_case": "coding",
        "priority": "balanced",
        "backend": "heuristic",
        "judge_model": "heuristic",
        "top_n_detail": 3
    })
    assert resp.status_code == 202
    data = resp.json()
    assert "judge_id" in data
    assert data["status"] == "pending"

def test_post_judge_wait_true_returns_verdict(client, tmp_path):
    resp = client.post("/api/judge?wait=true", json={...})
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"
    assert "verdict" in resp.json()

def test_get_judge_by_id(client, tmp_path):
    # insert verdict row, then GET
    ...

def test_get_judge_history(client, tmp_path):
    # insert 3 verdicts, GET /api/judge/history?limit=20
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_judge_api.py -v`
Expected: FAIL

- [ ] **Step 3: Implement routes in `lm_optimizer/ui/web.py`**

```python
@app.post("/api/judge")
async def judge_submit(request: JudgeRequest, wait: bool = False, background: BackgroundTasks = None):
    # 1. Validate request (Pydantic model)
    # 2. If wait: run synchronously, return verdict
    # 3. Else: generate judge_id, store pending status, background.add_task(run_judge_async), return judge_id
    pass

@app.get("/api/judge/{judge_id}")
async def judge_get(judge_id: str):
    # Query judge_verdicts table, return verdict_json
    pass

@app.get("/api/judge/history")
async def judge_history(limit: int = 20):
    # Query judge_verdicts ORDER BY created_at DESC LIMIT ?
    pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_judge_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/ui/web.py tests/test_judge_api.py
git commit -m "feat: add judge API endpoints"
```

---

### Task 6: UI — Results Page Checkbox Column + Ask Judge Button + Modals

**Files:**
- Modify: `lm_optimizer/ui/static/js/results.js`
- Modify: `lm_optimizer/ui/templates/results.html` (if separate template)
- Test: `tests/test_judge_ui.py` (Playwright or static analysis)

**Interfaces:**
- Consumes: `/api/models` (for LOCAL judge model dropdown), `/api/judge` endpoints
- Produces: Interactive judge modal, verdict display modal, history modal

- [ ] **Step 1: Write failing test (Playwright)**

```python
def test_results_page_has_judge_button_and_checkboxes(page):
    page.goto("http://localhost:8080/results")
    assert page.locator("th:has-text('Judge')").is_visible()
    assert page.locator("button:has-text('Ask Judge')").is_visible()
    # checkbox in header selects all
    page.locator("th input[type=checkbox]").click()
    assert page.locator("tbody input[type=checkbox]").all()[0].is_checked()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_judge_ui.py::test_results_page_has_judge_button_and_checkboxes -v`
Expected: FAIL

- [ ] **Step 3: Implement UI in `lm_optimizer/ui/static/js/results.js`**

```javascript
// 1. Add checkbox column to configurations table (header + each row)
// 2. "Ask Judge" button: disabled until ≥1 checked OR "no selection = all" logic
// 3. Judge Modal: use-case dropdown, priority radio, backend radio, judge model select,
//    API key/base inputs (conditional), top-N slider, reasoning matrix checkbox,
//    fastest-context checkbox, Run Judge button
// 4. On Run Judge: POST /api/judge, poll /api/judge/{id} until completed
// 5. Verdict Modal: tabs (Overall, Per-Model, By Task, Avoid, Raw JSON)
// 6. History Modal: list from /api/judge/history, click to open verdict
// 7. Toast notifications for pending/completed/error
```

Follow existing patterns in `results.js` (e.g., `renderConfigurations`, `showModal`, `toast`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_judge_ui.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/ui/static/js/results.js tests/test_judge_ui.py
git commit -m "feat: add Judge UI to Results page"
```

---

### Task 7: Integration Test — Full Flow Heuristic Backend

**Files:**
- Test: `tests/test_judge_integration.py`

**Interfaces:**
- Consumes: All previous tasks
- Produces: End-to-end verification

- [ ] **Step 1: Write failing test**

```python
@pytest.mark.asyncio
async def test_full_judge_flow_heuristic(tmp_path):
    # 1. Create DB with 2 runs × 3 configs each (different models)
    # 2. POST /api/judge with heuristic backend, use_case=coding
    # 3. Poll /api/judge/{id} until completed
    # 4. Verify verdict structure: per_model_winner, overall_winner, task_breakdown, avoid, warnings
    # 5. Verify DB row in judge_verdicts
    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_judge_integration.py -v`
Expected: FAIL

- [ ] **Step 3: Fix any integration issues (no new code, just wiring)**

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_judge_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_judge_integration.py
git commit -m "test: add judge integration test"
```

---

### Task 8: Polish — Token Estimation, Error Messages, Documentation

**Files:**
- Modify: `lm_optimizer/services/judge_prompt.py` (add token estimation)
- Modify: `lm_optimizer/services/judge_service.py` (better error handling)
- Modify: `lm_optimizer/ui/static/js/results.js` (help tooltips)
- Test: `tests/test_judge_polish.py`

- [ ] **Step 1: Add token estimation to PromptBuilder**

```python
@staticmethod
def estimate_tokens(text: str) -> int:
    # Rough: 1 token ≈ 4 chars for English
    return len(text) // 4

def build(self, run_data, request):
    # ... existing build ...
    # While estimate_tokens(prompt) > 8000 and request.top_n_detail > 1:
    #     request.top_n_detail -= 1
    #     rebuild
```

- [ ] **Step 2: Improve error messages in JudgeService**

```python
except httpx.ConnectError:
    raise JudgeError("LM Studio not reachable. Start LM Studio or use API/Heuristic backend.")
except httpx.TimeoutException:
    raise JudgeError("Request timed out. Try Heuristic backend or increase timeout.")
```

- [ ] **Step 3: Add help tooltips in UI**

```javascript
// Add title attributes to modal fields explaining each option
```

- [ ] **Step 4: Run all judge tests**

Run: `pytest tests/test_judge*.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/judge_prompt.py lm_optimizer/services/judge_service.py \
      lm_optimizer/ui/static/js/results.js tests/test_judge_polish.py
git commit -m "feat: judge polish — token estimation, errors, tooltips"
```

---

### Task 9: Granular Test Selection — Core Benchmark Filtering

**Files:**
- Modify: `lm_optimizer/benchmark/suite.py` (add `include_categories` to `create_benchmark_cases`)
- Modify: `lm_optimizer/services/benchmark.py` (pass filter through)
- Modify: `lm_optimizer/cli/main.py` (add `--test`, `--test-only`, `--speed-only`, `--quality-only`, `--context`, `--mtp-only`/`--mtp-test` alias, `--reasoning-only`/`--reasoning-sweep` alias to `optimize`)
- Test: `tests/test_benchmark_filter.py`

**Interfaces:**
- Consumes: `BenchmarkPrompt.category` values
- Produces: Filtered benchmark cases per request

- [ ] **Step 1: Write failing tests**

```python
def test_create_benchmark_cases_filters_by_category():
    from lm_optimizer.benchmark.suite import create_benchmark_cases
    cases = create_benchmark_cases(4096, include_categories={"coding"})
    assert len(cases) == 1
    assert cases[0].name == "coding_task"

def test_create_benchmark_cases_multiple_categories():
    cases = create_benchmark_cases(4096, include_categories={"coding", "format"})
    assert len(cases) == 2
    names = {c.name for c in cases}
    assert names == {"coding_task", "structured_output"}

def test_create_benchmark_cases_empty_category_returns_all():
    cases = create_benchmark_cases(4096, include_categories=set())
    assert len(cases) == 5  # all tests

def test_reasoning_sweep_creates_5_configs_per_base():
    # When reasoning_sweep=True, returns 5 variants per base config
    # with generation.reasoning = off, low, medium, high, xhigh
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_filter.py -v`
Expected: FAIL

- [ ] **Step 3: Implement filtering in `lm_optimizer/benchmark/suite.py`**

```python
def create_benchmark_cases(context_length: int, seed: int = 42, 
                           include_categories: set[str] | None = None,
                           reasoning_sweep: bool = False) -> list[BenchmarkCase]:
    # If include_categories provided, filter BENCHMARK_SUITE by category
    # If reasoning_sweep: for each base case, create 5 variants with different reasoning settings
    pass
```

- [ ] **Step 4: Add CLI args in `lm_optimizer/cli/main.py`**

```python
# In optimize command (canonical *-only names, 2026-10-20):
test: list[str] = typer.Option(None, "--test", help="Comma-separated test names to run (coding_task,structured_output,...)")
test_only: bool = typer.Option(False, "--test-only", help="Run only specified tests, skip speed discovery")
speed_only: bool = typer.Option(False, "--speed-only", help="Run only speed probes at a pinned context")
quality_only: bool = typer.Option(False, "--quality-only", help="Quality battery on the baseline config only")
context: int = typer.Option(None, "--context", help="Pinned probe context for --speed-only/--mtp-only")
mtp_test: bool = typer.Option(False, "--mtp-only", "--mtp-test", help="Only the speculative/MTP probes (--mtp-test alias)")
reasoning_sweep: bool = typer.Option(False, "--reasoning-only", "--reasoning-sweep", help="Only the reasoning tests at off/low/medium/high/xhigh")
```

> 2026-10-20 wiring fix: Step 4's args existed but the `optimize` body never
> passed them to `_preflight_optimize` nor into the `advanced` dict, so every
> granular run silently executed a full campaign. Now the call site passes
> the flags, `_preflight_optimize` composes each mode's `--skip-phases` set
> (validated, printed in the confirm summary, persisted via
> `run.advanced_settings`), the `resume` command rebuilds
> `BenchmarkService` with the persisted flags, `_run_phase_ab` honors an
> explicit `speed_context`, quality-only/reasoning-only runs get the baseline
> config as their quality candidate, and `_phase_speed` filters the plan to
> S0 + speculative probes for MTP-only. Regression tests:
> `tests/test_granular_modes.py`.

- [ ] **Step 5: Wire through to benchmark service**

```python
# In benchmark.py _run_single_case or optimize.py:
# Parse test list → include_categories set
# Pass to create_benchmark_cases
# For reasoning_sweep: generate 5 config variants per base runtime
# For mtp_test: enable speculative decoding in LoadConfiguration
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_benchmark_filter.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add lm_optimizer/benchmark/suite.py lm_optimizer/services/benchmark.py \
      lm_optimizer/cli/main.py tests/test_benchmark_filter.py
git commit -m "feat: add granular test selection (CLI)"
```

---

### Task 10: Granular Test Selection — UI

**Files:**
- Modify: `lm_optimizer/ui/static/js/optimize.js` (or optimize page template)
- Modify: `lm_optimizer/ui/static/js/results.js` (add re-run specific test)
- Test: `tests/test_granular_ui.py`

**Interfaces:**
- Consumes: `/api/models`, new CLI-style options via API
- Produces: Test selection modal on Optimize page, re-run buttons on Results page

- [ ] **Step 1: Write failing test (Playwright)**

```python
def test_optimize_page_has_test_selection_modal(page):
    page.goto("http://localhost:8080/optimize")
    page.click("button:has-text('Test Selection')")
    assert page.locator("input[value='coding_task']").is_visible()
    assert page.locator("input[value='structured_output']").is_visible()
    assert page.locator("input[id='reasoning-sweep']").is_visible()
    assert page.locator("input[id='mtp-test']").is_visible()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_granular_ui.py -v`
Expected: FAIL

- [ ] **Step 3: Implement UI in `lm_optimizer/ui/static/js/optimize.js`**

```javascript
// 1. "Test Selection" button on Optimize page → opens modal
// 2. Modal tabs: "Quality Tests" (checkboxes per category), "Speed Test" (context input), "Advanced" (reasoning sweep, MTP test)
// 3. Store selection in optimize request payload
// 4. Results page: per-config "Re-run Test" dropdown with quick actions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_granular_ui.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/ui/static/js/optimize.js lm_optimizer/ui/static/js/results.js \
      tests/test_granular_ui.py
git commit -m "feat: add granular test selection UI"
```

---

### Task 11: Integration — Granular Tests + Judge

**Files:**
- Test: `tests/test_granular_judge_integration.py`

**Interfaces:**
- Consumes: Tasks 1-10
- Produces: Verification that judge can evaluate runs with partial test suites

- [ ] **Step 1: Write failing test**

```python
@pytest.mark.asyncio
async def test_judge_on_partial_test_suite(tmp_path):
    # 1. Create run with only coding_task + structured_output tests
    # 2. Run judge with use_case=coding
    # 3. Verify verdict only references available tests
    # 4. Verify reasoning matrix empty (no reasoning sweep run)
    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_granular_judge_integration.py -v`
Expected: FAIL

- [ ] **Step 3: Fix any integration issues**

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_granular_judge_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_granular_judge_integration.py
git commit -m "test: granular tests + judge integration"
```