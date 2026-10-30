# Test Manager + Recommender Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate a test manager + recommender into the LM Studio optimizer (`lab` CLI and web UI) that intelligently orders tests by model size/VRAM, pauses on slow speeds, allows assistant/local-only opt-in, cleans failed artifacts, and recommends best model/config — all within this repo and never leaking outside it.

**Architecture:** New `lm_optimizer/services/test_manager.py` service; new CLI commands (`lab test-manager`, `lab recommend`, `lab clean`) added to existing `lm_optimizer/cli/lab.py`; UI prompt/card added to existing `results.js`/`model-detail.js`; DB uses existing schema extended with `status='partial'` for intermediate results. No new repo; no external folder access.

**Tech Stack:** Python 3.11+, existing FastAPI/Typer/rich, `urllib.request` for LM Studio ping (no new HTTP client needed for ping), vanilla JS.

**Spec:** `docs/specs/2026-10-28-test-manager-design.md`

## Global Constraints

- Integration inside this repo only; no new repository or script outside `lm_optimizer/`.
- Never touches anything outside this folder; no leaks to system directories.
- Assistant endpoint (optional) uses `.env` (`LM_OPTIMIZER_*`) only; key never written to code/repo; no web access by default.
- No long local LLM runs; verify only mocked (`--verify` uses mock client); live `load_model` deferred until user returns.
- Keep all 61+ existing persistence loop tests green; add new tests rather than weakening old ones.
- Default ordering: ascending size → MoE before dense → dense last; VRAM floor 250–500 MB free; speed floor 5 tok/s (configurable); partial results saved; clean after 2 consecutive failures.

## Review Focus

1. Assistant API key must never appear in any file, test output, or git object in this repo.
2. `test-manager` must not call `load_model` unless explicitly enabled by user (`--use-assistant`); default `local-only`.
3. Cleanup must only affect files inside the working directory / LM Studio defaults directory, never system paths.
4. Partial/intermediate results (`status='partial'`) must not corrupt existing `success`/`failed` state tracking.
5. Web UI prompt/card must not break when `results.js` loads on runs without the feature enabled.

---

### Task 1: Test manager service skeleton and ordering logic

**Files:**
- Create: `lm_optimizer/services/test_manager.py`
- Modify: `lm_optimizer/services/lmstudio_defaults.py` (optional reuse of `resolve_defaults_path` if needed)
- Test: `tests/test_test_manager.py`

**Interfaces:**
- Produces: `class TestManager`, `order_models(models: list[ModelInfo]) -> list[str]`, `should_use_short_tests(config) -> bool`, `estimate_vram_needed(model_id) -> int`

- [ ] **Step 1:** Write failing tests for ordering (ascending size, MoE before dense) and VRAM floor guard.
- [ ] **Step 2:** Verify tests fail.
- [ ] **Step 3:** Implement `test_manager.py` with ordering and VRAM guard.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5:** Commit.

---

### Task 2: CLI commands (`test-manager`, `recommend`, `clean`)

**Files:**
- Modify: `lm_optimizer/cli/lab.py`
- Test: `tests/test_persistence_loop_cli.py` (extend with new commands)

**Interfaces:**
- Consumes: `TestManager`, `run_repo`, existing `_client()` / `get_lm_client()`
- Produces: CLI commands with `--interactive`, `--use-assistant`, `--speed-floor`, `--dry-run`

- [ ] **Step 1:** Add failing CLI tests for `test-manager` (interactive, dry-run), `recommend` (shows best config), `clean` (removes failed artifacts).
- [ ] **Step 2:** Verify failure.
- [ ] **Step 3:** Implement commands in `lab.py`.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5:** Commit.

---

### Task 3: Assistant endpoint integration (opt-in, no leak)

**Files:**
- Modify: `lm_optimizer/config.py` (add optional `assistant_url`, `assistant_key` via `.env` only)
- Modify: `lm_optimizer/services/test_manager.py` (use assistant if `--use-assistant` and `.env` set)
- Test: `tests/test_persistence_loop_api.py` (mock assistant endpoint, verify no real calls by default)

**Interfaces:**
- Consumes: `config.assistant_url`, `config.assistant_key` (only if user explicitly sets `.env`)
- Produces: `use_assistant()` guard; default `local-only`; no web access unless configured.

- [ ] **Step 1:** Add test asserting no assistant call when `.env` missing / `--use-assistant` not given.
- [ ] **Step 2:** Verify failure (or pass if guard already exists from skeleton).
- [ ] **Step 3:** Implement `.env` read (only `LM_OPTIMIZER_ASSISTANT_URL` / `LM_OPTIMIZER_ASSISTANT_KEY`) and guard.
- [ ] **Step 4:** Tests pass (no leaks, no unexpected external calls).
- [ ] **Step 5:** Commit.

---

### Task 4: Partial results and DB state tracking

**Files:**
- Modify: `lm_optimizer/domain/models.py` (add `ConfigurationStatus.PARTIAL` if not present; extend `RunStatus` if needed)
- Modify: `lm_optimizer/services/test_manager.py` (save partial results before pause)
- Test: `tests/test_persistence_loop_api.py` (verify partial results returned correctly)

**Interfaces:**
- Produces: `partial` status preserved; `best_config_id` not overwritten by partial; resume works.

- [ ] **Step 1:** Add failing test for partial save/resume.
- [ ] **Step 2:** Verify failure.
- [ ] **Step 3:** Implement partial tracking.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5:** Commit.

---

### Task 5: Cleanup logic and artifacts

**Files:**
- Create/modify: `lm_optimizer/services/test_manager.py` (cleanup method)
- Modify: `lm_optimizer/cli/lab.py` (`clean` command)
- Test: `tests/test_persistence_loop_cli.py`

**Interfaces:**
- Produces: `clean_failed()` removes artifacts and DB entries after 2 consecutive failures; configurable threshold.

- [ ] **Step 1:** Write failing cleanup test.
- [ ] **Step 2:** Verify failure.
- [ ] **Step 3:** Implement cleanup with configurable failure count.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5:** Commit.

---

### Task 6: UI prompt/card (`results.js`, `model-detail.js`, `model_detail.html`)

**Files:**
- Modify: `lm_optimizer/ui/static/js/results.js` (add interactive prompt before save/test start; opt-in checkbox for assistant)
- Modify: `lm_optimizer/ui/static/js/model-detail.js` (show `test-manager` status/card)
- Modify: `lm_optimizer/ui/templates/model_detail.html` (container for card)
- Test: `tests/test_webui_regressions.py` (static guard ensuring card present and prompt logic intact)

**Interfaces:**
- Produces: `test-manager` card visible; interactive prompt shows; no break when feature disabled.

- [ ] **Step 1:** Write failing static guard test (`md-test-manager` present, `getTestManagerStatus` or similar method referenced).
- [ ] **Step 2:** Verify failure.
- [ ] **Step 3:** Implement UI changes.
- [ ] **Step 4:** Tests pass; node --check clean.
- [ ] **Step 5:** Commit.

---

### Task 7: Final verification and docs

**Files:**
- Modify: `docs/test_manager.md` (usage, CLI, API references)
- Modify: `CHANGELOG.md` (add `[Unreleased]` entries)
- Run: full targeted suite (`tests/test_persistence_loop_*`, `tests/test_lmstudio_defaults.py`)

**Interfaces:**
- Consumes: all previous tasks.
- Produces: updated docs, changelog, green suite.

- [ ] **Step 1:** Update docs with CLI/API references and security rules.
- [ ] **Step 2:** Update changelog.
- [ ] **Step 3:** Run full targeted suite; verify 61+ pass.
- [ ] **Step 4:** Verify repo clean (`git status`), no secrets, no temporary artifacts.
- [ ] **Step 5:** Commit and push.

---

## Self-Review

1. **Spec coverage:** Each spec section (ordering, assistant opt-in, cleanup, partial results, isolation, no web) maps to a task.
2. **Step scan:** Every step is one action (test, run, implement, verify, commit). No "handle edge cases" placeholders.
3. **Type consistency:** `TestManager` used consistently; CLI commands match (`test-manager`, `recommend`, `clean`); DB status `partial` consistent.
4. **Review Focus:** All 5 uncovered inputs covered by tests (key isolation, default local-only, cleanup, partial DB, UI presence).
5. **Proportion:** Plan is shorter than spec; code bodies left to implementer except signatures pinned.
