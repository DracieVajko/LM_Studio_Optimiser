# Pause/Resume Partial Success & Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement pause/resume functionality for partial successes/paused reports/optimizations so they can be resumed overnight, and create a release with proper tag, title, and description.

**Architecture:** Extend existing pause/resume infrastructure (checkpointing, PauseRequested exception, CLI pause/resume commands) to support:
1. **Web UI pause/resume** - Add pause/resume buttons in web UI for running optimizations
2. **Partial success resume** - Allow resuming from partial_success runs (best config found but optimization incomplete)
3. **Web UI pause/resume buttons** - Add pause/resume buttons in web UI for running optimizations
4. **Release** - Create v1.5.1 release with proper tag, title, and description

**Tech Stack:** Python, FastAPI, Typer CLI, orjson, pytest

**Spec:** User requirements from conversation 2026-10-07

## Global Constraints

- TDD: failing test first for every task
- Full pytest green before every commit
- Never commit `results/` or `*.db`
- ASCII-only in reports
- Never touch unrelated dirty working-tree files; commit only task files
- Web UI uses vanilla JS + FastAPI

## Review Focus

- Web UI pause/resume buttons work correctly
- Partial success runs can be resumed from where they left off
- Paused runs can be resumed from web UI
- Checkpoint data integrity on resume
- Release tag follows semantic versioning

---

### Task 1: Web UI Pause/Resume Buttons

**Files:**
- Modify: `lm_optimizer/ui/templates/results.html`
- Modify: `lm_optimizer/ui/static/js/results.js`
- Modify: `lm_optimizer/api/routes.py` (add pause/resume endpoints)
- Test: `lm_optimizer/tests/test_web_pause_resume.py` (create)

**Interfaces:**
- Consumes: Existing pause/resume CLI commands, checkpoint system
- Produces: Web UI pause/resume buttons, API endpoints `/api/runs/{run_id}/pause` and `/api/runs/{run_id}/resume`

- [ ] **Step 1: Write the failing test**

```python
def test_pause_resume_buttons_in_web_ui():
    # Test that pause/resume buttons appear in results page
    # Test pause API returns 200 and creates flag file
    # Test resume API returns 200 and resumes optimization
    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_web_pause_resume.py -v`
Expected: FAIL with "function not defined" or 404 on endpoints

- [ ] **Step 3: Implement pause/resume API endpoints in `lm_optimizer/api/routes.py`**

Add POST `/api/runs/{run_id}/pause` and POST `/api/runs/{run_id}/resume` endpoints that call the existing pause/resume logic.

- [ ] **Step 4: Add pause/resume buttons to `results.html` template**

Add pause/resume buttons in the results page header, conditionally shown based on run status.

- [ ] **Step 5: Implement JavaScript handlers in `results.js`**

Add click handlers for pause/resume buttons that call the API endpoints and update UI state.

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_web_pause_resume.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add lm_optimizer/api/routes.py lm_optimizer/ui/templates/results.html lm_optimizer/ui/static/js/results.js lm_optimizer/tests/test_web_pause_resume.py
git commit -m "feat: add web UI pause/resume buttons for running optimizations"
```

### Task 2: Partial Success Resume Support

**Files:**
- Modify: `lm_optimizer/services/optimizer.py` (handle partial_success resume)
- Modify: `lm_optimizer/cli/main.py` (resume command for partial_success)
- Test: `lm_optimizer/tests/test_partial_success_resume.py` (create)

**Interfaces:**
- Consumes: Existing `resume_from_checkpoint` logic, checkpoint system
- Produces: Ability to resume from `partial_success` runs (runs with best config found but incomplete)

- [ ] **Step 1: Write the failing test**

```python
def test_resume_partial_success_run():
    # Create a run with status=PARTIAL_SUCCESS and best_config_id
    # Call resume_from_checkpoint
    # Verify it resumes from where it left off (continues from best config)
    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_partial_success_resume.py -v`
Expected: FAIL - partial_success runs not handled in resume logic

- [ ] **Step 3: Implement partial success resume in `optimizer.py`**

Modify `resume_from_checkpoint` to handle `PARTIAL_SUCCESS` status - should resume from best config if exists, or continue from where it left off.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_partial_success_resume.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/optimizer.py lm_optimizer/tests/test_partial_success_resume.py
git commit -m "feat: support resuming from partial_success runs"
```

### Task 3: Web UI "Run All Pending" Button

**Files:**
- Modify: `lm_optimizer/ui/templates/history.html` (add "Resume All Pending" button)
- Modify: `lm_optimizer/ui/static/js/history.js` (add handler)
- Test: `lm_optimizer/tests/test_web_history_resume.py` (create)

**Interfaces:**
- Consumes: Existing resume API, history page
- Produces: "Resume All Pending" button in history page that resumes all paused/partial_success runs

- [ ] **Step 1: Write the failing test**

```python
def test_resume_all_pending_button():
    # Create multiple paused/partial_success runs
    # Click "Resume All" button
    # Verify all are queued for resume
    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_web_history_resume.py -v`
Expected: FAIL

- [ ] **Step 3: Implement "Resume All Pending" button in history UI**

Add button in history page that finds all paused/partial_success runs and resumes them sequentially.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_web_history_resume.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/ui/templates/history.html lm_optimizer/ui/static/js/history.js lm_optimizer/tests/test_web_history_resume.py
git commit -m "feat: add 'Resume All Pending' button in history UI"
```

### Task 4: Release v1.5.1

**Files:**
- Create: Release tag v1.5.1
- Modify: `CHANGELOG.md` (add v1.5.1 entry)
- Create: GitHub Release with proper title and description

**Interfaces:**
- Consumes: All implemented features
- Produces: GitHub release v1.5.1 with proper changelog

- [ ] **Step 1: Update CHANGELOG.md with v1.5.1 entry**

Add changelog entry for v1.5.1 with all new features.

- [ ] **Step 2: Create git tag v1.5.1**

```bash
git tag v1.5.1
git push origin v1.5.1
```

- [ ] **Step 3: Create GitHub Release**

Use GitHub CLI or web UI to create release with proper title and description.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md
git commit -m "chore: prepare v1.5.1 release"
git tag v1.5.1
git push origin v1.5.1
```

### Task 5: Documentation Updates

**Files:**
- Modify: `docs/OPTIMIZATION_METHOD.md` (add pause/resume section)
- Modify: `README.md` (add pause/resume usage)

**Interfaces:**
- Consumes: Implemented features
- Produces: Updated documentation

- [ ] **Step 1: Add pause/resume documentation to OPTIMIZATION_METHOD.md**

- [ ] **Step 2: Update README.md with pause/resume usage**

- [ ] **Step 3: Commit**

```bash
git add docs/OPTIMIZATION_METHOD.md README.md
git commit -m "docs: add pause/resume documentation"
```

---

## Execution Approach

**Plan complete and saved to `docs/superpowers/plans/2026-10-07-pause-resume-release.md`. Please review the plan. Which execution approach would you prefer?**

- **Subagent-driven** - A fresh subagent implements each task and a fresh reviewer checks it before the next one starts, then a whole-branch review at the end. Most thorough; costs a fresh context per task and per review.
- **Native** - I implement every task myself in this session, the way this harness runs work, then one fresh reviewer on the most capable model checks the whole branch. Cheapest and fastest; no independent review until the end. Runs well with a mid-tier session model, since the plan carries the design.

**For this plan I recommend Subagent-driven**, because the tasks involve web UI changes (which need browser testing), API changes, and database migrations - each task can be independently verified. Does the plan capture what you want, and which approach should we use?