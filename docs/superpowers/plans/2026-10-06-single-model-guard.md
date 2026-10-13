# Single-Model Guard + Job Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The endpoint never holds two models at once and two optimizer jobs never run concurrently — foreign models trigger a wait-loop then pause+resume, second jobs are refused with a message.

**Architecture:** Central `ensure_exclusive_access()` in `hostguard.py` called from `prepare_host` (all CLI commands get it automatically); cross-process lock file with PID liveness in new `services/joblock.py`, acquired at CLI/web run start and released in `finally`.

**Tech Stack:** Python, pytest, typer CLI, psutil (already a dependency).

**Spec:** Conversation 2026-10-06 (decisions: wait-loop 60s × max 30 then pause+resume; shared CLI+web lock with stale takeover; lock at `results/.optimizer.lock`).

## Global Constraints

- TDD: failing test first for every task.
- Full pytest green before every commit.
- Never commit `results/` or `*.db` (lock file lives under gitignored `results/` by design, created at runtime only).
- Fail-closed: any doubt about endpoint state aborts the load, never measures blind.
- Default paths byte-identical when no contention exists.
- Do not touch unrelated dirty working-tree files; commit only task files.
- ASCII-only in reports/docs.

## Review Focus

- Foreign model unload succeeds halfway then the app reloads it mid-sweep; expected behavior is the pre-load re-check catches it every time, not just at startup.
- Lock holder process killed -9 (no finally ran); expected behavior is stale takeover after PID-dead check, never a permanent block.
- Two jobs starting in the same second; expected behavior is exactly one wins (atomic create), never both.
- Wait-loop with a model that unloads at 29/30 retries; expected behavior is immediate proceed, not a full sleep-through.
- Web server running two workers sharing one lock file; expected behavior is documented single-worker requirement or worker-safe locking, not silent double-run.

---

### Task 1: ensure_exclusive_access + wait loop

**Files:**
- Modify: `lm_optimizer/services/hostguard.py`
- Test: `lm_optimizer/tests/test_hostguard.py` (extend or create)

**Interfaces:**
- Consumes: `client.ensure_unloaded`, instance listing used by `prepare_host`.
- Produces: `ensure_exclusive_access(client, purpose, wait_interval_s=60.0, max_waits=30) -> dict` returning `{free: True, waited_s, attempts}`; raises `HostBusyTimeout(blocker=...)` after exhausting waits; `class HostBusyTimeout(Exception)` with `.blocker` attribute.

- [ ] **Step 1: Write the failing test**

```python
async def test_exclusive_access_waits_then_proceeds(mock_client_foreign_then_gone):
    out = await ensure_exclusive_access(mock_client_foreign_then_gone, "test", wait_interval_s=0, max_waits=3)
    assert out["free"] is True and out["attempts"] >= 1

async def test_exclusive_access_raises_after_retries(mock_client_foreign_forever):
    with pytest.raises(HostBusyTimeout):
        await ensure_exclusive_access(mock_client_foreign_forever, "test", wait_interval_s=0, max_waits=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_hostguard.py -v -k exclusive`
Expected: FAIL with "function not defined".

- [ ] **Step 3: Implement in `lm_optimizer/services/hostguard.py`**

List loaded instances; none → return immediately; try ensure_unloaded; on persistent foreign model sleep wait_interval_s and re-check (no full sleep-through: re-check right after a successful unload); raise HostBusyTimeout with blocker model ids after max_waits.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_hostguard.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/hostguard.py lm_optimizer/tests/test_hostguard.py
git commit -m "feat: add exclusive-access guard with wait loop"
```

### Task 2: Cross-process job lock

**Files:**
- Create: `lm_optimizer/services/joblock.py`
- Test: `lm_optimizer/tests/test_joblock.py` (create)

**Interfaces:**
- Consumes: nothing (standalone; psutil for PID liveness).
- Produces: `acquire_lock(purpose) -> dict` (raises `JobBusyError(holder=...)` when a live holder exists; takes over stale locks); `release_lock() -> None` (idempotent); `LOCK_PATH = results/.optimizer.lock` resolved from configured results dir; atomic create via O_CREAT|O_EXCL.

- [ ] **Step 1: Write the failing test**

```python
def test_second_acquire_refuses(tmp_path):
    acquire_lock("job-a", lock_dir=tmp_path)
    with pytest.raises(JobBusyError):
        acquire_lock("job-b", lock_dir=tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_joblock.py -v`
Expected: FAIL with "function not defined".

- [ ] **Step 3: Implement lock with PID-liveness stale takeover**

Lock content `{pid, started, purpose}` as JSON; holder PID dead → take over + log; release removes only own lock (PID match).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_joblock.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/joblock.py lm_optimizer/tests/test_joblock.py
git commit -m "feat: add cross-process optimizer job lock"
```

### Task 3: Wiring (prepare_host + CLI + web + pause)

**Files:**
- Modify: `lm_optimizer/services/hostguard.py` (`prepare_host` calls ensure_exclusive_access; refused-empty semantics unchanged)
- Modify: `lm_optimizer/cli/main.py` (acquire lock at run-start commands, release in finally; `--wait-interval/--max-waits` on optimize/auto/deep/context-sweep; HostBusyTimeout in optimize triggers pause flow with resume hint, elsewhere aborts with message)
- Modify: web run-start path (locate where web launches runs — acquire before start, release on finish; refuse with 409 + holder info)
- Test: extend `lm_optimizer/tests/test_hostguard.py` + `test_joblock.py`

**Interfaces:**
- Consumes: Task 1 + Task 2 outputs.
- Produces: every load path guarded; every run start locked; pause-on-busy in optimize.

- [ ] **Step 1: Write the failing test**

```python
def test_prepare_host_enforces_exclusivity(mock_client_foreign_forever):
    with pytest.raises(HostBusyTimeout):
        prepare_host_sync_wrapper(mock_client_foreign_forever, "test", wait_interval_s=0, max_waits=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_hostguard.py -v -k exclusivity`
Expected: FAIL.

- [ ] **Step 3: Implement wiring (guard in prepare_host, lock acquire/release around run execution, flags, pause conversion, web 409)**

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_hostguard.py lm_optimizer/tests/test_joblock.py -v`
Expected: PASS, plus full suite green.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/services/hostguard.py lm_optimizer/cli/main.py lm_optimizer/api/routes.py lm_optimizer/tests/test_hostguard.py lm_optimizer/tests/test_joblock.py
git commit -m "feat: wire exclusive access and job lock into all run paths"
```

### Task 4: Docs

**Files:**
- Modify: `docs/OPTIMIZATION_METHOD.md` (single-model invariant section: wait-loop defaults, pause+resume, lock behavior, flags)

**Interfaces:**
- Consumes: Tasks 1-3 as built.
- Produces: documented truth (defaults 60s × 30, lock path, 409 semantics, resume procedure).

- [ ] **Step 1: Write the docs section**

Content decisions only: exact flags, defaults, lock path, what the user sees in each contention case.

- [ ] **Step 2: Verify suite still green**

Run: `pytest lm_optimizer/tests -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add docs/OPTIMIZATION_METHOD.md
git commit -m "docs: single-model invariant and job lock"
```
