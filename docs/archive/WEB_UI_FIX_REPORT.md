# Web UI + Startup Launcher Fix Report (2026-09-23)

Backend verified healthy first: `GET /`, `/api/status`, `/api/models`,
`/api/runs`, `/history`, `/settings` all HTTP 200 with correct JSON/HTML.
The faults were frontend resilience + test hygiene + launcher UX.

## Root causes

1. **Blank Dashboard**: `App.init()` calls `loadDashboard()` directly with no
   `try/catch`, and `loadDashboard()` awaits `API.getStatus()` which has no
   timeout and rejects on connection failure. Any failure (server starting,
   refused connection, slow response) kills init silently -> empty
   `#main-content`, header badge stuck on static "Disconnected". History and
   Settings survived because they load through `loadView()` (has catch) or
   simpler paths.
2. **View spinner forever**: `results.js loadRun()` swallowed errors (toast
   only) leaving `run=null`; `render()` then dereferenced `run.model` and
   threw, stranding the static spinner. No request timeout existed anywhere.
3. **Disappearing BAT**: old launcher `start`ed a detached server (errors
   invisible), assumed process-start == ready, and opened the browser
   unconditionally.
4. **Test pollution**: every pytest run wrote into production
   `data/optimizer.db` (191 runs, incl. `resume-model-x`/`m` pending/running
   rows). Confirmed by row counts before/after test runs.

## Files changed

- `ui/static/js/api.js`: 12 s `AbortController` timeout on all requests +
  export; network/timeout errors carry messages; HTTP errors carry `status`.
- `ui/static/js/app.js`: `init()` and `loadDashboard()` wrapped with visible
  error cards + Retry; header badge updates immediately after load.
- `ui/static/js/ui.js`: `renderError(message, retry)`; new
  `renderRunError(runId, status, reason)`; disconnected banner now has
  DISCONNECTED header, [Open Settings] link and last-error line.
- `ui/static/js/results.js`: null-run guard (error card, never spinner),
  Chart.js absence guard, charts.js CDN-init guard.
- `ui/static/js/charts.js`: CDN failure can no longer break importing pages.
- `tests/conftest.py`: autouse tmp-DB + tmp results_dir isolation.
- `tests/test_db_isolation.py` (3, new): tmp-DB routing proof, production
  untouched proof (rows + checkpoint files).
- `database/repositories.py`: `RunRepository.delete` (cascade),
  `delete_orphan_model` (only when unreferenced).
- `cli/main.py`: `cleanup-test-runs` (dry-run default, `--confirm` deletes
  test-model runs + configs + checkpoints + orphan models; stale genuine
  rows listed, never deleted) + `best.score is not None` guard in `runs`.
- `start_optimizer.bat` (rewritten): [1/4] Python, [2/4] venv+deps,
  [3/4] server start with output to `data/logs/startup.log`, [4/4] curl poll
  of `/api/status` (30 s), browser only on HTTP 200, FAILED panel with log
  tail + pause on any failure. Server console stays visible; closing it
  stops the server.
- `tests/test_dashboard_api.py`: invalid/missing run + missing model return
  JSON (never HTML), launcher contract test, same-services test.
- `routes.py`, `websocket.py`, `optimizer.py` progress counters now count all
  terminal failure classes (not only legacy FAILED); `results.js`/`ui.js`
  badge maps cover the new statuses.
- `docs/LM_STUDIO_PARAMETER_MATRIX.md`: regenerated (102 parameters).

## Database cleanup (executed, strict fingerprint)

`cleanup-test-runs --confirm`: removed **171** test-fixture runs
(`m`/`resume-model-x`), their configurations (cascade), orphan test models.
Own E2E snippet leftover (`db18cef3`, 1 genuine load_failed config) kept as
a real result. Before 191 runs -> after 20. Remaining stale genuine rows
(2 old `running`, listed not deleted).

## Smoke-test results

- `pytest --collect-only -q`: 233 collected.
- `pytest -q`: 233 passed (2 pre-existing third-party warnings).
- Live: `/` 200, `/api/status` 200 (`connected: True`), `/api/runs` 200.
- Playwright: not installed -> browser assertions not automated; frontend
  failure paths verified by code-level contract (timeout, error cards,
  null-guards) + backend TestClient coverage.
- `python -m compileall lm_optimizer`: OK.
- `ruff check lm_optimizer`: 1157 (531 legacy + 521 new files + 105 edited
  net; exact per-file reconciliation vs the saved 531 transcript; new service
  files `failure_states.py`/`control_adapter.py` clean at 0).
  `ruff format --check`: only legacy/untouched files remain.
- `mypy lm_optimizer`: blocked repo-wide by missing third-party stubs
  (pre-existing); new service files clean.

## Round 2 — live user feedback (same session)

User reported: BAT fails with `... was unexpected at this time`; dashboard
still blank in a real browser (only nav + empty main; server log showed just
`/` + one static file).

1. **BAT root cause**: `echo ... (editable + dev)...` inside a parenthesized
   `if/else` block — the `)` closed the block early (classic batch pitfall).
   Fix: whole launcher rewritten to linear goto flow with zero
   parens-in-blocks; server starts via `cmd /c` idiom so `startup.log`
   capture actually applies to the child.
2. **Blank-page hardening**: every page now ships an inline (non-module)
   boot reporter that converts any module/import/CDN failure into a visible
   error card with Retry (previously: eternal blank/spinner). All local
   scripts cache-busted (`?v=2`); live-verified all 8 JS assets return 200.
3. The dashboard shell the user pasted matches the pre-fix template
   (no spinner fallback, no reporter, unversioned scripts) — the fixes above
   directly address what they saw.

## Round 4 — user feedback: unwanted loads, truncated IDs, no number pick

1. **Every `connect()` loaded qwen3.8** (echo probe of smallest model) — even
   for `status`/`models`. Fix: `connect(echo_probe=False)` skips the echo
   phase (safe 400-probing still runs); `status --quick` and `models`
   (always quick) never load anything. Verified live: only safe 404 probes.
2. **`models --full-ids`**: numbered `N|full-id` plain list (screenshot
   truncation made IDs uncopyable). BAT [2] prints it and accepts number OR
   pasted ID, shows resolved model + profile, and asks Start? Y/n before
   any optimization (no more accidental 5040-candidate runs from typing "1").
   BAT [4] shows table + full-ID list.
3. Menu [3] now runs `status --quick` (connection proof only).

## Round 3 — BAT menu + visible logs + CDN check (same session)

User feedback: BAT died with `... was unexpected at this time`; server
window black/empty; dashboard look still off; want model choice in launcher.

1. **BAT syntax**: `echo ... (editable + dev)...` inside a parenthesized
   `if/else` block closed the block early (classic batch pitfall, matched the
   pasted failure point exactly). Whole file rewritten to linear goto flow
   with zero parens-in-blocks.
2. **Visible server logs**: redirect-to-log hid the console; server now runs
   in its own visible window with live logs, stages/poll still logged.
3. **Menu restored**: [1] Web UI + browser, [2] optimize one model
   (foreground, Ctrl+C native), [3] check, [4] models, [5] exit. Return to
   menu stops the server via window title (no orphans).
4. **CDN check from this machine**: tailwind 302, chart.js 200 — reachable,
   so unstyled look is not a blocked CDN here. Every page now also carries a
   boot reporter, so any residual browser-side failure names itself on-page.

## Round 5 — unwanted loads, truncated IDs, runtime noise (same session)

1. **Every `connect()` loaded qwen3.8** (echo probe). New
   `connect(echo_probe=False)`; `status --quick` + `models` (always quick)
   never load — verified live (only safe 404 key-probes).
2. **`models --full-ids`**: numbered `N|full-id` plain list; BAT [2] prints
   it, accepts number or pasted ID, shows resolved model + profile, asks
   Start? Y/n. BAT [4] shows table + full IDs.
3. **`lms` UnicodeDecodeError spam** (cp1252 vs spinner bytes): `run_lms`
   now uses explicit UTF-8 + replace.
4. **Rejected reasoning values remembered per (model, value)**: gpt-oss-20b
   did 2 requests per chat for the whole run; now the doomed first attempt
   is skipped after the first refusal.
5. New tests: `test_quick_connect.py` (5). Total 212.

## Round 6 — live 9b run review (same session)

User ran a real qwen3.8-9b-distill optimization (~5 h). Findings:

1. **`UnicodeDecodeError` spam is stale code, not a new bug**: the running
   process predates the `run_lms` UTF-8 fix; all remaining `text=True`
   call sites (`hardware/detection.py`, `services/hardware.py` x4,
   `hostguard.py` x3) now carry explicit UTF-8 + replace.
2. **Failure machine verified live**: `QUALITY_FAILED` 28/30 < 0.97 ->
   `CONFIG_REJECTED` with metrics preserved and no score; `CONFIG_ELIGIBLE`
   only for scored winners; 5% band + workload tie-break events present.
3. **Progress bar frozen at 0% for hours**: new optional
   `progress_cb(stage, tested, total)` on `optimize()` (default None, no
   behavior change) wired to the CLI rich bar (`[stage tested/total]`);
   fired at stage boundaries and per candidate via `_report_progress()`.
4. Full `connect()` (with echo probe) still loads qwen3.8 once per
   `optimize` run — legitimate capability verification, documented; only
   `status`/`models` went load-free.

## Round 6 — boot root cause from the real trace (same session)

Browser trace proved the failure class: all JS modules fetched (200/304),
**zero `/api/*` requests followed** — the app module never began executing.
Killed every silent-boot path:

- `readyState`-safe boot in all 4 entry files (a deferred module that
  evaluates after `DOMContentLoaded` previously never ran `init()` at all —
  no error, no requests, eternal spinner).
- Whole module graph cache-busted to `?v=3` (entries *and* their relative
  imports — previously only entries were versioned, so browsers could mix
  fixed entries with stale `api.js` lacking timeouts/guards).
- 25 s watchdog in every template: silent hangs become visible cards.
- Consistency test pins a single static version across templates + imports.
- Live-verified `?v=3` assets serve 200.

## Not done (per STOP instruction)

GPT-OSS optimization not started. Noted for next step: 8k context cap,
16 GB RAM constraint, speed + accuracy first (user directive on file).
