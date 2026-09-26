# P1 Truth Audit (repository-evidence only)

Date: 2026-09-22 (updated P1.5: 136 tests, real concurrent throughput).

## Test inventory (verified: `pytest --collect-only -q`)

| File | Count |
|---|---|
| `lm_optimizer/tests/test_core.py` | 15 |
| `lm_optimizer/tests/test_optimization_correctness.py` | 46 |
| `lm_optimizer/tests/test_selection.py` | 17 |
| `lm_optimizer/tests/test_preheat.py` | 11 |
| `lm_optimizer/tests/test_workload.py` | 19 |
| `lm_optimizer/tests/test_p1_ux.py` | 33 |
| `lm_optimizer/tests/test_failure_states.py` | 17 |
| `lm_optimizer/tests/test_checkpoint_windows.py` | 5 |
| `lm_optimizer/tests/test_identity.py` | 4 |
| `lm_optimizer/tests/test_control_adapter.py` | 28 |
| `lm_optimizer/tests/test_dashboard_api.py` | 14 |
| `lm_optimizer/tests/test_db_isolation.py` | 3 |
| `lm_optimizer/tests/test_quick_connect.py` | 5 |
| `lm_optimizer/tests/test_reporting_transparency.py` | 16 |
| Total | 233 |

Full run `pytest -q`: 221 passed, 0 failed, 0 skipped
(2 pre-existing third-party deprecation warnings from starlette TestClient).

## Feature status (IMPLEMENTATION + CALL SITE + TEST + DOCS)

| Feature | Status | Implementation | Call site | Tests | Docs |
|---|---|---|---|---|---|
| 5% non-inferiority selection | IMPLEMENTED (this task) | `services/selection.py` (`SelectionConfig`, `select_best`, `select_by_score`) | `services/optimizer.py::_update_best` | `tests/test_selection.py` (14) | README services list |
| Pre-heat (LOAD/VERIFY/PREHEAT/MEASURE) | IMPLEMENTED (this task) | `services/preheat.py` (`run_preheat_phase`) | `services/benchmark.py::run_benchmark` (records `generation["preheat"]`) | `tests/test_preheat.py` (10) | README status line |
| Workload (INTERACTIVE/THROUGHPUT) | IMPLEMENTED (this task) | `services/workload.py` (`WorkloadType`, `apply_to_space`, `aggregate_throughput`, `workload_tiebreak_key`) | `services/search_space.py::_generate_parallels`, `services/optimizer.py::_update_best` band tie-break; `workload_type` in `run.benchmark_params` | `tests/test_workload.py` (10) | README CLI table (`--workload`) |
| Context sweep | IMPLEMENTED (prior) | `services/context_sweep.py` (geometric + bisect, step 500/1000, capacity vs performance-optimal vs balanced) | `cli/main.py::ctx`, `TestContextSweep` | 6 in `test_p1_ux.py` | README CLI table |
| Checkpointing | IMPLEMENTED (prior) | `storage/run_checkpoint.py` (atomic tmp+rename) | optimizer stages, API cancel/WS-cancel/shutdown | 5 in `test_p1_ux.py::TestCheckpoint` | `checkpoints` cmd |
| Resume | IMPLEMENTED (prior) | `services/optimizer.py::resume_from_checkpoint` (skips completed keys) | `cli/main.py::resume`, `api/routes.py::resume_run_from_checkpoint` | 2 in `test_p1_ux.py::TestResume` | README CLI table |
| WebSocket frontend | IMPLEMENTED (prior) | `api/websocket.py` + `ui/static/js/results.js::connectLive` (subscribed, reconnect) | results page live panel | 3 in `test_p1_ux.py::TestWebsocket` | README web UI |
| Live logging | IMPLEMENTED (prior) | `optimizer._log_event` + WS `log` messages | stage/test/config paths | covered in WS tests | — |
| Zero/partial success | IMPLEMENTED (prior) | `services/run_summary.py` (`classify_run`, `best_passed`) | CLI `_display_optimization_result`, `results.js::renderHeader` | 8 in `test_p1_ux.py::TestFinalStates` | — |
| ctx CLI | IMPLEMENTED (prior) | `cli/main.py::ctx` | `context_sweep.sweep` | `TestCli::test_ctx_*` + `TestContextSweep` | README CLI table |
| Micro-stage (<=6) | IMPLEMENTED (prior) | `services/optimizer.py::_stage_micro_refinement` | optimize pipeline after batch stage | 2 micro tests | README (5-stage + micro) |
| Final recommendation | IMPLEMENTED (prior) | `run_summary.alternatives/why_text` + CLI FINAL RECOMMENDATION + UI crowns | `_display_optimization_result`, `renderAlternatives` | `test_recommendation_crowns` | — |

NOT IMPLEMENTED items found: none remaining after this task. Previously
`selection.py`, preheat module and workload module did not exist (verified by
glob/grep: no `selection*.py`, `preheat`, `workload` files or symbols outside
JS variable names); prior reports claiming them were not backed by the repo.

## Changes made (this task)

- Added `services/selection.py`, `services/preheat.py`, `services/workload.py`.
- Wired selection into `_update_best` (band + workload tie-break), preheat into
  `BenchmarkService.run_benchmark`, workload into search space + optimizer +
  CLI (`optimize`/`auto --workload`) + API (`AdvancedSettingsSchema.workload_type`,
  `OptimizationRequest.workload_type`, merged advanced in `start_optimization`).
- Added `tests/test_selection.py` (14), `tests/test_preheat.py` (10),
  `tests/test_workload.py` (10). No existing test deleted, renamed, weakened or skipped.
- Fixed 3 Pydantic warnings locally: `schemas.py` `@validator` -> `@field_validator`,
  `websocket.py` 2x `.dict()` -> `.model_dump()`. No global suppression.
- README truth fixes: 61 -> 125 tests, 13 -> 16 commands, added ctx/resume/checkpoints
  rows and `--workload`, listed new services. No `133 tests` claim exists in the repo.

## Warnings / validation (measured 2026-09-22)

- `pytest -q`: 125 passed, 0 failed, 0 skipped. `pytest --collect-only -q`: 125.
- Pydantic warnings: 0 (was 3; fixed locally, verified with `-W error::DeprecationWarning`).
- `ruff check lm_optimizer`: see `docs/P1_5_STABILIZATION_AUDIT.md` for the exact
  reconciliation (baseline 531 -> current 866; legacy vs new split per file/rule).
  No ruff baseline file exists in the repo. Mechanical safe fixes applied
  (unused imports, mypy type args); remaining new violations match the
  surrounding code style (existing tests have the same violations).
- `ruff format --check .`: 26 files would be reformatted repo-wide (pre-existing;
  formatting not enforced). The 6 new files were formatted with `ruff format`.
- `mypy` (strict): the 3 new service files are clean; the only errors in scope
  are 2 pre-existing ones in untouched `logging_config.py`. Repo-wide mypy was
  already non-clean at baseline.
- `python -m compileall lm_optimizer`: OK.
- `git status` / `git diff`: not a git repository (no `.git`); nothing to commit.

## Known limitations

- `RunStatus.COMPLETED` kept as legacy alias of SUCCESS for old DB rows/UIs.
- `workload_type` persisted in `run.benchmark_params` (no DB migration) by design.
- Preheat timings persist via `result.generation["preheat"]` (config_json channel).
- Routes still use `.dict()` on request models (runtime paths, untested); only the
  3 test-visible warnings were fixed per scope.
