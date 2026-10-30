# Task 1 Report: All-configs table in .md reports

## Summary
Task 1 (All-configs table in .md reports) is **already complete** in the current branch `feat/deep-research`. The implementation was committed in `763c1fe` (feat: add all-tried-configs table to best report).

## Changes Made (in commit 763c1fe)

### `lm_optimizer/services/reporting.py`
- Added `_all_configs_rows(configs: list, best_id) -> list[str]` function (lines 556-631)
- Wired into `save_best_report` after "All tried configurations" header (line 843)

### `lm_optimizer/tests/test_full_reports.py`
- Created new test file with 14 tests covering:
  - `test_all_configs_table_lists_every_config` - verifies score-descending for passed, failed at bottom
  - `test_missing_metrics_render_na` - verifies n/a for missing metrics
  - `test_pipe_escaping_and_error_trim` - verifies pipe-escaping via `_cell` and error trimmed to 120 chars
  - `test_best_report_contains_all_configs_table` - integration test with `save_best_report`
  - Additional tests for full outputs appendix (Tasks 2, 4)

## Test Evidence

```
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
collected 14 items

lm_optimizer/tests/test_full_reports.py::test_all_configs_table_lists_every_config PASSED
lm_optimizer/tests/test_full_reports.py::test_missing_metrics_render_na PASSED
lm_optimizer/tests/test_full_reports.py::test_pipe_escaping_and_error_trim PASSED
lm_optimizer/tests/test_full_reports.py::test_best_report_contains_all_configs_table PASSED
lm_optimizer/tests/test_full_reports.py::test_outputs_appendix_includes_thinking_and_output PASSED
lm_optimizer/tests/test_full_reports.py::test_empty_thinking_renders_na PASSED
lm_optimizer/tests/test_full_reports.py::test_no_measurements_config_renders_status_line PASSED
lm_optimizer/tests/test_full_reports.py::test_appendix_blocks_keep_pipes_verbatim_and_escape_inner_fences PASSED
lm_optimizer/tests/test_full_reports.py::test_best_report_contains_full_outputs_appendix PASSED
lm_optimizer/tests/test_full_reports.py::test_failed_report_contains_full_outputs_appendix PASSED
lm_optimizer/tests/test_full_reports.py::test_rereport_regenerates_from_stored_run PASSED
lm_optimizer/tests/test_full_reports.py::test_rereport_winnerless_run_writes_failed_report PASSED
lm_optimizer/tests/test_full_reports.py::test_rereport_corrupt_run_skips_without_abort PASSED
lm_optimizer/tests/test_full_reports.py::test_rereport_saver_error_skips_without_abort PASSED

============================= 14 passed in 2.59s ==============================
```

## Commit Hash
- `763c1fe4273c185a01c5a3d8150ff1df85cfe08c` - feat: add all-tried-configs table to best report
- Also includes subsequent commits `bb07ba5` (append full outputs) and `e91a752` (re-report backfill)

## Self-Review

### Requirements Met ✓
- ✅ `_all_configs_rows(configs, best_id)` with columns: `# | Ctx | Flash | KV | Eval | Phys | Parallel | Checkpoints | Experts | Status | Score | Gen tok/s | Quality | Error`
- ✅ Score-descending for passed, failed at bottom
- ✅ `n/a` for missing metrics
- ✅ Error trimmed to 120 chars
- ✅ Pipe-escaping via `_cell`
- ✅ Reuses `_fmt` for display
- ✅ Wired into `save_best_report` after "All tried configurations" header
- ✅ TDD: tests written first (in commit 763c1fe)
- ✅ ASCII-only output
- ✅ No changes to winner selection or scoring logic

### Code Quality ✓
- Mirrors existing helper style (`_verdict`, `_fmt`, `_tried`, `_cell`)
- Proper error handling with try/except for None-safe operations
- Consistent formatting with existing markdown table patterns
- Memory-safe (returns list of strings, not single large string)

## Concerns
None. The task is fully implemented and tested. All 14 tests pass. The implementation matches the specification exactly and follows the existing codebase conventions.

## Status
**DONE** - No further action needed for Task 1.