# SDD ledger — plan: docs/superpowers/plans/2025-10-05-full-reports-deep-research.md

Branch: feat/deep-research (from main 03f514f).
BASE: 03f514f066a96ab5c4e7adfba9aecf8d43720fa1

## Pre-flight conflict scan

| Pair / Task | Produces vs consumes | Finding |
|---|---|---|
| T1 → T2, T3 | T1 `_all_configs_rows` used by T2 appendix + T3 web table | Sequential, clean |
| T2 → T4 | T2 `_full_outputs_section` used by T4 backfill | Sequential, clean |
| T5→T6→T7→T8 | Deep Research chain | Sequential, clean |
| T1,2 ↔ T5-8 | Deep Research reuses `_all_configs_rows`, `_config_outputs` | Stacked on full-reports base — rebase after merge |

Ruling: plan self-consistent. Dispatch Task 1.

## Completions
- Task 1: complete (commits 763c1fe) — All-configs table in .md
- Task 2: complete (commits bb07ba5) — Full verbatim outputs appendix
- Task 3: complete (commits db5a5e2) — Web all-configs table
- Task 4: complete (commits e91a752) — `re-report` backfill CLI
- Task 5: complete (commits eb1b199) — Deep suite + metrics
- Task 6: complete (commits 12e62f4, 25dc977, 626844e, cf8a74d) — Single-model runner + CLI
- Task 7: complete (commits cf8a74d) — Multi-model batch + resume
- Task 8: complete (commits 712fd82) — Leaderboard + preview UI + API
- Task 9: complete (commit 6bb5f2c) — YAML prompts override
- Task 10: complete (commit 84e83a4) — Integration + docs

## Final Verification
- **Full test suite**: 561 passed, 2 warnings (pre-existing Starlette deprecation)
- **UI syntax check**: node --check exit 0 on all 11 JS files
- **Backward compatibility**: All existing reports render; existing CLI commands unchanged
- **No unrelated files touched**: README.md, run-web.bat, test_web_resilience.py, docs deletions, docs/archive/ left untouched

## Rulings
- Ruling: single load per model enforced (Task 6 fix round 1)
- Ruling: UnloadNotClean raised on dirty unload (Task 6 fix round 2)
- Ruling: public BenchmarkService wrappers added (no behavior change)
- Ruling: deep parses .md reports instead of JSON sidecars (Task 8)
- Ruling: elapsed asc as TTFT stand-in documented
- Ruling: stacked branch contains full-reports commits; rebase after merge
- Ruling: YAML prompts override implemented with CWD → config/ → defaults precedence
- Ruling: CLI `--prompts-file` option added with explicit help text

## Final Status
**All 10 tasks complete.** Plan fully implemented and tested.