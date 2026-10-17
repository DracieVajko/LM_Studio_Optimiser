# Safety Review — Stale `running` Runs (dry-run only)

- **Date:** 2026-10-14
- **Scope:** the six optimization runs whose persisted status is `running`
- **Mode:** `lm-optimizer reconcile-runs` **default dry-run only** (no `--apply`)
- **Author:** automated review (read-only)

## Method & safety controls

1. Captured a pre-run integrity snapshot of the active SQLite database, the
   checkpoint directory, and the log file.
2. Ran the dry-run: `python -m lm_optimizer.cli.main reconcile-runs`
   (entry point equivalent to `lm-optimizer reconcile-runs`; the `lm-optimizer`
   console script is not installed on PATH in this environment).
3. Ran a read-only detail pass over every live run (per-run status, timestamps,
   config breakdown, winner policy, checkpoint age).
4. Captured a post-run integrity snapshot and compared bytes.
5. **No `--apply` was passed. No run status, checkpoint, report, or preset was
   modified. No model was started.**

Commands used:

```
python -m lm_optimizer.cli.main reconcile-runs
python C:\Users\DracieVajko\AppData\Local\Temp\opencode\stale_running_review.py
```

## Integrity evidence (dry-run made zero persistent changes)

| Artifact | Before | After | Changed? |
|---|---|---|---|
| `data/optimizer.db` SHA-256 | `61B021F27B6D877529DFB94AFB8FD1B9670119F91082ACA216797AE8883C45EE` | `61B021F27B6D877529DFB94AFB8FD1B9670119F91082ACA216797AE8883C45EE` | **No** |
| `data/optimizer.db` size | 5 033 984 B | 5 033 984 B | **No** |
| `data/optimizer.db` mtime | 2026-10-13 23:31:16 | 2026-10-13 23:31:16 | **No** |
| `optimizer.db-wal` / `-shm` | absent | absent | **No** |
| `data/reports/checkpoints/` count | 0 files | 0 files | **No** |
| `data/logs/optimizer.log` size | 46 481 307 B | 46 481 307 B | **No** |

The dry-run connection performed read-only queries only; the SHA-256 of the
database is byte-identical and the mtime is unchanged.

> Note: the real checkpoint directory is `data/reports/checkpoints/` (not
> `data/checkpoints/`, which is empty). All six runs have a checkpoint file
> there.

## Dry-run output (verbatim)

```
                            Stale runs to interrupt
┌──────────────────────────┬──────────────────────────┬─────────┬─────────────┐
│ Run ID                   │ Model                    │ From    │ To          │
├──────────────────────────┼──────────────────────────┼─────────┼─────────────┤
│ 4aac923b-697c-405e-a562… │ qwen3.8-27b-uncensored-… │ running │ interrupted │
│ bae3386f-cb2c-4ce6-9c2a… │ qwen3.8-9b-distill       │ running │ interrupted │
│ f518131b-359c-4a32-905c… │ qwen3.8-9b-distill       │ running │ interrupted │
│ d6580613-36c5-4407-98a2… │ qwen3.8-9b-distill       │ running │ interrupted │
│ a1ac6c9c-8269-4d4a-8602… │ qwen3.8-9b-distill       │ running │ interrupted │
│ 3b1be9a9-9d55-4ffe-aeb2… │ prism-ml/bonsai-27b      │ running │ interrupted │
└──────────────────────────┴──────────────────────────┴─────────┴─────────────┘
Dry run: 6 stale run(s). Re-run with --apply to mark them interrupted (history
preserved).
```

## Per-record detail

| # | Run ID | Model | Status / stage | created_at | best_config_id | Configs | Winner policy | Checkpoint age | Proposed | Review |
|---|--------|-------|----------------|------------|----------------|---------|---------------|----------------|----------|--------|
| 1 | `3b1be9a9-9d55-4ffe-aeb2-0741fb87dcd3` | prism-ml/bonsai-27b | running / speed_discovery | 2026-10-12T01:20 | — | 13 (12 passed, 1 load_failed) | NO_WINNER | 64.14 h | interrupted | safe |
| 2 | `a1ac6c9c-8269-4d4a-8602-14fde91a67ee` | qwen3.8-9b-distill | running / recovery | 2026-10-12T07:48 | — | 22 (16 passed, 1 load_failed, 5 quality_failed) | NO_WINNER | 56.83 h | interrupted | safe |
| 3 | `d6580613-36c5-4407-98a2-aff7ff61f1b6` | qwen3.8-9b-distill | running / recovery | 2026-10-12T10:25 | — | 19 (16 passed, 1 load_failed, 2 quality_failed) | NO_WINNER | 54.53 h | interrupted | safe |
| 4 | `f518131b-359c-4a32-905c-9d528c7b4fbe` | qwen3.8-9b-distill | running / recovery | 2026-10-12T21:12 | — | 19 (16 passed, 1 load_failed, 2 quality_failed) | NO_WINNER | 43.55 h | interrupted | safe |
| 5 | `bae3386f-cb2c-4ce6-9c2a-43892cd46357` | qwen3.8-9b-distill | running / final_validation | 2026-10-12T22:51 | `7425e701-9b64-4a5d-a900-97c1a2588854` | 21 (17 passed, 1 load_failed, 3 quality_failed) | **SUCCESS (validated winner)** | 41.44 h | interrupted | **manual review** |
| 6 | `4aac923b-697c-405e-a562-2d6485f22bcf` | qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp | running / quality_check | 2026-10-13T20:41 | — | 19 (17 passed, 1 load_failed, 1 benchmark_failed) | **SUCCESS (validated winner, score+quality evidence)** | 18.86 h | interrupted | **manual review** |

All `completed_at` values are `None`; all `duration_seconds` are `0.0` — these
runs never finalised.

## Policy verification

- **Reconciliation policy** (`services/run_reconcile.py`): a run is reconciled
  only when its status is `running`/`resumed` **and** it has no checkpoint newer
  than the 2 h stale window. All six are `running` with checkpoint ages
  18.86–64.14 h (> 2 h) → all six are stale → proposed `interrupted`. ✓
- **`final_status_for_run`** is applied only on the completion path
  (`_finalize`). Reconcile never calls it; it marks a dead *running* run
  `interrupted`, which is the truthful terminal state for a dead process. No run
  currently persisted as `success`/`completed` is touched. ✓
- **Winner-downgrade rule (item 4):** none of the six is being downgraded from a
  persisted `success`. Four classify as `NO_WINNER` (they would have finalised as
  `completed_no_validated_winner` had the campaign *completed* — but it did not,
  so `interrupted` is more accurate, not a downgrade). Two classify as `SUCCESS`
  (validated winner present) and **are flagged for manual review** — they are
  better *resumed* than abandoned. ✓ (flagged, not silently applied)
- **Idempotent & non-destructive:** the reconcile only changes
  `running/resumed` rows; it never deletes results, checkpoints, or runs. ✓

## Findings

- **4 of 6 transitions are unambiguously safe** (rows 1–4): the model generated
  usable output but no configuration passed the quality threshold; the campaign
  was interrupted mid-recovery. `interrupted` is correct and truthful.
- **2 of 6 require manual review** (rows 5–6): both hold a validated winner and
  were interrupted mid `final_validation` / `quality_check`. Marking them
  `interrupted` is factually correct, but **resuming** them would likely let them
  finalise as `success` and preserve the found winner. Recommend **resume, not
  abandon**, for these two before any bulk reconcile.
- No winner is silently downgraded; no completed/success run is affected.

## Recommendation

Do **not** run `--apply` yet. Two options:

1. **Preferred:** resume rows 5 and 6 (`lm-optimizer resume <run-id>`) so their
   validated winners can finalise; then reconcile the remaining 4 dead runs.
2. Or accept `interrupted` for all six if the campaign results are no longer
   needed.

**Status: dry-run review complete. All six proposed transitions follow the
implemented reconciliation policy; 4 are safe as-is, 2 need manual review.
Awaiting approval before applying any database change.**
