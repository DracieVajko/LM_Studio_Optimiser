# Safe Resume Feasibility Audit — Six Stale Runs (read-only)

- **Date:** 2026-10-14
- **Mode:** read-only inspection only. No `reconcile-runs --apply`, no `resume`
  execution, no live LM Studio/Ollama calls, no downloads, no production writes,
  no commits.
- **DB integrity re-checked after every step:** `data/optimizer.db` SHA-256
  `61B021F27B6D877529DFB94AFB8FD1B9670119F91082ACA216797AE8883C45EE` — unchanged.

## 1. Resume implementation trace (end to end)

`lm-optimizer resume <run_id>` → `cli/main.py::_run_resume` → acquires the
cross-process job lock (`_acquire_run_lock("resume:<id>")`, prevents duplicate
workers) → `client.connect()` → `AdaptiveOptimizer.resume_from_checkpoint`.

`resume_from_checkpoint` (services/optimizer.py:3122):

1. `run_repo.get(run_id)` — run must exist.
2. `load_checkpoint(run_id)` — checkpoint must exist and parse (raises on corrupt).
3. **Model-identity guard:** `ckpt["model"] != run.model.id` → refuse resume.
4. `_rebuild_resume_space` — search space from DB `run.search_space` (fallback ckpt).
5. `_restore_tested_state` — **completed candidates come from the DB**
   (`self.state.tested_configs = list(full.configurations)`); winner restored via
   `full.best_config_id`; pareto restored; failed regions rebuilt; recovery state
   reconciled (longer log wins, DB authoritative, greatest cursor wins).
6. Sets `run.status = RESUMED`, `stage = DISCOVERY`, saves run.
7. **Pipeline** (`_resume_phase_pipeline`): speed → quality → recovery →
   final validation → (phase B if enabled) → `_finalize`. Every stage skips
   already-tested `_candidate_key`s; recovery skips `recovered_ids` from the log.
8. `_finish_resume`: on `PauseRequested` → keep PAUSED; on `CancelledError` →
   CANCELLED; `finally` stamps `completed_at`/`duration_seconds` and saves.

Report/preset writing is **not** part of resume: `_run_resume` only calls
`_display_optimization_result`. No `save_best_report` / `_save_preset_and_report`
on the resume path.

## 2. Resumability + live-op assessment

| Run | Checkpoint | Model-identity guard | phase pipeline | Technically resumable | Executes live model ops |
|---|---|---|---|---|---|
| bae3386f…6357 | present, `recovery:unrecovered:3` | ckpt model == run model ✓ | yes (phase_ab present) | **yes** | **yes** |
| 4aac923b…2bcf | present, `phase:speed` | ✓ | yes | **yes** | **yes** |

Resume **always** executes live model work: speed/quality probes not yet tested,
final-validation repetitions on the winner, plus `ensure_unloaded` on the way out.
It requires LM Studio to be free and the model loadable.

## 3. Invariant verification (code-level)

- **Validated winner not lost/overwritten:** `_restore_tested_state` restores
  `best_config` from `best_config_id`; `_finalize` nulls it only if its status is
  not PASSED. A PASSED winner is preserved. `_update_best` replaces only with a
  *higher*-scoring candidate (normal optimisation, not a loss).
- **No unnecessary repeat of completed work:** tested candidate keys are skipped
  in every stage; recovery skips logged `config_id`s (pinned by
  `test_resume_reconcile.py`, `test_pause_hardening.py`).
- **No duplicate candidate records:** a new UUID is written only for a *new*
  candidate key; `_record_probe_result` de-dups by id.
- **Checkpoint not discarded:** resume loads and rewrites it; nothing deletes it.
- **No premature `success`:** `_finalize` is the last pipeline step (after final
  validation). Pause → PAUSED, cancel → CANCELLED, and `_finish_resume`'s
  `finally` never sets a success status. `success` requires the completion path.
- **No preset/report overwrite:** neither `_finish_resume` nor `_run_resume`
  writes presets or reports.

## 4. Model availability (read-only DB, NOT live)

| Model ID | DB quantization | ctx_limit | last_seen (DB) |
|---|---|---|---|
| `qwen3.8-9b-distill` | Q4_K_M | 262144 | 2026-10-12T22:51 |
| `qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp` | IQ3_XS | 262144 | 2026-10-13T20:41 |
| `prism-ml/bonsai-27b` | Q1_0 | 262144 | 2026-10-13T07:51 |

**Current live availability is UNKNOWN** — strict limits forbid a provider call.
The user reported models were deleted and are being re-downloaded, so the
currently-installed quantization may differ from the historical one.

## 5. Same model version / quantization?

- `resume <run_id>` takes **no model argument** → it cannot substitute a different
  model ID. Strong safety property.
- There is **no quantization guard**: the checkpoint stores
  `model_context_limit` but not quantization. If the model was re-downloaded at a
  different quant under the same ID, resume would silently measure the new quant.
  **This must be confirmed by an authorized read-only `/api/models` check before
  resuming** (compare `quantization` against the table above).

## 6. Reconcile preserves artifacts (the other four)

`services/run_reconcile.py` only sets `status=interrupted`, `completed_at`, and
appends an `error` note, then `run_repo.save`. It issues **no DELETE**. Candidates
(`configurations`), reports (`results/`), checkpoints
(`data/reports/checkpoints/`), and diagnostics all survive. The API `abandon`
path also no longer removes the checkpoint.

## 7. Backup requirement + recovery procedure

A status/`best_config_id` change is a persistent SQLite write; take a verified
backup first. The repo has **no built-in backup command**.

**Backup (before any apply/resume):**
1. Confirm no optimizer process/job lock is active.
2. Consistent DB copy via the SQLite online backup API (WAL-safe):
   `python -c "import sqlite3; s=sqlite3.connect('data/optimizer.db'); d=sqlite3.connect(r'<BACKUP>/optimizer.db'); s.backup(d); d.close(); s.close()"`
3. Copy `data/reports/checkpoints/` → `<BACKUP>/checkpoints/`.
4. Verify: `PRAGMA integrity_check` on the copy + row counts for `runs`,
   `configurations`.
5. Record SHA-256 of the copy.

**Recovery:**
1. Stop any optimizer/LM Studio optimizer task.
2. Remove `optimizer.db-wal`/`-shm` if present; restore `<BACKUP>/optimizer.db` to
   `data/optimizer.db`; restore the checkpoints directory.
3. `PRAGMA integrity_check`; re-verify `runs.status` for the affected ids.

## Per-run verdict

| # | Run ID | State | Winner | Checkpoint readiness | Proposed action | Risk |
|---|---|---|---|---|---|---|
| 1 | `3b1be9a9…dcd3` | running / speed_discovery | none | ckpt at run start (0 completed) | reconcile → interrupted | low |
| 2 | `a1ac6c9c…67ee` | running / recovery (cursor 4) | none | ckpt `recovery:unrecovered:4`, 21 completed | reconcile → interrupted | low |
| 3 | `d6580613…f1b6` | running / recovery | none | ckpt `phase:quality`, 19 completed | reconcile → interrupted | low |
| 4 | `f518131b…4fbe` | running / recovery (cursor 2) | none | ckpt `recovery:rollback`, 19 completed | reconcile → interrupted | low |
| 5 | `bae3386f…6357` | running / final_validation | **yes** (`7425e701…`, best_candidates=1) | ckpt 20 completed | resume (preferred) or reconcile | medium: quant drift; live ops |
| 6 | `4aac923b…2bcf` | running / quality_check | **yes** (implicit; best_config_id null) | ckpt `phase:speed`, 18 completed (DB 19) | resume (preferred) or reconcile | medium: quant drift; ckpt/DB phase mismatch |

## Recommendations

**1. Safe to reconcile as `interrupted`**
Rows 1–4: no validated winner, dead process, checkpoints/reports/candidates
preserved. Reconcile is truthful and non-destructive.

**2. Winner-bearing, potentially safe to resume (conditional)**
Rows 5 and 6 — only after: (a) an authorized read-only model-availability check
confirming the same ID **and quantization** is installed; (b) LM Studio confirmed
free for the whole run; (c) a verified DB+checkpoint backup is taken. Resume
preserves the winner and does not duplicate work.

**3. Needs further investigation before action**
- Rows 5 and 6: quantization-drift risk (possible re-download) must be resolved.
- Row 6 additionally has a DB(`quality_check`/phase `speed`) vs
  checkpoint(`speed_discovery`/phase `speed`) phase mismatch; resume would
  re-enter SPEED and skip tested keys — expected, but verify it re-derives the
  same winner.

**Do not apply reconciliation or resume any run until explicitly authorized.**
