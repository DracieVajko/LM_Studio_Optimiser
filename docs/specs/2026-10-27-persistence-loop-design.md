# Persistence loop design

**Date:** 2026-10-27
**Status:** Approved (design gate)
**Scope:** Close the loop around the per-model LM Studio load defaults introduced in
`docs/LM_STUDIO_DEFAULTS.md`: backups, diff, restore, optional verify, and a model
detail card.

## 1. Goal

The first release of `lmstudio_defaults` can write a winner. It cannot yet answer
"what will change", "undo it" or "did LM Studio really apply it". The persistence
loop adds these so saving a default is reversible, reviewable and verifiable.

## 2. Non-goals

- No new search space, campaign runner or notification work.
- No change to the optimizer core or benchmark stages.
- No live LLM/hardware runs during implementation; `--verify` stays mocked-tested
  and is marked HW-pending.

## 3. Backup model

- Timestamped backups next to the defaults file:
  `<name>.optimizer-backup-<UTC timestamp>` (e.g. `...gguf.json.optimizer-backup-20261027T094612Z`).
- Keep the newest 10; prune older ones after each backup.
- The legacy single `.optimizer-backup` from the previous release is still
  recognized and restorable.
- Every real restore creates a backup of the current state first, so restore is
  reversible.
- If the target backup is byte-identical to the current file, the operation is a
  no-op: no backup is created and `noop=true` is returned.

## 4. Diff

`diff_defaults(model_id, config)` returns:

- `changes`: only added/changed mapped keys, each `{key, current, winner, kind}`.
- `unchanged`: mapped keys that are already equal.
- `preserved`: keys present in the current file that the winner does not manage
  (CPU threads, K/V cache quantization, mmap, ...).
- `exists`: whether the defaults file exists today.

Diff reuses `_updates_for`, so legacy `llama.speculativeDecoding.*` keys are
aligned with canonical `speculativeDecoding.*` keys exactly as in the writer.

## 5. Restore

- `restore_defaults(model_id, backup="latest", dry_run=False, keep_backups=True)`
  reads the chosen backup, validates that it is a JSON object with a `load.fields`
  list, and writes it atomically.
- `backup` may be `"latest"` or a bare filename. Absolute paths and `..` are
  rejected; only names returned by `list_backups()` are accepted.
- Before writing, the current file is backed up with a fresh timestamp.
- If the backup equals the current file, return `noop=true` and write nothing.
- Invalid JSON in the backup raises `DefaultsError`; nothing is overwritten.

## 6. Verify

`verify_defaults(model_id, config, client)` is async and uses the existing
`LMStudioClient`:

1. Ensure the client is not currently loading the model (best effort).
2. `load_model(model_id, LoadConfiguration())` — no explicit parameters, so LM
   Studio resolves the saved default; `echo_load_config=True` is already sent by
   the client.
3. Compare only fields the echo actually carries:
   `context_length`, `flash_attention`, `offload_kv_cache_to_gpu`,
   `eval_batch_size`, `physical_batch_size`, `parallel`, `context_checkpoints`,
   `reasoning_budget_message`, `num_experts`, `speculative_*`.
4. `gpu_ratio` and `rope_freq_*` are reported under `unverified`; they are not
   REST-echoable and no claim is made about them.
5. Unload by `identifier` in a `finally` block; report `unloaded`.

Result shape:

```json
{
  "verified": true,
  "checked": {"context_length": {"expected": 8192, "actual": 8192, "match": true}},
  "mismatches": {},
  "unverified": ["gpu_ratio", "rope_freq_base", "rope_freq_scale"],
  "unloaded": true,
  "error": null
}
```

A load failure sets `verified=false`, records `error`, and still attempts unload.

## 7. Surfaces

CLI:

```bash
lm-optimizer lab diff-default <run-id>
lm-optimizer lab restore-default <model-id> [--backup NAME] [--dry-run] [--yes]
lm-optimizer lab save-default <run-id> --verify
lm-optimizer lab verify-default <run-id>
```

API:

```
GET  /api/model-default?model_id=...
GET  /api/runs/{run_id}/default-diff
POST /api/model-default/restore
POST /api/runs/{run_id}/verify-default
POST /api/runs/{run_id}/save-default?verify=true
```

UI:

- Results: "Save as LM Studio default" first opens a diff modal (old -> new,
  unchanged count, preserved count). "Save anyway" writes. If there are no
  changes, show "already up to date".
- Model detail: new "LM Studio default" card with existence, current values,
  backup count/latest, restore, diff vs latest winner, save winner.

All file operations work while LM Studio is offline except `verify`.

## 8. Error handling and security

- Model identifiers are resolved through `resolve_defaults_path`; `..` and
  absolute/drive paths raise `UnsafePathError`.
- Restore accepts only backup names returned by `list_backups()`.
- Unreadable or non-object files raise `DefaultsError` and are never overwritten.
- Backup pruning never deletes a file whose name is not produced by this service.
- `verify` never writes; it only loads and unloads.

## 9. Verification

- Unit tests for backup naming/retention/legacy recognition, restore no-op and
  invalid-backup refusal, diff, and expected-echo mapping.
- Async verify tests with a fake client: match, mismatch, unverified fields,
  load failure, unload in `finally`.
- API, CLI and static UI guard tests.
- Full test suite plus CI on ubuntu/macos/windows.
- No real LLM/hardware load during implementation; live verify is HW-pending.

## 10. Files to touch

- `lm_optimizer/services/lmstudio_defaults.py`
- `lm_optimizer/services/lmstudio_default_verify.py`
- `lm_optimizer/api/routes.py`, `lm_optimizer/api/schemas.py`
- `lm_optimizer/cli/lab.py`
- `lm_optimizer/ui/static/js/api.js`, `results.js`, `model-detail.js`
- `lm_optimizer/ui/templates/model_detail.html`
- `tests/...`
- `docs/LM_STUDIO_DEFAULTS.md`, `docs/README.md`, `CHANGELOG.md`
