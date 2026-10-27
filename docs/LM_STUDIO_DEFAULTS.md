# LM Studio per-model load defaults

How the optimizer persists a validated winner so the next LM Studio load uses
it, without a public REST endpoint and without restarting LM Studio.

## Where LM Studio stores the values

```
~/.lmstudio/
  .internal/
    model-data.json                         # concrete model path -> uiIdentifier
    user-concrete-model-default-config/
      <publisher>/<model>.json              # hub model, e.g. prism-ml/bonsai-27b
      <publisher>/<repo>/<file>.gguf.json   # concrete GGUF, e.g. .../Qwen3.8-9B-Q4_K_M.gguf.json
```

File shape:

```json
{
  "preset": "",
  "operation": {"fields": [{"key": "llm.prediction.temperature", "value": 0.7}]},
  "load": {"fields": [{"key": "llm.load.contextLength", "value": 8192}]}
}
```

`model-data.json` maps a concrete path to the UI identifier used by the API
(`uiIdentifier`). That is how a run for `swift-qwen3.8-27b@q2_k` or
`qwen3.8-9b-distill` resolves to the exact `.gguf.json` file instead of a
sibling quantization.

## Verified behaviour

Live test on 2026-10-27 with `qwen3.8-9b-distill`:

1. Wrote `llm.load.contextLength=3072`, `llm.load.llama.flashAttention=false`,
   `llm.load.offloadKVCacheToGpu=false` (and disabled q4_0 KV quantization so
   flash-off is legal).
2. `POST /api/v1/models/load` with `{"model": "...", "echo_load_config": true}`
   and no explicit load parameters.
3. Server echoed `context_length=3072`, `flash_attention=false`,
   `offload_kv_cache_to_gpu=false`.
4. Unloaded by `instance_id`; the original defaults file was restored
   byte-for-byte (SHA-256 match).

No LM Studio restart was needed between the write and the load.

## Field mapping used by `lmstudio_defaults.py`

| Optimizer field | LM Studio key | Notes |
|---|---|---|
| `context_length` | `llm.load.contextLength` | live-verified |
| `gpu_ratio` | `llm.load.llama.acceleration.offloadRatio` | float 0..1 |
| `flash_attention` | `llm.load.llama.flashAttention` | live-verified |
| `offload_kv_cache_to_gpu` | `llm.load.offloadKVCacheToGpu` | live-verified; no `llama.` prefix |
| `eval_batch_size` | `llm.load.llama.evalBatchSize` | live-verified |
| `physical_batch_size` | `llm.load.llama.physicalBatchSize` | schema-confirmed |
| `parallel` | `llm.load.numParallelSessions` | schema-confirmed |
| `context_checkpoints` | `llm.load.llama.contextCheckpoints` | schema-confirmed |
| `reasoning_budget_message` | `llm.load.llama.reasoningBudgetMessage` | schema-confirmed |
| `num_experts` | `llm.load.numExperts` | schema-confirmed |
| `rope_freq_base` | `llm.load.llama.ropeFrequencyBase` | stored as `{"checked": true, "value": N}` |
| `rope_freq_scale` | `llm.load.llama.ropeFrequencyScale` | stored as `{"checked": true, "value": N}` |
| `speculative_draft_*` | `llm.load.speculativeDecoding.*` | current UI schema; an existing legacy `llm.load.llama.speculativeDecoding.*` key is updated in place |

“Schema-confirmed” means the key appears in LM Studio's own renderer bundle and
defaults files but was not part of the live echo test. Values that the
optimizer never searches (CPU thread pool, K/V cache quantization, mmap,
keep-in-memory) are left untouched.

## Safety rules

- Only non-`None` winner fields are written; nothing is guessed.
- Unknown `load.fields`, `operation.fields` and unknown top-level keys survive
  the merge.
- The previous file is copied to `<name>.optimizer-backup` before a real write.
- Writes go to a temporary file in the same directory and use `os.replace`
  (atomic on Windows and POSIX).
- Unreadable JSON or a non-object document is never overwritten: the CLI exits
  1 and the API returns HTTP 409.
- Model identifiers are validated against the defaults directory; `..` and
  absolute/drive paths are rejected.

## Persistence loop (backups, diff, restore, verify)

Backups are timestamped (`<file>.optimizer-backup-<utc>`); the newest 10 are kept. A legacy `.optimizer-backup` file is recognized but not overwritten by new backups.

```bash
lm-optimizer lab diff-default <run-id>
lm-optimizer lab restore-default <model>
lm-optimizer lab restore-default <model> --yes
lm-optimizer lab verify-default <run-id>
```

API: `GET /api/model-default`, `GET /api/runs/{run_id}/default-diff`, `POST /api/model-default/restore`, `POST /api/runs/{run_id}/verify-default`, `POST /api/runs/{run_id}/save-default?verify=true`. Diff compares current defaults against the winner; restore validates JSON before writing; verify loads with an empty config and compares only echoed fields, always unloading in `finally`.

Results page shows a diff summary before save; the model detail page shows an "LM Studio default" card with current fields and backup count.

## Usage

```bash
lm-optimizer lab save-default <run-id>            # write
lm-optimizer lab save-default <run-id> --dry-run  # show fields only
lm-optimizer lab save-default <run-id> --no-backup
```

```http
POST /api/runs/{run_id}/save-default
POST /api/runs/{run_id}/save-default?dry_run=true
```

Results page: **Save as LM Studio default** (separate from the one-shot
**Apply Best Config**, which never changes saved defaults).

The API response contains `path`, `fields`, `created`, `backup_path` and
`dry_run`. `LM_OPTIMIZER_LMSTUDIO_HOME` can point the service at a portable
LM Studio home; it only affects this optimizer, not LM Studio itself.

## Limitations

- This is an LM Studio implementation detail, not a public API. A future
  LM Studio release may rename keys or move the directory. If the file layout
  changes, resolution fails loudly instead of writing somewhere arbitrary.
- Each GGUF quantization has its own defaults file. Saving for
  `model@q2_k` does not change `model@q4_k_s`.
- The file channel is persistence, not tuning: it does not make KV-cache
  quantization, CPU threads or mmap searchable by the optimizer.
