# auto-run

Complete every model optimization that is not finished yet, and re-run only the
work that a real change invalidated.

```
lm-optimizer auto-run [MODELS...] [OPTIONS]
```

## What it does

1. Discovers models from LM Studio (load-free discovery).
2. Filters by `--skip` / `--max-size-gb`, sorts smallest first.
3. For each model, looks at its latest run and the current **fingerprint**.
4. Applies the decision matrix, then executes the chosen action.
5. Writes a markdown report.

## Decision matrix

| Run state | Fingerprint | Action |
|-----------|-------------|--------|
| No run | — | `FULL` (fresh optimize) |
| failed / cancelled | any | `FULL` |
| complete | matches | `SKIP` (nothing to do) |
| complete | differs | `RE_RUN` (full optimize) |
| incomplete (running/paused/interrupted/resumed) | matches | `RESUME` |
| incomplete | differs in environment fields | `RESUME` + revalidate affected |
| incomplete | absent fingerprint | `RESUME` (fingerprint backfilled) |

A **complete** run is `success`, `completed`, or `completed_no_validated_winner`.

## Fingerprint

SHA256 over these components, stored in `runs.benchmark_params_json`
(no schema migration):

| Field | Source |
|-------|--------|
| `model_id` | model identity |
| `quantization` | model identity |
| `architecture` | model identity |
| `lm_studio_version` | `lms` CLI |
| `hardware_fingerprint` | GPU names + VRAM, CPU cores, RAM (16-char prefix) |
| `benchmark_suite_version` | `lm_optimizer/benchmark/suite_version.py` |

Two rules matter:

- **Absent fingerprint is unknown, not changed.** A run created before
  fingerprinting is preserved: completed work is skipped, incomplete work is
  resumed, and the fingerprint is backfilled so the next run compares honestly.
- **Identity fields never scope a revalidation.** `model_id`, `quantization`
  and `architecture` say *which* model the run is. A run is looked up by model
  id, so a stale stored value cannot invalidate measurements that are still
  valid on this host.

### Revalidation scope

| Changed field | Effect on an incomplete run |
|---------------|------------------------------|
| `lm_studio_version` | all completed configs |
| `hardware_fingerprint` | all completed configs |
| `benchmark_suite_version` | only quality/recovery/validation phases |
| identity fields | none (excluded) |

Completed measurements are never deleted; affected configs are removed from the
in-memory tested set so they are measured again, and their new results replace
the old rows.

## Options

| Option | Default | Meaning |
|--------|---------|---------|
| `--profile` | `balanced` | speed / balanced / context / quality |
| `--quality` | `0.97` | minimum quality threshold |
| `--style` | `balanced` | precise / balanced / creative |
| `--repetitions` / `--validation` | 3 / 5 | benchmark + validation reps |
| `--max-context` | 32768 | maximum context length |
| `--min-gpu` / `--max-gpu` | 0.0 / 1.0 | GPU ratio bounds |
| `--workload` | `interactive` | interactive / throughput |
| `--skip` | — | comma-separated model id substrings |
| `--max-size-gb` | — | only models up to this file size |
| `--revalidate-fingerprint` | off | ignore the fingerprint and re-check |
| `--dry-run` | off | print the plan, run nothing |
| `--output` | `results` | report directory |
| `--yes` / `--non-interactive` | off | skip confirmation |

## Output

`results/auto-run-YYYYMMDD-HHMMSS.md` with a summary table, one section per
model (action, run id, best score, error), a fingerprint-changes table, and the
environment (GPU/CPU/RAM, LM Studio version, benchmark suite version).

## Preview without touching anything

```bash
lm-optimizer auto-run --dry-run
```

## Notes

- Never loads a model to decide: discovery is load-free.
- Completed models are reported, not re-measured.
- `completed_no_validated_winner` runs are skipped; raising the quality
  threshold and re-running that model deliberately is a separate decision.