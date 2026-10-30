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

## Model eligibility

Only text-generation / chat models are optimization targets. An embedding,
encoder or reranker model has no token-generation throughput and no
instruction-following quality, so benchmarking it burns hours and produces
meaningless numbers.

| Signal | Verdict |
|--------|---------|
| provider `type` is `llm` / `chat` / `text-generation` / … | eligible |
| provider `type` is `embedding` / `reranker` / `encoder` / `bert` / … | excluded |
| provider `type` unrecognized | **unknown** (not eligible) |
| no provider `type`, encoder-only architecture | excluded |
| no provider `type`, name/architecture signal | excluded |
| no provider `type`, no signal at all | **unknown** (not eligible) |

Unknown is never silently assumed to be a generation model. Use
`--force-generation` to opt a model in; the override cannot invent evidence
where none exists, so a model with no signal at all stays unknown.

An excluded model is **not a failure** — it was never a candidate. Historical
runs for those models remain visible in history views.

Provider metadata wins over name matching: a model typed `llm` stays eligible
even if its name contains `embed`.

## Concurrency

`auto-run` takes the same job lock as `optimize`, `auto`, `fit`, `ctx`,
`quality-ladder` and `resume` — one optimizer job per host. A second
invocation exits 1 immediately with a clear message, before discovery, so it
cannot mutate run state, checkpoints or provider state. The lock is released on
both the normal and the exception path. Stale locks from a dead process are
reclaimed; PID reuse is detected by comparing the holder's recorded creation
time.

## Fingerprint

SHA256 over these components, stored in `runs.benchmark_params_json`
(no schema migration):

| Field | Source |
|-------|--------|
| `model_id` | model identity |
| `quantization` | model identity |
| `architecture` | model identity |
| `lm_studio_version` | **compatibility key** (see below) |
| `hardware_fingerprint` | GPU names + VRAM, CPU cores, RAM (16-char prefix) |
| `benchmark_suite_version` | `lm_optimizer/benchmark/suite_version.py` |

### Version identity

`lms --version` reports a build identifier (`CLI commit: dcefd5a`), not an LM
Studio semantic version. Fingerprinting that raw string revalidated every run on
each CLI rebuild. The fingerprint now uses a normalized **compatibility key**:

| Source string | `semantic` | `compatibility_key` |
|---------------|-----------|---------------------|
| `0.3.12` | `0.3.12` | `semver:0.3.12` |
| `CLI commit: dcefd5a` (CLI does not load) | — | `build:non-loading` |
| `CLI commit: dcefd5a` (CLI performs loads) | — | `build:dcefd5a` |
| `unknown (no lms CLI)` | — | `unknown` |

A real version change always matters. A CLI rebuild only matters when `lms`
actually performs loads for us. The raw string and the parsed identity are kept
under `benchmark_params.fingerprint_components._version_identity` for
diagnostics and are excluded from the diff.

> **One-time transition:** runs recorded before this change stored the raw
> string. They will report a single `lm_studio_version` diff on the next
> auto-run, then compare stably.

Two further rules:

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
| `lm_studio_version` (compatibility key) | all completed configs |
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

Two report kinds, distinguished by filename and title:

| Kind | Filename | Claims |
|------|----------|--------|
| plan (`--dry-run`) | `auto-run-plan-<ts>.md` | intent only: proposed action, fingerprint diff, existing run id. No score, no status, no result run id. |
| results | `auto-run-results-<ts>.md` | real run ids, scores, statuses and errors |

Both include a summary table, one section per model, a failures section
(results only), a fingerprint-changes table, and the environment. Counts always
reconcile: the per-action counts sum to the stated total.

Filenames carry microseconds plus a numeric suffix, so two invocations in the
same second never overwrite each other.

## Known historical artifacts

`results/auto-run-2026*.md` (8 files) predate this fix and were written by a
**unit test** into the repository's persistent `results/` directory instead of a
temporary one. Their content is a test fixture (`m1`/`run1`/`0.900`,
"Connection failed") and their timestamps correspond one-to-one with test-suite
runs. They are not real results. They were left in place as evidence; delete
them whenever convenient.

Fixes applied:

- `_generate_report` takes an explicit `kind` and refuses to render outcomes in
  a plan report (and refuses unknown kinds outright).
- Every report-writing test now uses pytest's `tmp_path`.
- Report filenames can no longer collide.
- A record with an error stays an error; no score or run id is invented.
- Execution failures record `action="FAILED"` instead of the intended action,
  so a failure can never be filed under the action it failed to perform.

## Preview without touching anything

```bash
lm-optimizer auto-run --dry-run
```

## Notes

- Never loads a model to decide: discovery is load-free.
- Completed models are reported, not re-measured.
- `completed_no_validated_winner` runs are skipped; raising the quality
  threshold and re-running that model deliberately is a separate decision.