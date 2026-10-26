# Full test policy + scheduled artifact test (decision 2026-10-26)

## Decision

**Full test is coverage-complete, not time-bounded.** Slower hardware takes
longer; it must not lose stages, tests, or repetitions. No time budget, no
silent downgrades. Everything skipped is explicit and reported.

Sandbox stays a separate manual feature. Duel/compare stays separate. A full
run may end with one user-preselected **artifact test** (web, 3D, research,
custom) executed against the validated winner.

## What "Full" guarantees

| Stage | Full behaviour | Cannot be shortened by |
|---|---|---|
| Preflight | host guard, model resolve, HW detect, baseline | — |
| SPEED | staged plan S0–S5, cap `max_speed_probes` (25 standard) | wall-clock limits |
| QUALITY | full 5-category suite on every finalist (up to 4), 3 reps | "acceptance subset" shortcuts |
| RECOVERY | up to 3 causal rollback probes on quality failures | time pressure |
| FINAL_VALIDATION | adaptive 2–5 independent repetitions | one-rep claims |
| Artifact test (optional) | one selected task on the exact winner config | — |
| Context sweep | opt-in (`--optimize-context`), strategies per preset | — |
| Sandbox lab | never part of full; separate manual area | — |
| Duel/compare | never part of full; separate feature | — |

## Allowed optimizations (no coverage loss)

1. **Quality context default 8192** (instead of inheriting `--max-context
   32768`). Same tests, same reps — cheaper context only. Explicit
   `--max-context` still wins.
2. **Warm validation**: reuse the loaded winner config for final validation
   instead of unloading and reloading it.
3. **Skip-plan gating** for optional phases (context strategies, artifact
   test) — already enforced in `_finalize` since 2026-10-26.
4. **Fingerprint cache**: identical model/runtime/hardware/benchmark evidence
   is reused on later runs; stale evidence is re-measured.
5. **Resume**: interrupted full runs continue from checkpoint, never restart.
6. **Probe dedup**: identical load configs are never re-run (candidate key).

## Forbidden shortcuts in Full

- Dropping test categories or repetitions to save time.
- Time-budget planner that silently ends the campaign early.
- Treating a scoreless speed probe as a validated winner.
- Reporting `0/0 checks` as quality evidence.
- Folding the sandbox/duel into the full report as if they were optimizer
  evidence.

## Runtime expectations (reference HW: 6 GB VRAM, 31 GB RAM)

Measured on this machine; Full is expected to be slow here and fast on a
larger GPU. No cap is applied.

| Model class | Speed probes (25) | Quality (4 finalists × 5 × 3) | Validation | Total |
|---|---|---|---|---|
| 8–9B | 10–20 min | 20–40 min | 5–10 min | ~1 h |
| 20–30B (partial spill) | 40–90 min | 1.5–3 h | 10–20 min | 2.5–4.5 h |
| 27B at 2–3 tok/s | 60–120 min | 3–5 h | 20–40 min | 5–8 h |
| 40B (4K only) | 60–120 min | 3–6 h | 30–60 min | 5–9 h |

## Fourth option: scheduled artifact test

Purpose: prove the winner can produce a real artifact, and make that result
comparable across models in history.

### UX
- `--artifact-test <task-id>` on `optimize` (and the API field
  `artifact_test_id`).
- Built-ins come from `SANDBOX_TASKS` (single source of truth):
  `web_dashboard`, `blender_skyblock`, `research_synthesis`.
- `--artifact-test custom:<path.json>` for a validated custom task.
- Defaults: **off** for quick, **off unless chosen** for full, **all built-ins**
  only for thorough with explicit opt-in.
- Quick/full/thorough still show the resolved choice in the pre-flight summary.

### Execution
- Runs after FINAL_VALIDATION, against the exact winner `LoadConfiguration`.
- Loads through the verified client, records requested vs applied config and
  echo mismatches.
- Sequential, one task; unloaded in `finally`; artifacts isolated per run.
- Records: task id + version/hash, config fingerprint, duration, success,
  score (only when checks exist), artifact paths, unload status.

### Persistence and comparison
- Stored in `run.benchmark_params["artifact_test"]` (JSON; no migration).
- API `OptimizationRunResponse.artifact_test` exposes a compact summary.
- History list shows a badge (`artifact: web ✓ 0.86`).
- A comparison query can filter runs by `artifact_test.task_id`, so
  q2_k vs q4_k_s vs coder-30b can be compared on the same artifact task.
- Sandbox lab results remain separate and are never mixed into this field.

### Relationship to existing code
- Reuses `SandboxTask` definitions and `run_sandbox_task()`.
- `final_sandbox_mode` remains the low-level mechanism; `artifact_test_id` is
  the user-facing selector that maps to `mode="one"` + that preset id.
- No new evaluator: scoring stays in `sandbox_lab.score_attempt`.

## Implementation sketch

1. `OptimizationRun.artifact_test_id: str | None` + `AdvancedSettingsSchema`
   field + CLI option.
2. `prepare_run`: resolve/validate id (fail fast on unknown, no silent off),
   freeze it in `resolved_settings`.
3. `_finalize`: if set and `artifact_test` not skipped, call
   `_run_final_sandbox` with that one task and write
   `benchmark_params["artifact_test"]` (compact summary).
4. API/history: expose the summary; add filter by task id.
5. Tests: selector validation, unknown id fails, exact-config load, score only
   with checks, history badge/filter, sandbox unaffected.

## Non-goals

- No time budget for Full (explicit decision).
- No automatic task choice; the user picks the artifact test before start.
- No mixing duel/compare outcomes into optimizer scores.
