# Optimization Strategy Audit (2026-09-23, repository evidence only)

> HISTORICAL as of 2026-09-24: this audit proposed the redesign; the
> implementation now lives in `docs/OPTIMIZATION_STRATEGY.md` (CURRENT
> authority). Pipeline §1–§2 below describe the pre-redesign flow, kept
> for archaeology. Legacy `_stage_*` methods remain in the tree (tested)
> and pre-Phase-A/B checkpoints still resume through them.

No code changed. No models run. All claims below cite the current tree.

## 1. Current optimization pipeline

`AdaptiveOptimizer.optimize()` (`lm_optimizer/services/optimizer.py:76`) runs,
in order: baseline capture → Discovery → Coarse search → Refinement →
Batch optimization (Stage 4) → Micro-refinement (≤6) → Validation →
recompute scores → finalize (`SUCCESS`/`PARTIAL_SUCCESS`/`FAILED`).
`auto` wraps it with smoke → precision probe → context ladder → VRAM matrix
(`cli/main.py`, `auto` command). Interrupt → `CANCELLED`/`INTERRUPTED` with
checkpoint-before-ack; resume skips completed keys
(`resume_from_checkpoint`).

## 2. Current stages (cost-relevant sizes)

- **Baseline**: 1 full benchmark of the as-found config (if loadable).
- **Coarse** (`_generate_coarse_configs`, optimizer.py:556): deterministic
  sampling — ≤4 contexts × ≤3 GPU ratios × flash × KV × ≤2 batches, capped
  at **20** candidates.
- **Refinement**: best + up to 2 Pareto configs × local neighborhood
  (context ±steps, GPU ±0.05–0.15 filtered to space, batch ×0.5–2.0) plus
  flash/KV interaction toggles. No hard cap (typically tens of candidates).
- **Stage 4** (`_generate_stage4_configs`): one-dimension-at-a-time around
  the anchor over batch/physical/parallel/checkpoints (~a dozen at most).
- **Micro** (`_stage_micro_refinement`, max 6): profile-shaped neighborhood,
  never temperature.
- **Validation**: best config re-run **5×** (`validation_repetitions`), median
  kept. CLI flags `--repetitions`, `--validation` (defaults 3/5),
  `--max-tokens-scale` (0.25–1.0).

## 3. Every benchmark type currently executed

| Benchmark | Cost per invocation | Where |
|---|---|---|
| Full suite (`run_benchmark`): 5 cases × (1 preheat + N reps), load+unload | Dominant cost (minutes per candidate on big models) | `services/benchmark.py` |
| `smoke_test`: 1 load + 1×30-token chat + unload | ~1 load cycle | benchmark.py, `fit`, `ctx`, auto |
| `precision_probe`: 3 tests × 2 temps, one load | ~6 short chats + 1 load | auto (`--precision`) |
| `measure_throughput`: 1 load + N concurrent 32-token chats | Bounded; only parallel>1 THROUGHPUT configs | benchmark.py |
| `matrix.run_one`: 1 load + 1×30-token chat | Per cell (ctx×flash×KV) | auto matrix |
| `fit.ladder`: smoke per rung × KV paths + ≤3 refine probes | Bounded ladder | `fit`, auto |
| `ctx` sweep: smoke per geometric point + ≤8 bisect probes | Bounded sweep | `ctx` command |
| `speculative.ab_compare`: opt-in draft vs baseline A/B | Only with `--speculative-draft` | auto |

## 4. Every parameter currently tested

Search space (`services/search_space.py`): `context_length`,
`gpu_ratio` (CLI channel — REST cannot set it), `flash_attention`,
`offload_kv_cache_to_gpu`, `eval_batch_size`, `num_experts` (MoE only),
`physical_batch_size`, `parallel`, `context_checkpoints`.
Sampling/benchmark controls are frozen per case (temperature per test/style,
seed 42); RoPE is experimental opt-in; speculative draft fields are
pass-through, never searched.

## 5. Candidate multiplication (typical balanced run)

20 coarse + ~15–40 refinement + ~10 stage-4 + ≤6 micro + 5 validation ≈
**50–70 full benchmarks** ≈ 250–350 measured test executions + loads.
Observed: qwen3.8-9b run took 8863 s (~2.5 h) for 38 passed configs.

## 6. Where runtime is currently wasted

1. **Quality gate runs last**: every candidate pays the full benchmark
   (loads + 3 reps × 5 cases, incl. the 1024-token context case) BEFORE
   `passes_threshold` is checked (`_test_config`: benchmark → quality →
   score). Slow-but-failing configs cost as much as winners.
2. **Fixed repetitions**: reps=3 even when the first rep already separates
   the candidate from the frontier by 2×.
3. **Validation always 5×** regardless of stability (stability≈1.0 needs 1–2).
4. **Refinement has no hard cap** (unlike coarse=20, micro=6).
5. **Long-tail cases dominate**: at ~1 tok/s, the 1024-token context case
   alone costs ~17 min × reps. `--max-tokens-scale` exists but defaults 1.0.
6. **Reasoning-400 retry**: remembered per (model, value) now, but first
   contact per new value still doubles that request.
7. **Echo probe-load** (smallest model) on every `connect()` unless
   `echo_probe=False` (already used by `status --quick`/`models`).

## 7. Proposed Phase A pipeline (runtime only, context FIXED)

```
fixed ctx (model default or user cap)
  → GPU fit check (estimate + max/full-first; spillover → boundary refine)
  → cheap screening: 1 rep × SHORT cases only (short_instruction,
     structured_output; caps honored) for all coarse candidates
  → speed frontier (gen + prompt + TTFT)
  → top-K by 5% non-inferiority band (profile secondary)
  → FULL suite (all 5 cases × reps) ONLY for band members
  → quality/coding gate on full results
  → stage-4 one-at-a-time + micro (≤6) on survivors
  → validation with adaptive reps (see §11)
```

Context stays fixed through all of Phase A; the context ladder/sweep moves
exclusively to Phase B. Expected effect: 60–80% of candidates eliminated
after ~15–20% of current per-candidate cost.

## 8. Proposed optional Phase B context pipeline

Checkbox `[ ] Optimize Maximum Context` (default off). Input: Phase A winner
config, frozen. Steps: geometric anchors (powers/doubling from current ctx
to model max) → PASS/FAIL bracket → bisect refinement (step 500/1000,
existing `context_sweep.sweep`) → report `min(hardware_stable,
model_limit)` plus performance-optimal and balanced recommendation (already
computed). No GPU/KV/flash/batch/MoE re-search. Reuses current `ctx` command
core; only the UI trigger + frozen-config handoff is new.

## 9. GPU-only vs system-max logic

Phase A must emit both concepts: (a) GPU-ONLY MAX — best config with full
residency if the model fits (estimate + empirical max-load proof), else
explicit NOT FIT; (b) SYSTEM MAX — spillover optimum via CLI-tested ratios
with boundary refinement (0.7 PASS / 1.0 FAIL → 0.85, 0.925, 0.9625…
`refine_gpu_boundary` exists). Scoring must not penalize free VRAM headroom
(current `compute_memory_score` already ties comfortable headrooms at 1.0 —
keep that invariant and add a regression test if changed).

## 10. Speed-first strategy

Order all Phase A decisions by measured speed: screen on decode (+prompt)
rate first; quality/coding validation only inside the frontier band. The 5%
non-inferiority rule (`services/selection.py`) already encodes exactly this
(primary dominates outside the band; profile secondary inside). Keep it as
the single selection authority for both phases.

## 11. Adaptive budget strategy (threshold recommendations)

Base the budget on the first measured speed of the run (baseline smoke or
first coarse candidate), since cost ≈ tokens ÷ tok/s:

| Observed decode | Class | Reps | Validation | Refinement breadth |
|---|---|---|---|---|
| >40 tok/s | FAST | 3 | 5 | full |
| 10–40 tok/s | NORMAL | 2–3 | 3 | full |
| 3–10 tok/s | SLOW | 1–2 | 2 | narrowed (top-1 anchor) |
| <3 tok/s | VERY SLOW | 1 + `--max-tokens-scale 0.5` | 1–2 | finalists only |

Thresholds are cost-derived, not hardcoded constants to worship: with 5
cases × ~300 avg tokens, reps=3 at 1 tok/s ≈ 75 min/candidate vs ~2 min at
40 tok/s. Recommend: implement as defaults computed from the first
measurement, overridable by today's flags. Validation reps scale with
measured instability (`1 - stability`): stable (≥0.98) → 2, else 5.

## 12. Early elimination rules (evidence-only)

- Eliminate on MEASURED speed only: candidate gen rate < 50% of frontier
  best after the cheap screen, with no memory advantage (same/lower VRAM)
  and no quality data yet → drop before full suite. Never eliminate on
  estimates (`lms --estimate-only` prunes escalation only, never proof —
  current `precheck` semantics, keep).
- OOM/structural failures prune their (ctx, gpu) region as today
  (`failed_regions`); INCOMPATIBLE hard-locks the dimension value.
- Keep a small "contrarian" quota (e.g. 1–2 farthest-from-frontier configs
  get full runs) so the frontier cannot blind itself.

## 13. Quality/coding validation strategy

Keep the frozen suite + temperatures; validate only frontier survivors.
Deterministic evaluator per text (no randomness — verified); variance across
identical configs is sampling noise, handled by median + stability score.
`structured_output` (temp 0.0) and `coding_task` (temp 0.1) are the coding-
correctness anchors — always include them in the full-suite subset even
under reduced budgets; drop `long_context` first when scaling tokens (it is
the costliest and least quality-discriminating per run data).

## 14. Checkpoint/resume fit

Unchanged architecture: atomic per-run checkpoints with read-back verify,
resume skips completed keys. Phase split adds: checkpoint records
`phase: A|B` and Phase B resumes from the frozen winner config (no
re-testing of runtime dims). Resume granularity (per-candidate keys) already
supports the cheaper Phase A loop with zero changes.

## 15. Estimated runtime reduction

- Fast models (60+ tok/s): screening saves little per candidate (~30–40%
  total: 2.5 h → ~1.5 h class) — quality gate dominates cost.
- Normal (10–40 tok/s): ~50% (cheap screen + fewer reps).
- Slow (3–10 tok/s): ~60–70% (reps 1–2 + narrowed refinement).
- Very slow (~1 tok/s): order of magnitude (single-rep screening +
  token-scale 0.5 + finalists-only: hours-per-candidate → ~30 min).

## 16. Risks

1. Cheap screen may misrank configs whose speed depends on batch/context
   (screen at fixed small batch underestimates throughput-oriented configs)
   → mitigate with the contrarian quota (§12) and workload-aware screen
   (THROUGHPUT screens with parallel>1 probe).
2. Token scaling weakens the quality gate (truncation fails checks by
   design) → scale is opt-in with warning; never default below 1.0.
3. Skipping validation reps on "stable" configs hides slow drift (thermal)
   → keep minimum 2 reps and the identical-runs typical report.
4. Phase B frozen-config assumption breaks if best runtime dim interacts
   with ctx (KV placement does) → Phase B re-verifies the winner AT the
   recommended context once (single confirmation run, not a search).
5. Early elimination on partial data can always be wrong once; the band +
   quota design bounds the damage to documented, reviewable rules.

## 17. Recommended implementation order

1. Cheap-screen subset + gate reorder inside `_test_config` (biggest win,
   uses existing suite/cases only).
2. Adaptive reps/validation from first measurement + `--max-tokens-scale`
   already exists (wire defaults, no new flags).
3. Phase A context freeze (constrain `min/max_context` to one value through
   the pipeline) + Phase B UI checkbox reusing `ctx` core.
4. GPU-ONLY MAX vs SYSTEM MAX dual reporting (reporting-only).
5. Contrarian quota + elimination rules with tests.
6. Hard cap on refinement breadth.
7. Docs + E2E on one fast and one slow model before touching defaults.
