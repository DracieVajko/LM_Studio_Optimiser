# Optimization Strategy (CURRENT, implemented)

> Authority for the CURRENT pipeline. Historical proposal:
> `docs/OPTIMIZATION_STRATEGY_AUDIT.md` (superseded where it differs).
> Benchmark procedure detail: `docs/BENCHMARK_TESTS_DETAIL.md`.
> LM Studio capability source: registry (`parameter_registry.py`) +
> `docs/LM_STUDIO_PARAMETER_MATRIX.md`.

Principle: **first find what makes this model fast, then verify which fast
configurations are correct. If the fastest is wrong, roll back the most
likely causal setting. Only after the runtime is verified may optional
max-context optimization run.**

```
PREPARE → SPEED (fixed small ctx) → FRONTIER → QUALITY (frozen finalists)
  → RECOVERY (bounded rollback) → FINAL VALIDATION → [optional] CONTEXT
```

## Phase A — runtime (context frozen)

- `speed_context` = user `--max-context` cap if configured, else **2048**
  (clamped to the model limit). `quality_context` = user cap if configured,
  else `min(model limit, 8192)`, never below speed context.
- Every speed candidate uses the same context. Context never enters the
  Phase-A score (context weight forced to 0, other weights renormalized).

## Speed probe (cheap by design)

- One short prompt ("Explain in one or two short sentences what a hash
  table is."), max 64 tokens, temp 0.1, reasoning off, minimal preheat.
- NEVER the full 5-test suite. Records load/warmup ms, prompt/completion
  tokens, generation + prompt tok/s, TTFT, peak VRAM/RAM, channel,
  requested/applied, success/failure. No quality score.
- Adaptive budget from the first (S0) measurement — budget only, NEVER
  rejection: FAST (>40 tok/s): 3 reps, ≤5 finalists; NORMAL (10–40): 2/4;
  SLOW (3–10): 1/3; VERY_SLOW (<3): 1/2. A 0.5 tok/s model fully completes.
  User `--repetitions` still governs the full suite/validation.

## Staged search (no Cartesian product, ≤25 probes)

- **S0** baseline anchor (as-loaded config, else server defaults) at the
  frozen speed context. Classifies the budget.
- **S1** high-impact levers, one at a time: GPU ratio (CLI only when
  enabled), KV placement, Flash toggle, MoE experts (MoE only),
  speculative/MTP (only with a discovered draft model).
- **S2** batch around the anchor: eval/physical batch, checkpoints.
- **S3** concurrency (parallel 1/2): aggregate throughput recorded in a
  SEPARATE block; single-stream generation stays the primary metric.
- **S4** load/memory (mmap, keep-in-memory, threads): MANUAL_ONLY — reported
  as guidance, never auto-probed, never claimed as tuned.
- **S5** interactions of the strongest settings only (≤2 combos).
- Frontier: 5% non-inferiority band (capped by budget) + one contrarian
  (most different runtime strategy out of band).

## Quality + causal recovery

- Only finalists run the full 5-test suite (existing prompts, temps,
  evaluator — unchanged) at the frozen quality context.
- Raw fastest is tracked even when it fails; the winner is the fastest
  quality-passing config. Failed loads never score, never win.
- On quality failure: (1) reconfirm ONLY failed tests (noise vs
  deterministic); a reconfirm-pass is admitted through the full suite
  (noisy configs are never silently dropped); (2) roll back ONE suspect
  in tier order (HIGH: speculative/MTP, experts, KV representation;
  MEDIUM: placement, flash, checkpoints; LOW: batches, threads, mmap) —
  see `services/semantic_risk.py`; (3) admit via full suite on recovery;
  (4) bounded pairwise only for ≤3-key deltas. Default budget: 3 probes.
- Rollback reference priority: fastest safe config ("safe",
  quality-verified) first, measured S0 anchor ("baseline", NOT
  quality-verified) as fallback; the kind is recorded in the recovery log.
- Failed-test selection (`rank_failed_tests`) is severity-ordered and
  None-safe: invalid/unavailable evidence first, never compared numerically.
  Invalid JSON is an explicit 0/6 failure (overall 0.0), never null checks.
- Reports show RAW FASTEST vs QUALITY-SAFE vs RECOVERED + likely culprit.
- Winner architecture: SPEED probes stay scoreless (recompute skips them);
  FINAL RECOMMENDATION = run's validated best, fallback = quality-evidenced
  passed configs. CLI and `-best.md` agree by construction.
- Measurements carry phase tags (speed/quality/recovery/final_validation);
  validation stability uses same-kind measurements only.
- Displayed score breakdowns show the actual recorded phase weights
  (Phase A: context = 0), also exposed via API `score_weights`.
- Containment: one crashing probe is logged and skipped (phase continues);
  evaluator crashes fail the recheck, never recovery; an optional Phase B
  sweep crash is contained and reported, never fatal to the run.
- Resume: completed probes/quality tests are never re-run; budget class,
  repetitions, baseline anchor and recovery log survive via checkpoint;
  `revalidate` clears all Phase A/B progress; pre-Phase-A/B runs resume
  through the legacy stage path.

## Final validation + Phase B

- Adaptive reps from stability (≥0.95 → 2, ≥0.85 → 3, else up to 5;
  minimum 2), median wins, existing median/stability semantics kept.
- Phase B is OPT-IN (`--optimize-context` / `optimize_context`, default
  OFF): frozen winner, only `context_length` changes, geometric → bisect
  sweep. Reports GPU-ONLY MAX vs SYSTEM MAX, model limit vs hardware
  stable limit, final capacity = min(...), plus performance-optimal and
  balanced contexts.

## Cost comparison (estimate, labeled)

Old flow per candidate: full suite (5 tests × up to 256–1024 tokens × 3
reps ≈ 250–350 measured generations incl. the 1024-token context test) —
for ~40 candidates ≈ 10k+ heavy generations plus loads.

New flow: ~16 probes × (1–3 reps × ≤64 tokens) ≈ under 3k light tokens of
generation for the whole speed phase, then ≤5 finalists × full suite, then
≤3 recovery subsets + ≤5 validations. Slow models concentrate ~90% of
tokens on finalists instead of spreading the full suite over ~40
candidates. Percentages are estimates from probe/budget arithmetic, not
measurements; a real-model E2E is still pending user confirmation.

## What is NOT auto-tuned (capability-gated)

REST: context_length, flash_attention, offload_kv_cache_to_gpu,
eval_batch_size, physical_batch_size, parallel, context_checkpoints,
num_experts (MoE), speculative draft keys (only with draft model).
CLI: gpu_ratio (when `--gpu-via-cli` and `lms` available).
MANUAL_ONLY (reported, never probed): KV quantization, n_threads,
try_mmap, keep_model_in_memory, unified_kv_cache.
DETECTED_ONLY/unsupported: never claimed as tuned.
