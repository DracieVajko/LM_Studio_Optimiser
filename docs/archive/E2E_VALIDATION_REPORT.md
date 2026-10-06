# E2E Validation Report — real LM Studio installation (2026-09-23)

Environment: Windows, AMD 8P/16L, 15.4 GB RAM, NVIDIA RTX 3060 Laptop 6.0 GB.
Server `http://127.0.0.1:1234` (v1, connected). `lms` CLI present.
Note: `openai/gpt-oss-20b` was pre-loaded at session start; `prepare_host`
unloaded it before instrumented runs (documented side effect, restorable).

Unit state: 199 collected / 199 passed (2 pre-existing third-party warnings).

## §1 GPU CLI control (qwen3.5-4b, ctx 2048, 1 rep)

Commands:
- `lms load --estimate-only qwen3.5-4b` -> 3.15/3.15 GiB, LOW, rc=0
- `lms load qwen3.5-4b --gpu 0.8 -y` -> "Model loaded successfully in 7.90s",
  identifier `qwen3.5-4b`, 3.15 GiB
- `BenchmarkService(gpu_via_cli=True).run_benchmark` at 1.0 / 0.9 / 0.8

| Requested GPU | Actual applied | Load | Gen tok/s | Prompt tok/s | Channel | Verification |
|---|---|---|---|---|---|---|
| 1.0 | max | PASS | 68.4 | 469 | CLI | PARTIAL_MATCH (REST keys unverified, listed) |
| 0.9 | 0.9 | PASS | 38.4 | 336 | CLI | PARTIAL_MATCH |
| 0.8 | 0.8 | PASS | 23.6 | 276 | CLI | PARTIAL_MATCH |

`lms load --gpu` is genuinely invoked (server-side load, not a default
echo). Speed is monotonic in residency — full GPU first is correct here.

## §2 Requested vs applied

Every candidate above records `generation["load_channel"]` +
`generation["load_verification"]` (MATCH/PARTIAL/MISMATCH/UNKNOWN with diffs).
CLI-applied dicts only attest gpu_ratio+context; REST keys show as diffs
instead of false "verified". `load_with_channel` falls back CLI->REST.

## §3 Context refinement (qwen3.5-4b, real server)

- 8192/16384/32768/65536/131072/262144: all PASS (spillover works).
- Forced bracket 262144 PASS (60.3) / 1000000 FAIL (~10 s refusal):
  refinement probed strictly inside —
  762000 F, 643000 P (23.1), 702000 F, 672000 P (22.4), 687000 F,
  679000 P (23.3), 683000 F, 681000 F.
- Maximum stable: **679000**. Failed boundary: **681000** (2000-token bracket).
- Performance-optimal: 262144 (60.3). Balanced: 262144. Capacity,
  optimum and recommendation are three different measured answers.

## §4 VRAM-first

Small fitting model: 68.4 (max) > 38.4 (0.9) > 23.6 (0.8) — full residency
first, never preferring lower ratio for free VRAM. Search space samples 1.0.

## §5 Large-model bounded path (ministral-3-14b-reasoning, 9.12 GB)

- Estimate @32768/max: 12.63 GiB (spillover expected), rc=0.
- REST smoke @8192: PASS 5.4 tok/s (61 s load).
- `lms load --gpu max`: 9 s (warm), REST chat 5.37 tok/s, unload verified.
- ctx 8192/16384/32768/65536: all PASS @5.3-5.4 (spillover, no cliff).
- Reasoning-400 retry path exercised live. No full search (cost documented).

## §6 Failure semantics (live)

`ctx=1000000` refusal through `optimizer._test_config`:
`status=load_failed`, `score=None`, `quality=None`, no best, empty tested
list, `LOAD_FAILED` event present, `CONFIG_ELIGIBLE` absent. CLI benchmark
shows Failed with zeroed metrics and no quality.

## §7 Report (`results/qwen3.5-4b-best.md`, tiny real optimize)

Run SUCCESS, best ctx 2048 score 0.411, preset saved, capability snapshot #8.
Contains: exact model_id + run_id, display/quant/arch identity, backend
(version `v1`), GPU channel per candidate, requested==applied MATCH (REST),
contexts trio, score breakdown, crowns, failures table, storage/swap,
control channels. N/A appears only with explicit retained-default reasons —
no contradictions. No baseline section: none existed (`prepare_host`
requires an empty server, so CLI runs structurally have no baseline —
known limitation, reported as "N/A (no baseline)", never fabricated).

## §8 User pipeline

Web UI served live: `/` 200, `/api/status` 200 (`connected: True`),
`/api/runs` 200 (3 runs incl. E2E runs). Routes and CLI both construct
`AdaptiveOptimizer` + `BenchmarkService` (asserted in
`test_optimization_uses_same_services`). `start_optimizer.bat` verified by
inspection + static test (venv bootstrap, menu, no bulk auto-runs; interactive
menu not executed here).

## Failures encountered (all handled, none hidden)

- `ctx` max clamps to model limit (300000 never probed) — by design; bracket
  forced via service call instead.
- 300000 unexpectedly PASSED (server clamps internally) — bracket moved to
  1000000; server behavior documented, not assumed.
- Estimate confidence often LOW/None — estimates recorded, never used as proof.
- 14B reasoning-400 on `reasoning` key — existing retry path handled it.

## Fixes still required / known limitations

1. OOM (true VRAM exhaustion) is nearly untriggerable on this server —
   spillover absorbs it; OOM paths remain mock-tested only.
2. CLI runs structurally lack baselines (`prepare_host` empties the server);
   baseline comparison needs a loaded-at-start flow or explicit baseline run.
3. No full multi-config optimize was run E2E (cost); pipeline mechanics proven
   by the tiny run + unit coverage.
4. `gpt-oss-20b` pre-loaded state was displaced by E2E loads (reload manually
   if needed).
5. No commit/push performed (not a git repository).
