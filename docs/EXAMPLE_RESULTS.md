# Example results (reference run: 2026-10-25/26)

These are real outputs from this repository on the reference machine
(RTX 3060 Laptop 6 GB, 31 GB RAM, LM Studio with KV cache + 8 threads).
They are examples of what the tooling produces, not marketing claims.

## 1. Test suite

```
$ python -m pytest tests/ lm_optimizer/tests/ -q
1315 passed, 1 warning in 131.49s
```

Static checks:

```
$ python -m compileall -q lm_optimizer      # clean
$ ruff check lm_optimizer/services/optimizer.py ... --select E9,F63,F7,F82,F821
All checks passed!
```

## 2. Speed @16K vs @4096 (median generation tok/s)

Measured with `--preset quick --speed-only --context <N>`.
Different LM Studio settings between the two columns: the 4096 column has
KV cache + 8 threads enabled, so compare rows within a column, not across.

| Model | @16K | @4096 |
|---|---:|---:|
| mirothinker 30B | 14.82 | 18.28 |
| qwen coder 30B | 12.51 | 12.32 |
| gpt-oss-20b heretic | 3.50 | 8.38–10.50 |
| gemma-4-26b | 12.31 | 10.45 |
| qwen3.6-35b MoE | 9.87 | 9.84 |
| bonsai 27B | 11.00 | 9.14 |
| swift q2_k | 3.33 | 3.32 |
| swift q4_k_s | 2.25 | 2.71 |
| swift q3_k_m | 1.92 | 2.01 |
| qwen3.8-9b | 11.63 | 6.49–8.78 |
| flint 40B | load failed | 8.91 (11/12 passed) |

## 3. Reasoning sweep (5 levels, 8K context)

`--reasoning-only --max-context 8192`; generation tok/s per level.

| Level | swift q2_k | swift q4_k_s |
|---|---:|---:|
| off | 3.42 | 2.58 |
| low | 3.41 | 2.54 |
| medium | 3.37 | 2.56 |
| high | 3.34 | 2.36 |
| xhigh | 3.29 | 2.22 |

Note: these runs predate the 2026-10-26 quality fix, so their stored
`quality=1.000` had `0/0 checks` and must not be read as correctness evidence.
Re-run with the current code to get real checks.

## 4. Implementation verification (persisted run invariants)

```
$ python tools/verify_runs.py --since 2026-10-25T03:44:00
[PASS] 5fb0a2d9 prism-ml/bonsai-27b    completed_no_validated_winner
[PASS] d49f51ac prism-ml/bonsai-27b    completed_no_validated_winner
16/16 checks passed, 0 failed
```

Checked per run: `resolved_settings`, preset match, `speed_floor` block,
probe cap, stage skips, explicit no-winner path, positive contexts.

## 5. Speed-floor edge cases (measured, bonsai)

| Case | Flags | Result |
|---|---|---|
| Floor enforced | `--min-generation-tok-s 999` | `NO_CONFIG_MEETS_SPEED_FLOOR`, fastest candidate kept as diagnostic |
| Floor disabled | `--min-generation-tok-s 999 --no-enforce-min-generation-speed` | `enforced=false`, no forced no-winner reason |
| Custom stage + cap | `--preset custom --stage speed --max-speed-probes 5` | exactly 5 probes, `selected_stages=["speed"]` |

## 6. Known reporting caveats

- Speed-only runs save a "NO WINNER" report (no quality evidence) named
  `*-failed-*.md`. That is intentional honesty, but the filename is easy to
  misread.
- The heuristic quality score is not an intelligence benchmark. Only trust it
  when `checks_total > 0`.
