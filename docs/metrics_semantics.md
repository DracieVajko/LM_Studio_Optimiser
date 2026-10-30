# Metric semantics & winner/Pareto consistency (canonical definitions)

Authoritative reference for every speed/quality number the optimizer exports.

## Aggregation pipeline (generation speed)

| Stage | Aggregation | Code |
| --- | --- | --- |
| Single measurement | tokens / elapsed wall time | `BenchmarkRunner._run_single_case` |
| Repetitions of one test | **median** across repetitions | `runner.py::_aggregate_results` (comment R33) |
| Across tests for a config | **median** across per-test medians | `BenchmarkRunner.get_median_generation_tok_s` |
| Arithmetic mean across tests | explicit, separately named | `BenchmarkRunner.get_mean_generation_tok_s` |

- `get_avg_generation_tok_s` is a **historical alias for the median** (kept for
  compatibility). It is NOT the arithmetic mean.
- Markdown reports headline the **median** (`- Generation: X tok/s`).
- External ad-hoc exports (e.g. the older night JSON queue summaries) used the
  **arithmetic mean over tests**. That is why a winner showed
  `12.691` in JSON but `11.3` in Markdown: per-test medians were
  (10.6, 10.3, 11.3, 11.4, 19.9); median = 11.3, mean = 12.691. The 19.9 tok/s
  test is short structured output — fast by design, so the mean is skewed.
  Future exports must label the fields distinctly, e.g.
  `gen_tok_s_median_over_tests` vs `gen_tok_s_mean_over_tests`.

## Quality

- Per output: 6 deterministic checks; per-test overall = mean of checks.
- Config aggregate: **mean over tests** (shown as X/30).
- Eligibility gate: `aggregate quality >= profile threshold`
  (speed 0.95, balanced 0.97, context/quality higher). Inclusive at the
  boundary (Gemma hitting exactly 0.97 passes).
- Quality applies to the most complete repetition sample (R33).

## Winner selection (notifiable rules)

- Progressive during the search: `select_by_score` (`services/selection.py`)
  with a 5% non-inferiority band inside recomputed-at-arrival bounds.
- After the search, final validation re-runs the winner's config several times
  and the **winner record becomes the median-speed validated copy**
  (`_phase_final_validation` → `_median_result(validated)`).
- Final score recompute (`_recompute_all_scores`) may rank a different
  eligible configuration higher. This is surfaced (not silently hidden):
  the report prints a NOTE whenever an eligible non-winner closes with a
  score > 1.05× the winner's. Winner status stays with the validated config.

## Pareto eligibility

- Same policy as winner candidacy: PASSED + quality-validated + scored
  (`_score_eligible` only scores configs that passed the quality gate).
- Frontier rule: multi-objective non-domination on
  (quality, gen tok/s, prompt tok/s, context, peak VRAM) —
  `optimizer._compute_pareto_frontier`.
- A candidate with a higher *recomputed* final score may legally remain
  off-winner (see section above); it stays on the frontier so it is visible.

## Case traces (2026-10-18 night queue)

- qwen3.8-9b-distill: winner a4bd..., JSON winner gen 12.691 (mean) vs
  Markdown headline 11.3 (median) — see aggregation table above.
- mirothinker-30b: winner score 0.6754 vs Pareto candidate bb72a640 score
  0.8363 @ ~20.9 tok/s (quality both ≥ 0.97). Winner was validated and is the
  median validated copy of its config; the higher-closing-score candidate
  remains on the Pareto frontier (documented NOW in reports via NOTE).
