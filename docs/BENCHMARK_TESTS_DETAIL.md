# Benchmarks in detail: what is tested, how, and why

> CURRENT: since the Phase A/B strategy, a cheap speed probe runs first (ch. 9)
> and the full 5-test suite only on finalists. The 5-test description below
> applies to the QUALITY phase unchanged. Current strategy:
> `docs/OPTIMIZATION_STRATEGY.md`.

Source of truth: `lm_optimizer/benchmark/suite.py`, `lm_optimizer/services/benchmark.py`,
`lm_optimizer/services/quality.py`, `lm_optimizer/services/optimizer.py`.
No running models — code reading only.

## 1. Life of one candidate (exact procedure)

Every tested configuration (`LoadConfiguration`: context, gpu_ratio, flash,
KV on/off, eval/physical batch, parallel, checkpoints, MoE) goes through this:

| # | Step | What happens | Settings |
|---|---|---|---|
| 0 | Host preparation | unload everything, snapshot VRAM/RAM/swap, verify empty server | `prepare_host()` |
| 1 | LOAD | load the model with the given configuration (REST, or `lms load --gpu` with `--gpu-via-cli`) | timeout up to 600 s |
| 2 | VERIFY | echo-load configuration vs. applied (`MATCH/PARTIAL/MISMATCH/UNKNOWN`) | `echo_load_config=true` |
| 3 | PREHEAT | 1× verification + N× warmup chats (results are DISCARDED) | prompt "Say hi in 5 words.", max 30 tok., temp 0.3; only `warmup_time_ms` is measured |
| 4 | MEASURED RUNS | each of the 5 tests × `repetitions` (default 3) | temperatures and max_tokens per test (ch. 5) |
| 5 | UNLOAD | unload + verify empty server | always in `finally` |
| 6 | AGGREGATION | median across repetitions; quality from the most complete sample | median of speeds, stability = 1 − CV |
| 7 | QUALITY GATE | 6 heuristic checks, average ≥ profile threshold (speed 0.95 / balanced 0.97 / quality 0.99), otherwise `QUALITY_FAILED` with no score | `QualityConfig(minimum_score)` |
| 8 | SCORE | weighted sum of normalized components × profile weights | 7 components (ch. 4) |

On load failure: status `LOAD_FAILED`, `score=None`, `quality=None`, kept in history, never eligible to win.

## 2. What changes in each phase

| Phase | Changes | Stays fixed |
|---|---|---|
| Coarse (≤20) | context, gpu_ratio, flash, KV, eval batch | physical/parallel/checkpoints |
| Refinement | winner neighborhood + flash/KV interactions | — |
| Stage 4 | one at a time: eval batch, physical batch, parallel, checkpoints | everything else |
| Micro (≤6) | neighborhood per profile | temperature NEVER |
| Validation (5×) | nothing — the same config repeatedly | everything |
| `ctx` sweep | only context (KV path separately) | runtime |

Temperature is never tuned: it is fixed per test (0.0–0.3, creative exception 0.9) and
the precision probe (0.2 vs default) is informational only.

## 3. What is tracked during a test (every single run)

From `BenchmarkMetrics` + the result envelope:

- `load_time_ms` — the load alone (warmup does not contaminate it)
- `warmup_time_ms` — separate, does not enter the metrics
- `prompt_tokens / completion_tokens / total_tokens` — from the server (`usage`)
- `prompt_tok_s` — prompt_tokens / prompt_processing_ms
- `generation_tok_s` — from the server (`tokens_per_second`), otherwise completion/generation_ms
- `estimated_ttft_ms` — real `time_to_first_token_seconds` from the server, otherwise the 10%-of-total-time heuristic (hence "estimated")
- `prompt_processing_ms / generation_ms` — split of total time
- `peak_vram_gb / peak_ram_gb` — nvidia-smi + psutil before/after (total − min free); `None` = unmeasurable, never estimated
- `output_text` — full answer text (truncated in DB), `error`, `success`
- Events: `LOAD_REQUESTED/SUCCEEDED/FAILED`, `PREHEAT_STARTED` (+ms), `BENCHMARK_*`, `QUALITY_*`, `CONFIG_ELIGIBLE/REJECTED`
- Verification: channel (`REST`/`CLI`), `requested vs applied`, `MATCH/...`

## 4. Speed tests — mechanics (`_run_single_case`)

1. `perf_counter()` stopwatch → `POST /api/v1/chat` (test prompt, its temperature, its `max_tokens`, `reasoning="off"`).
2. Empty output + reasoning off → **exactly one** retry with `reasoning="on"` (some models do not answer without thinking); still empty → `success=False, "Empty output"` (does not corrupt statistics with absurd tok/s).
3. Time is split: if the server gave `tokens_per_second`, `generation_ms = completion / tok_s`, the rest is prompt; otherwise proportional split by tokens.
4. TTFT: real number from the server, otherwise 10% of time (marked estimated).
5. Aggregation: **median** across repetitions (outlier-robust); stability = `1 − stdev/mean` of speeds; text for quality = the **most complete** sample (not median — one truncated run must not kill a good config).

## 5. Accuracy tests — mechanics (`QualityEvaluator`)

Six dimensions 0.0–1.0, `overall` = average; **check = dimension ≥ 0.9**
(`28/30` = 28 passed checks out of 30). Aggregate = average across tests,
`sum of passed / sum of total`. The evaluator is deterministic (no randomness);
run-to-run variability is model sampling, not measurement error.

**`structured_output` (format):** strips ``` fences; invalid JSON → immediately
`overall=0.0`. Otherwise: required keys `name/age/skills/address` (missing −0.25
each), types (`age` int, `skills` list, `address` dict, else 0.5), ends with
`}` (else 0.5).

**`coding_task` (coding):** strips fences; `def find_duplicates` in code
(0/1); mention of `O(n)/O(1)/linear/constant` (1.0/0.7); no `class /`
`if __name__` (1.0/0.7); proper ending (1.0/0.5); `coding_correctness` =
`task_completion`.

**General (`instruction/reasoning/context`):** length vs `min_tokens`
(ratio, max 1.0); reasoning: must contain something from
`difference/pattern/add/sequence/72` (else 0.5); context: keywords
`solar/wind/hydro/geothermal/biomass`, score `min(1.0, found/5*1.5)`;
instruction factual always 1.0. Proper ending: last character after
stripping markdown in `. ! ? ) " ' ] }` (else 0.5). **Repetition:** word
2–4-grams (without `*` `` ` `` `# _ ~ > |`); if unique < 70% → `no_malformed = 0.0`.

## 6. Exact prompts and settings (verbatim from `suite.py`)

Styles change only temperatures: precise `{coding: 0.1, reasoning: 0.1}`,
creative `{context: 0.9}`, balanced unchanged. `max_tokens` scales with
`--max-tokens-scale` (min 32) and is capped at `context_length // 4`.
Honestly: `seed = 42` is only recorded in `benchmark_params`, REST rejects
it, so it is never sent — determinism rests on fixed low temperatures.
Likewise `stop_sequences` are defined in the suite, but the server rejects
`stop`, so they are never sent and fences are stripped in the evaluator.

**T1 `short_instruction` [instruction]** — temp 0.3, max 256:
> Write a concise explanation of how a hash table works in 3-4 sentences.
Purpose: basic instruction + fluency; cheap (short).

**T2 `medium_reasoning` [reasoning]** — temp 0.3, max 768:
> You are given a sequence: 2, 6, 12, 20, 30, 42, 56. What is the next number
> in the sequence? Explain your reasoning step by step.
Purpose: reasoning chain + correct answer (72); keywords above.

**T3 `long_context` [context]** — temp 0.3, max 1024:
> Below is a document about renewable energy. Please read it carefully and
> answer the question at the end. DOCUMENT: [~350-word document on solar,
> wind, hydro, geothermal energy and biomass + storage, grid, policies,
> costs, trends] QUESTION: Summarize the main renewable energy sources,
> their key advantages, primary challenges, and two future trends mentioned
> in the document.
Purpose: long-input understanding + keyword coverage; most expensive test
(1024 tokens) — ~17 min/repetition on 1 tok/s models.

**T4 `coding_task` [coding]** — temp 0.1, max 768, stop `["```", "def ", "class "]`:
> Write a Python function `find_duplicates(nums: list[int]) -> list[int]`
> that returns all duplicate integers in a list. The function should:
> 1. Run in O(n) time complexity
> 2. Use O(1) extra space (excluding output)
> 3. Handle negative numbers
> 4. Return duplicates in ascending order
> Provide only the function definition with docstring.
Purpose: code correctness; low temperature = determinism. (The server
rejects `stop`, so fences are stripped in the evaluator.)

**T5 `structured_output` [format]** — temp 0.0, max 256, stop `["}"]`:
> Output a JSON object with the following structure exactly:
> { "name": "string", "age": integer, "skills": ["string", "string", "string"],
>   "address": { "city": "string", "country": "string" } }
> Use realistic data for a software engineer. No extra text, just the JSON.
Purpose: strict format; temperature 0.0 = maximum determinism.

**Precision probe** (`auto --precision` only): T4+T2+`structured_output` at
suite temperature vs 0.2, one load — informational, does not enter the score.

## 7. Other runs (settings)

- **smoke_test**: load + 1 chat ("Say hi in 5 words.", 30 tok., 0.3) + unload.
- **measure_throughput**: 1 load + N concurrent 32-token chats
  (`asyncio.gather`); aggregate = tokens / wall-clock.
- **matrix.run_one**: 1 load + 1×30-token chat (ctx×flash×KV cell).
- **fit ladder / ctx sweep**: smoke per point + `ctx` bisect refinement (step
  500/1000, max 8 probes).
- **speculative.ab_compare**: opt-in, draft vs baseline on the winning configuration.

## 8. Cost model and levers for slow models

Candidate cost ≈ load + preheat + `repetitions` × Σ(max_tokens) / tok_s.
At 1 tok/s: T3 alone ~17 min × 3 repetitions ≈ 1 h/candidate.
Levers: `--repetitions 1 --validation 1` (3–5× down),
`--max-tokens-scale 0.5|0.25` (with truncation-failure risk, warning is
printed), narrower `--min/max-context`, `--dry-run` shows the candidate count upfront.
The report (`-best.md`) contains every table above + per-test rows with temperature,
tokens, speeds, and quality.

## 9. Speed probe (Phase A) — what the 5-test suite does NOT run against

- One prompt: "Explain in one or two short sentences what a hash table is.",
  max 64 tokens, temperature 0.1, reasoning off, minimal preheat.
- No quality evaluation. The result carries `generation.phase = "SPEED"`,
  score stays None until quality passes.
- Repetitions by budget class from the first (S0) measurement: FAST 3, NORMAL 2,
  SLOW 1, VERY_SLOW 1. Slow models are never rejected.
- Context: `speed_context` (user cap or 2048) for all probes;
  `quality_context` (user cap or min(limit, 8192)) for finalists.
  Context does not enter the Phase-A score (weight 0, rest renormalized).
- Throughput probes (parallel) measure the aggregate into a separate
  `generation.throughput` block; the primary metric stays single-stream tok/s.
