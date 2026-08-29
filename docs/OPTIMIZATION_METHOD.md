# LM Studio Auto Optimizer — Optimization Method (v0.1.1)

This document explains the adaptive optimization method after the **v0.1.1 Optimization Correctness Pass**. It replaces hardware-specific heuristics with hardware-agnostic, run-relative scoring.

---

## 1. Overview

The optimizer finds the best **model load configuration** for your actual hardware and model by benchmarking feasible candidates and scoring them according to a profile (Speed / Balanced / Context / Quality / Custom).

**Load parameters tuned** (depending on LM Studio capabilities):
- `context_length`
- `gpu_ratio` (fraction offloaded to GPU)
- `flash_attention` (bool)
- `offload_kv_cache_to_gpu` (bool)
- `eval_batch_size` (int)
- `num_experts` (MoE only)
- `rope_freq_base` / `rope_freq_scale` — **experimental**, disabled by default

The optimizer never assumes a fixed GPU (e.g. RTX 3060), fixed VRAM (6 GB), or fixed speed numbers. All scoring is **hardware-agnostic** or **run-relative**.

---

## 2. Candidate Generation

### 2.1 Search Space

`lm_optimizer/services/search_space.py:48`

A `SearchSpace` is generated from:
- **Model**: `context_limit`, `parameter_count`, `quantization`, `is_moe`
- **Hardware**: `GPU VRAM`, `system RAM`, `GPU count`
- **LM Studio capabilities**: which load parameters are actually supported (via `LMStudioClient.capabilities`)
- **Advanced settings**: user overrides for `min_context`, `max_context`, `min_gpu_ratio`, etc.

**Remaining constants — benchmark-specific, documented:**

- **Context candidates**: `[2048, 4096, 8192, 12288, 16384, 24576, 32768, 65536, 131072]`  
  *Reason*: standard powers-of-two and common LLM context sizes. Filtered to `[min_context, max_context]` and always includes `model_max_context`. Not used for scoring, only as test points.

- **Batch sizes**: `[64, 128, 256, 512, 1024]`  
  *Reason*: powers of two covering typical `eval_batch_size` range. Used when `auto_batch=True`; otherwise user-defined range.

- **GPU ratio steps**: `0.1` default, `0.05` for refinement  
  *Reason*: granularity for offload ratio sweep, hardware-relative (filtered to `[min_gpu_ratio, max_gpu_ratio]`).

- **VRAM headroom threshold**: `0.15` (15%)  
  *Reason*: memory scoring (see §4). Not hardware-specific; a relative fraction.

- **Epsilon for zero-range**: `1e-9`  
  *Reason*: numerical stability when all candidates have identical metric.

All other former scoring constants (`50 tok/s`, `1000 prompt tok/s`, `2000 ms TTFT`, `32768 context`, `6 GB VRAM`) have been **removed**.

### 2.2 Coarse Search — Deterministic Intelligent Sampling

`lm_optimizer/services/optimizer.py:185`

The naive Cartesian product could be >500 combos (e.g. 5 contexts × 5 GPU ratios × 2 flash × 2 KV × 5 batches). The optimizer samples **20** representative candidates:

1. **Deterministic sampling per dimension** (`_deterministic_sample`): for each list, picks **evenly spaced** values covering low/high (e.g. for 5 contexts, picks indices `0, 2, 4` → low, mid, high). Sorted, no randomness.
2. **Cartesian product** in sorted order `(context, gpu, batch, flash, kv)`.
3. **Evenly spaced global sampling**: if `>20` combos, sorts all configs deterministically and picks every `N/20`-th element, ensuring coverage of:
   - low / high context
   - low / high GPU ratio
   - KV alternatives (GPU vs CPU)
   - Flash alternatives (ON vs OFF)
   - batch alternatives

Results are **reproducible**: same hardware + model + LM Studio caps + advanced settings → same candidate list (fixed seed `42` for generation, no RNG without seed).

### 2.3 Refinement — Adaptive, Based on Measured Results

`lm_optimizer/services/optimizer.py:285`

Refinement is not blind; it uses measured data:

- **Promising candidates**: the current best + top-2 Pareto frontier configs (not just best), to avoid local optima.
- **Context refinement**: `± step` where `step = max(512, best_ctx/4)` plus nearest neighbors in the sorted search space, filtered to search space bounds. Model-relative, not hardcoded 32768.
- **GPU refinement**: `±0.05, ±0.10, ±0.15` around best, filtered to valid ratios from search space.
- **Batch refinement**: `×0.5, ×0.75, ×1.5, ×2.0` around best, rounded to nearest 32, capped `32–2048`, filtered to search space.
- **Interaction tests**: for each promising candidate, explicitly toggles `flash_attention` and `kv_cache` at the best `(ctx, gpu, batch)` to test important parameter interactions.
- **Avoids inferior regions**: tracks `failed_regions` (OOM / failed `(ctx, gpu)` pairs) and skips refining them; skips already-tested `(ctx, gpu, batch)` keys.

Batch optimization (Stage 4) separately sweeps all batch sizes for the best config, skipping already tested.

### 2.4 RoPE — Experimental

`rope_freq_base` / `rope_freq_scale` are **DISABLED by default**.

- To enable: `advanced_settings.enable_rope = true`
- When enabled:
  - Run is flagged `is_experimental = true` with reason `RoPE parameters enabled (experimental)`
  - Quality threshold is raised to `max(threshold, 0.98)` (stronger validation)
  - A faster RoPE config is **never** claimed better unless quality passes the stricter threshold
  - UI and reports label the run “Experimental”

This prevents accidental inclusion of poorly validated RoPE configs in normal searches.

---

## 3. Benchmarking — Determinism

### 3.1 Fixed Suite

`lm_optimizer/services/benchmark.py:22` and `lm_optimizer/benchmark/suite.py`

Five **deterministic** prompts, same for every configuration:

1. Short instruction (~150 tok) — hash table explanation
2. Medium reasoning (~200 tok) — sequence 2,6,12,20… → 72
3. Long context (~300+ tok) — renewable energy document QA (scales with context)
4. Coding (~200 tok) — `find_duplicates` O(n)/O(1)
5. Structured output (~100 tok) — strict JSON

Generation settings are fixed per case:
- `temperature` per case (0.0–0.3)
- `seed = 42` for all completions
- `stop_sequences` where needed
- `max_tokens = min(case.max_tokens, context_length // 4)`

**Warmup + repetitions**: 1 warmup run (ignored) + 3 measured runs (median reported) → robust to noise. Configurable via `benchmark_repetitions`.

### 3.2 Determinism Guarantees

- Same prompts, same temperatures, same seed → reproducible outputs (modulo model nondeterminism).
- Candidate generation is sorted and deterministic, no unseeded `random`.
- All benchmark parameters (`repetitions`, `warmup`, `seed`, `temperature`, `candidate_generation=deterministic-sorted-sampling`) are stored in `OptimizationRun.benchmark_params` (JSON) in the database.

### 3.3 Known Measurement Noise

- **First-token timing** is estimated (see §5).
- **OS scheduling / background load** can jitter `prompt_tok_s` / `generation_tok_s` by 5–15%.
- **Thermal throttling** on laptops may reduce speed over long runs.
- **KV cache placement** effects are model- and driver-dependent.
- Mitigation: median over repetitions, stability score (`1 - CV`), validation stage (5× reruns of best).

---

## 4. Metric Normalization — Hardware-Agnostic

`lm_optimizer/scoring/normalization.py`

All former hardcoded formulas:

```
gen / 50
prompt / 1000
1 - TTFT/2000
context / 32768
1 - VRAM/6
```

are removed. New normalization:

### 4.1 Speeds & TTFT — Run-Relative

For each run, compute `min` / `max` among **feasible** (passed) configs:

- **Generation speed** (higher better): `norm_gen = (gen - min_gen)/(max_gen - min_gen)`
- **Prompt speed** (higher better): `norm_prompt = (prompt - min_prompt)/(max_prompt - min_prompt)`
- **Estimated TTFT** (lower better): `norm_ttft = (max_ttft - ttft)/(max_ttft - min_ttft)`

Zero-range (all equal) → `1.0` (no differentiation, not 0). Clamped to `[0,1]`. Never negative or >1 unless explicitly intended (none are).

This makes a 10 tok/s model on a 4 GB laptop and a 100 tok/s model on a 48 GB workstation both score fairly within their own run.

### 4.2 Context — Model-Relative

`norm_context = context_length / model_max_context`

- `model_max_context` = `ModelIdentity.context_limit` (from LM Studio) or observed max in run as fallback.
- Clamped `[0,1]`.
- A 8192 context on a 8192-max model scores `1.0`; same 8192 on a 131k-max model scores `0.0625`.

### 4.3 Memory — Hardware-Relative

`lm_optimizer/scoring/normalization.py:70`

Principle: **less VRAM is NOT always better**. Using more memory is fine if it yields substantially higher performance and still fits.

```
if peak_vram is None: return 0.5 (neutral)
if peak_vram > total_vram: return 0.0 (infeasible)
headroom = total_vram - peak_vram
ratio = headroom / total_vram
if ratio >= 0.15: return 1.0       # comfortable headroom
else:           return ratio / 0.15 # linear 0→1 as headroom 0→15%
```

- On a **24 GB GPU**: `5 GB/30 tok/s` (headroom 79% → 1.0) and `8 GB/50 tok/s` (66% → 1.0) **tie** on memory, so the faster config wins.
- On a **6 GB GPU**: `8 GB` → `0.0` (OOM), `5 GB` → headroom 16% → `1.0`; the feasible config wins.
- On a **4 GB GPU**: `3.9 GB` (headroom 2.5% → 0.17) scores lower than `2 GB` (50% → 1.0), reflecting tight fit risk.
- **CPU-only**: same logic with `total_ram_gb` and `peak_ram_gb`.
- **Fallback** (no hardware total known): run-relative `lower is better` for memory, documented as fallback.

### 4.4 Quality / Correctness — Already Normalized

`quality = overall` from `QualityScore` (0–1). No renormalization; directly used. Displayed as `X / Y checks passed` (see §6).

### 4.5 Stability — Already Normalized

`stability = 1 - CV` where `CV = stdev(gen_speeds)/mean(gen_speeds)` across benchmark repetitions, clamped `[0,1]`.

### 4.6 Weighted Score

`score = Σ weight[metric] * norm[metric]` with profile weights summing to ~1.0. Each component breakdown is exposed for explainability (see §7).

---

## 5. TTFT — Estimated, Not Measured

`lm_optimizer/services/benchmark.py:247` and `lm_optimizer/benchmark/runner.py:195`

LM Studio's `/api/v1/chat/completions` **without streaming** does not expose true time-to-first-token. The optimizer **estimates**:

```
estimated_ttft_ms = total_time_ms * 0.1  if prompt_tokens > 0 else total_time_ms
```

- Labeled **`estimated_ttft_ms`** in `BenchmarkMetrics` (alias `ttft_ms` for backward compat).
- UI and reports show **“Est. TTFT”** with note “estimated, no streaming”.
- If LM Studio adds streaming support with true TTFT, the estimator can be replaced and the field renamed to `ttft_ms` without breaking stored data.

**Do not present the estimate as exact**; it is used for relative ranking within a run (run-relative lower-is-better), not as an absolute latency SLA.

---

## 6. Quality / Correctness Scoring — Heuristics

`lm_optimizer/scoring/evaluator.py` and `lm_optimizer/services/quality.py`

### 6.1 What It Measures

Six deterministic heuristic dimensions per output, averaged to `overall` (0–1):

- `task_completion` — did the output contain required elements? (e.g. `def find_duplicates` for code, all JSON fields for format, token count for general)
- `factual_consistency` — keyword presence (e.g. `72` or `difference` for reasoning, `solar/wind/...` for context, type correctness for JSON)
- `format_compliance` — valid JSON, no extra markdown fences
- `coding_correctness` — function definition present
- `no_truncation` — ends with punctuation / `}` / `)` etc.
- `no_malformed` — no excessive n-gram repetition (threshold `0.3`)

Each dimension `≥0.9` counts as a **passed check**. Total `checks_passed / checks_total` is stored (`6` per test, `30` for 5 tests).

### 6.2 What It Does NOT Measure

- Scientific “model quality” percentage
- Human preference, creativity, or factual truth beyond keyword heuristics
- Performance on your own prompts

The score is a **proxy for “did the configuration break the model?”** (truncation, repetition, JSON invalid). A `30/30` is better than `29/30`, but `0.991` does **not** mean “99.1% objectively correct”.

### 6.3 Display Guidance

- Show **“Correctness / Quality Score (heuristic checks)”** and **`29 / 30 checks passed`** alongside the `0.XXX` value.
- Keep the threshold **configurable** (default `0.97`, but profile-dependent; see §7).
- In reports, include `details` (which checks failed) rather than a single decimal.

---

## 7. Profiles — Thresholds and Weights

`lm_optimizer/profiles/registry.py` and `lm_optimizer/domain/models.py:322`

| Profile | Threshold | Weights (gen / prompt / ttft / quality / stability / context / memory) | Intent |
|---------|-----------|--------------------------------------------------------------------------|--------|
| **Speed** | `0.95` | `0.40 / 0.25 / 0.15 / 0.10 / 0.05 / 0.03 / 0.02` | Maximize throughput; tolerates minor heuristic misses. |
| **Balanced** | `0.97` | `0.27 / 0.15 / 0.12 / 0.12 / 0.12 / 0.11 / 0.11` | Real-world even trade-off (default). |
| **Context** | `0.97` | `0.10 / 0.10 / 0.10 / 0.15 / 0.15 / 0.25 / 0.15` | Maximize stable context while keeping quality. |
| **Quality** | `0.99` | `0.08 / 0.08 / 0.08 / 0.35 / 0.13 / 0.21 / 0.07` | Maximize correctness and usable context; speed secondary. |
| **Custom** | user (0.90–1.0) | user-defined (must sum ≈1.0) | Fully configurable. |

**Why thresholds differ**: lower threshold for Speed allows a config that is 5% heuristic-fail but 2× faster to pass filtering; higher threshold for Quality strictly rejects any heuristic failure. The profile **consistently** affects **both**:
- **Filtering**: `passes_threshold(overall >= minimum_quality)` before scoring
- **Scoring**: `weight["quality"]` directly multiplies normalized quality.

A “quality” profile with low quality weight would be inconsistent — the registry enforces `quality ≥0.30` for the Quality profile.

---

## 8. Baseline — Actual Measurement

Before optimization, the optimizer **captures the actual current configuration**:

1. Checks `LMStudioClient.get_model(model_id).load_config` if the model is already loaded, otherwise uses the minimal feasible config.
2. **Benchmarks it** with the full suite (same repetitions, same quality evaluation).
3. Saves as `baseline_config_id` and `baseline_metrics` (gen, prompt, TTFT, context, VRAM, RAM, quality checks, stability).

After optimization, the report shows:

```
Baseline          Optimized        Change
Gen  12.3 tok/s   24.1 tok/s      +95.9%
Prompt 450 tok/s  620 tok/s       +37.8%
...
```

Percentage changes are **only from actual measurements** (`(optimized - baseline)/baseline * 100`), not from estimates or hardcoded baselines. If no baseline was captured (model not loaded), the comparison is omitted rather than fabricated.

---

## 9. Score Explainability

Every `ConfigurationResult` stores:

```json
{
  "score": 0.82,
  "score_breakdown": {
    "generation_speed": 0.91,
    "prompt_speed": 0.76,
    "ttft": 0.88,
    "quality": 1.00,
    "stability": 0.96,
    "context": 0.88,
    "memory_efficiency": 0.71
  }
}
```

The UI and CLI render:

```
Balanced score: 0.82
  Generation speed: 0.91 (weight 0.27) → +0.246
  Prompt speed:     0.76 (weight 0.15) → +0.114
  Context:          0.88 (weight 0.11) → +0.097
  ...
```

Users can see **why** a config won (e.g. “Context profile chose 16384 ctx despite 10% slower gen because context norm was 1.0 vs 0.5”).

---

## 10. Pareto Frontier

`lm_optimizer/services/optimizer.py:455` and `lm_optimizer/optimizer/engine.py:437`

Multi-objective **Pareto frontier**: a config is on the frontier if **no other** feasible config is **better in all** metrics and **strictly better in at least one** of:

- `quality (overall)`
- `generation tok/s`
- `prompt tok/s`
- `context_length`
- `peak_vram_gb` (lower is better)

Formally, `r` is dominated if ∃ `other ≠ r` with `other ≥ r` on all objectives and `other > r` on at least one. The frontier is shown in the Web UI as an interactive chart and as cards.

Pareto does not use weights; it shows **trade-offs** independent of profile.

---

## 11. Limitations

- **Heuristic quality**: keyword / JSON checks are proxies; a model can pass all checks yet hallucinate, or fail checks yet be useful. For critical use, add human or LLM-as-judge evaluation.
- **Estimated TTFT**: ranking within a run is meaningful, absolute values are not.
- **VRAM measurement**: `peak_vram_gb` may be `None` if the backend does not report it; memory score falls back to neutral `0.5` or run-relative.
- **Single-GPU focus**: multi-GPU is detected but optimization currently uses the primary GPU’s VRAM for memory scoring; multi-GPU sharding is not optimized.
- **Thermal / background load**: long runs may see speed drift; use the stability score and validation repetitions to gauge variance.
- **Model nondeterminism**: even with `seed=42`, some backends are nondeterministic; median over repetitions mitigates but does not eliminate.

---

## 12. Remaining Constants — Justification

| Constant | Location | Purpose | Justification |
|----------|----------|---------|---------------|
| `[2048,4096,…,131072]` | `search_space`, `discovery` | Benchmark test points for context | Standard LLM context sizes; filtered to model max, not used for scoring. |
| `[64,128,256,512,1024]` | `search_space` | Eval batch candidates | Powers of two covering typical `eval_batch_size` range. |
| `MEMORY_HEADROOM_THRESHOLD = 0.15` | `scoring/normalization` | Memory headroom for hardware-relative scoring | 15% free VRAM as “comfortable” — relative fraction, not fixed GB. |
| `EPSILON = 1e-9` | `scoring/normalization` | Zero-range guard | Numerical epsilon, not hardware-specific. |
| `seed = 42` | `benchmark` | Deterministic generation | Fixed seed for reproducibility, not a performance assumption. |
| `repetitions = 3`, `warmup = 1`, `validation = 5` | `config`, `optimizer` | Measurement robustness | Counts for median / stability, not hardware caps. |
| `0.95 / 0.97 / 0.99` | `profiles` | Quality thresholds | Intentional per-profile filtering levels, documented in §7. |

No remaining constant assumes `6 GB`, `RTX 3060`, `50 tok/s`, `1000 prompt tok/s`, `2000 ms`, or `32768` as a universal truth.

---

## 13. Version

- **v0.1.0**: initial benchmark UI with hardware-specific scoring (deprecated).
- **v0.1.1**: correctness pass — hardware-agnostic scoring, estimated TTFT labeling, deterministic sampling, RoPE experimental, baseline measurement, explainable scores, comprehensive tests.

For implementation details, see `lm_optimizer/scoring/normalization.py`, `lm_optimizer/services/optimizer.py`, and `lm_optimizer/tests/test_optimization_correctness.py`.
