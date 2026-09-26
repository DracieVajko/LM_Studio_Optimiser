"""Hardware-agnostic metric normalization for optimization scoring.

This module implements run-relative and hardware/model-relative normalization
to avoid hardcoded assumptions such as 50 tok/s, 1000 prompt tok/s, 2000ms TTFT,
32768 context, or 6GB VRAM.

All remaining constants in this file are either:
- benchmark-specific (e.g. list of candidate context lengths)
- mathematical epsilon for zero-range handling
- documented headroom thresholds (15% = 0.15) for memory feasibility

Design principles:
- Speeds (generation, prompt) are normalized against observed feasible
  configurations in the current run, not against fixed global maxima.
- Context is normalized against model_max_context (model-relative).
- Memory is hardware-relative: a configuration that fits with headroom
  scores 1.0; using more memory is not penalized unless headroom is tight.
- TTFT is estimated (not directly measured); normalization is inverted
  (lower estimated TTFT is better) and handled run-relative.
- Zero-range handling: if max==min, return 1.0 (no differentiation).
- All normalized scores are clamped to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

EPSILON = 1e-9
# Headroom threshold: configurations leaving >=15% of total VRAM/RAM free
# are considered to have comfortable headroom and score 1.0 for memory.
# Below this, score degrades linearly to 0 at 0% headroom.
MEMORY_HEADROOM_THRESHOLD = 0.15


def _clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def normalize_higher_better(value: float, min_val: float, max_val: float) -> float:
    """Normalize where higher value is better, run-relative.

    Returns 0 for min_val, 1 for max_val. If range is zero, returns 1.0
    (no differentiation - all candidates equal on this metric).
    Clamped to [0, 1].
    """
    rng = max_val - min_val
    if abs(rng) < EPSILON:
        return 1.0
    return _clamp01((value - min_val) / rng)


def normalize_lower_better(value: float, min_val: float, max_val: float) -> float:
    """Normalize where lower value is better (e.g. TTFT).

    Returns 1 for min_val (best), 0 for max_val (worst). Zero-range -> 1.0.
    Clamped to [0, 1].
    """
    rng = max_val - min_val
    if abs(rng) < EPSILON:
        return 1.0
    return _clamp01((max_val - value) / rng)


def normalize_context(
    context_length: int, model_max_context: int | None, observed_max: int | None = None
) -> float:
    """Normalize context relative to model capability.

    Uses model_max_context as denominator (model-relative). Falls back to
    observed_max if model_max_context is unknown. Clamped to [0,1].
    """
    if model_max_context and model_max_context > 0:
        return _clamp01(context_length / model_max_context)
    if observed_max and observed_max > 0:
        return _clamp01(context_length / observed_max)
    # No reference - fallback to run-relative or neutral
    return 1.0 if context_length > 0 else 0.0


def compute_memory_score(
    peak_vram_gb: float | None,
    peak_ram_gb: float | None,
    hardware_total_vram_gb: float | None,
    hardware_total_ram_gb: float | None,
    observed_vrams: list[float] | None = None,
    observed_rams: list[float] | None = None,
) -> float:
    """Hardware-relative memory scoring.

    Principles:
    - Do NOT define "less VRAM is always better".
    - If total memory is known, score based on headroom:
        headroom_ratio = (total - used) / total
        if headroom_ratio >= MEMORY_HEADROOM_THRESHOLD -> 1.0
        elif headroom_ratio >= 0 -> headroom_ratio / MEMORY_HEADROOM_THRESHOLD
        else -> 0.0 (OOM, exceeds capacity)
    - This means on a 24GB GPU, 5GB/30tok/s and 8GB/50tok/s both score 1.0
      for memory (both comfortable), so performance decides.
      On a 6GB GPU, 8GB scores 0.0 (infeasible), 5GB scores ~0.17/0.15=1.0?
      Actually 5GB on 6GB: headroom 1GB => ratio 0.166 => >=0.15 =>1.0 as well.
      So tight but still feasible both get high score, but if even tighter
      e.g. 5.7GB on 6GB => headroom 0.3GB ratio 0.05 => 0.33
    - If hardware total unknown, fallback to run-relative (less memory better,
      but documented as fallback).
    - If peak is None, return neutral 0.5.
    - GPU VRAM takes precedence if hardware has GPU; otherwise use RAM.
    """
    # Prefer VRAM if GPU exists
    has_gpu = hardware_total_vram_gb is not None and hardware_total_vram_gb > 0
    if has_gpu:
        if peak_vram_gb is None:
            return 0.5
        total = hardware_total_vram_gb
        used = peak_vram_gb
        if used > total:
            return 0.0
        headroom = total - used
        ratio = headroom / total if total > 0 else 0.0
        if ratio >= MEMORY_HEADROOM_THRESHOLD:
            return 1.0
        if ratio < 0:
            return 0.0
        return _clamp01(ratio / MEMORY_HEADROOM_THRESHOLD)

    # CPU-only: use RAM
    if hardware_total_ram_gb and hardware_total_ram_gb > 0:
        if peak_ram_gb is None:
            return 0.5
        total = hardware_total_ram_gb
        used = peak_ram_gb
        if used > total:
            return 0.0
        headroom = total - used
        ratio = headroom / total if total > 0 else 0.0
        if ratio >= MEMORY_HEADROOM_THRESHOLD:
            return 1.0
        return _clamp01(ratio / MEMORY_HEADROOM_THRESHOLD)

    # Fallback: run-relative (no hardware info)
    # Less memory is considered more efficient, but this is fallback only.
    if observed_vrams and len(observed_vrams) > 1 and peak_vram_gb is not None:
        mn = min(observed_vrams)
        mx = max(observed_vrams)
        return normalize_lower_better(peak_vram_gb, mn, mx)
    if observed_rams and len(observed_rams) > 1 and peak_ram_gb is not None:
        mn = min(observed_rams)
        mx = max(observed_rams)
        return normalize_lower_better(peak_ram_gb, mn, mx)
    return 0.5


@dataclass
class NormalizationBounds:
    """Observed min/max bounds for run-relative metrics within a run."""

    min_gen: float = 0.0
    max_gen: float = 0.0
    min_prompt: float = 0.0
    max_prompt: float = 0.0
    min_estimated_ttft: float = 0.0
    max_estimated_ttft: float = 0.0
    max_observed_context: int = 0
    observed_vrams: list[float] = None  # type: ignore
    observed_rams: list[float] = None  # type: ignore

    def __post_init__(self):
        if self.observed_vrams is None:
            self.observed_vrams = []
        if self.observed_rams is None:
            self.observed_rams = []


def compute_bounds(
    results: list, hardware=None, model_max_context: int | None = None
) -> NormalizationBounds:
    """Compute run-relative bounds from a list of ConfigurationResult or BenchmarkResult.

    Only passed/feasible configurations are considered.
    """
    # Filter to successful configs
    feasible = [
        r
        for r in results
        if getattr(r, "status", None) in ("passed", "PASSED") or getattr(r, "passed", False) is True
    ]
    # Also accept status as enum
    if not feasible:
        # Fallback: use all with metrics
        feasible = [r for r in results if getattr(r, "metrics", None)]

    gen_vals = []
    prompt_vals = []
    ttft_vals = []
    vram_vals = []
    ram_vals = []
    max_ctx = 0

    for r in feasible:
        # Support both ConfigurationResult and BenchmarkResult
        try:
            g = r.get_avg_generation_tok_s() if hasattr(r, "get_avg_generation_tok_s") else 0.0
            # need to handle bound method vs function
            if callable(g):
                g = g()
            # Actually get_avg_generation_tok_s is method on ConfigurationResult and also on BenchmarkResult via helper
            # For BenchmarkResult, we may need helper
        except Exception:
            g = 0.0
        # Alternative: try to call properly
        try:
            if hasattr(r, "get_avg_generation_tok_s"):
                # ConfigurationResult has method
                g = r.get_avg_generation_tok_s()  # type: ignore
            else:
                g = 0.0
        except Exception:
            g = 0.0

        try:
            p = r.get_avg_prompt_tok_s() if hasattr(r, "get_avg_prompt_tok_s") else 0.0
            if hasattr(r, "get_avg_prompt_tok_s"):
                p = r.get_avg_prompt_tok_s()  # type: ignore
        except Exception:
            p = 0.0

        try:
            t = (
                r.get_avg_estimated_ttft_ms()
                if hasattr(r, "get_avg_estimated_ttft_ms")
                else r.get_avg_ttft_ms()
                if hasattr(r, "get_avg_ttft_ms")
                else 0.0
            )
            if hasattr(r, "get_avg_estimated_ttft_ms"):
                t = r.get_avg_estimated_ttft_ms()  # type: ignore
            elif hasattr(r, "get_avg_ttft_ms"):
                t = r.get_avg_ttft_ms()  # type: ignore
        except Exception:
            t = 0.0

        if g and g > 0:
            gen_vals.append(g)
        if p and p > 0:
            prompt_vals.append(p)
        if t and t > 0:
            ttft_vals.append(t)

        v = getattr(r, "peak_vram_gb", None)
        if v is not None and v > 0:
            vram_vals.append(v)
        ram = getattr(r, "peak_ram_gb", None)
        if ram is not None and ram > 0:
            ram_vals.append(ram)

        ctx = getattr(r, "context_length", 0) or getattr(r, "context_length", 0)
        if ctx and ctx > max_ctx:
            max_ctx = ctx

    # If model_max_context known, use it as max for context normalization context?
    # But for bounds we just track observed max
    return NormalizationBounds(
        min_gen=min(gen_vals) if gen_vals else 0.0,
        max_gen=max(gen_vals) if gen_vals else 0.0,
        min_prompt=min(prompt_vals) if prompt_vals else 0.0,
        max_prompt=max(prompt_vals) if prompt_vals else 0.0,
        min_estimated_ttft=min(ttft_vals) if ttft_vals else 0.0,
        max_estimated_ttft=max(ttft_vals) if ttft_vals else 0.0,
        max_observed_context=max_ctx,
        observed_vrams=vram_vals,
        observed_rams=ram_vals,
    )


def score_result_breakdown(
    result,
    bounds: NormalizationBounds,
    hardware,
    model_max_context: int | None,
    weights: dict[str, float],
) -> dict[str, float]:
    """Compute normalized breakdown for a single result.

    Returns dict with keys matching profile weights:
    generation_speed, prompt_speed, ttft, quality, stability, context, memory_efficiency
    All values in [0,1].
    """
    # Generation speed - higher better, run-relative
    gen = 0.0
    try:
        if hasattr(result, "get_avg_generation_tok_s") or hasattr(result, "get_avg_generation_tok_s"):
            gen = result.get_avg_generation_tok_s()  # type: ignore
    except Exception:
        gen = 0.0
    norm_gen = normalize_higher_better(gen, bounds.min_gen, bounds.max_gen)

    # Prompt speed
    prompt = 0.0
    try:
        if hasattr(result, "get_avg_prompt_tok_s"):
            prompt = result.get_avg_prompt_tok_s()  # type: ignore
    except Exception:
        prompt = 0.0
    norm_prompt = normalize_higher_better(prompt, bounds.min_prompt, bounds.max_prompt)

    # Estimated TTFT - lower better, run-relative
    ttft = 0.0
    try:
        if hasattr(result, "get_avg_estimated_ttft_ms"):
            ttft = result.get_avg_estimated_ttft_ms()  # type: ignore
        elif hasattr(result, "get_avg_ttft_ms"):
            ttft = result.get_avg_ttft_ms()  # type: ignore
    except Exception:
        ttft = 0.0
    norm_ttft = normalize_lower_better(ttft, bounds.min_estimated_ttft, bounds.max_estimated_ttft)

    # Quality / correctness - already 0-1, use directly
    quality = 1.0
    qs = getattr(result, "quality_score", None)
    if qs is not None:
        if hasattr(qs, "overall"):
            quality = float(qs.overall)
        elif isinstance(qs, (int, float)):
            quality = float(qs)
        else:
            quality = 1.0
    norm_quality = _clamp01(quality)

    # Stability - already 0-1
    stability = float(getattr(result, "stability_score", 1.0) or 1.0)
    norm_stability = _clamp01(stability)

    # Context - model-relative
    ctx_len = int(getattr(result, "context_length", 0) or 0)
    norm_ctx = normalize_context(ctx_len, model_max_context, bounds.max_observed_context)

    # Memory - hardware-relative
    hw_vram = None
    hw_ram = None
    if hardware is not None:
        # Support both HardwareInfo (domain) and hardware detection HardwareInfo
        if hasattr(hardware, "gpus") and hardware.gpus:
            try:
                hw_vram = (
                    sum(g.vram_gb for g in hardware.gpus)
                    if len(hardware.gpus) > 1
                    else hardware.gpus[0].vram_gb
                )
            except Exception:
                hw_vram = None
        elif hasattr(hardware, "gpu") and hardware.gpu:
            hw_vram = hardware.gpu.vram_gb  # type: ignore
        if hasattr(hardware, "total_ram_gb"):
            hw_ram = hardware.total_ram_gb
        elif hasattr(hardware, "memory") and hasattr(hardware.memory, "total_gb"):
            hw_ram = hardware.memory.total_gb  # type: ignore

    peak_vram = getattr(result, "peak_vram_gb", None)
    peak_ram = getattr(result, "peak_ram_gb", None)
    norm_mem = compute_memory_score(
        peak_vram, peak_ram, hw_vram, hw_ram, bounds.observed_vrams, bounds.observed_rams
    )

    return {
        "generation_speed": norm_gen,
        "prompt_speed": norm_prompt,
        "ttft": norm_ttft,
        "quality": norm_quality,
        "stability": norm_stability,
        "context": norm_ctx,
        "memory_efficiency": norm_mem,
    }


def weighted_score(breakdown: dict[str, float], weights: dict[str, float]) -> float:
    """Compute weighted sum. Weights are expected to sum to ~1.0.

    Handles alternative weight keys: generation_speed vs generation_speed,
    prompt_speed vs prompt_processing, context vs context_capacity, etc.
    """
    # Map aliases
    alias_map = {
        "generation_speed": ["generation_speed"],
        "prompt_speed": ["prompt_speed", "prompt_processing"],
        "ttft": ["ttft"],
        "quality": ["quality"],
        "stability": ["stability"],
        "context": ["context", "context_capacity"],
        "memory_efficiency": ["memory_efficiency"],
    }
    total = 0.0
    for target_key, aliases in alias_map.items():
        w = 0.0
        for a in aliases:
            if a in weights:
                w = weights[a]
                break
        total += w * breakdown.get(target_key, 0.0)
    return _clamp01(total)
