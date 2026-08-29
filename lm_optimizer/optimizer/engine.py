"""DEPRECATED LEGACY ENGINE - will be removed in 2.0. Use lm_optimizer.services.optimizer.AdaptiveOptimizer instead.

This file is kept for backward compat with lm_optimizer.ui.web and storage/checkpoint type hints.
The authoritative implementation is services/optimizer.py (hardware-agnostic, DB-persistent, baseline, explainability).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from lm_optimizer.api.client import LMStudioClient, LoadConfig
from lm_optimizer.benchmark.runner import BenchmarkResult, BenchmarkRunner
from lm_optimizer.config import config
from lm_optimizer.discovery.inspector import ModelCapabilities, ModelDiscovery
from lm_optimizer.logging_config import get_logger
from lm_optimizer.profiles.registry import Profile, ProfileRegistry
from lm_optimizer.scoring.evaluator import QualityEvaluator
from lm_optimizer.scoring.normalization import (
    compute_bounds as compute_norm_bounds,
)
from lm_optimizer.scoring.normalization import (
    compute_memory_score,
    normalize_context,
)
from lm_optimizer.storage.checkpoint import CheckpointManager

logger = get_logger(__name__)


class OptimizationStage(str, Enum):
    """Optimization stages."""

    DISCOVERY = "discovery"
    COARSE_SEARCH = "coarse_search"
    REFINEMENT = "refinement"
    BATCH_OPTIMIZATION = "batch_optimization"
    FINAL_VALIDATION = "final_validation"
    COMPLETE = "complete"


@dataclass
class OptimizationState:
    """Current state of optimization."""

    model_id: str
    profile: Profile
    stage: OptimizationStage = OptimizationStage.DISCOVERY
    capabilities: ModelCapabilities | None = None
    tested_configs: list[BenchmarkResult] = field(default_factory=list)
    best_config: BenchmarkResult | None = None
    pareto_frontier: list[BenchmarkResult] = field(default_factory=list)
    current_config_id: str | None = None
    start_time: datetime = field(default_factory=datetime.now)
    errors: list[str] = field(default_factory=list)
    failed_regions: set = field(default_factory=set)


@dataclass
class OptimizationResult:
    """Final optimization result."""

    model_id: str
    profile: Profile
    recommended_config: LoadConfig
    recommended_context: int
    best_result: BenchmarkResult
    pareto_frontier: list[BenchmarkResult]
    all_results: list[BenchmarkResult]
    optimization_time_s: float
    speedup_vs_baseline: float | None = None
    quality_change: float | None = None
    score_breakdown: dict | None = None
    is_experimental: bool = False


class OptimizationEngine:
    """Main optimization engine with adaptive search, hardware-agnostic scoring."""

    def __init__(self, client: LMStudioClient):
        self.client = client
        self.discovery = ModelDiscovery(client)
        self.runner = BenchmarkRunner(client)
        self.evaluator = QualityEvaluator()
        self.profiles = ProfileRegistry()
        self.checkpoint = CheckpointManager()
        self.state: OptimizationState | None = None

    async def optimize(
        self,
        model_id: str,
        profile_name: str = "balanced",
        baseline_result: BenchmarkResult | None = None,
    ) -> OptimizationResult:
        """Run full optimization for a model."""
        profile = self.profiles.get(profile_name)
        self.state = OptimizationState(model_id=model_id, profile=profile)

        logger.info("Starting optimization", model=model_id, profile=profile_name)

        try:
            # Stage 1: Discovery
            await self._stage_discovery()

            # Stage 2: Coarse search
            await self._stage_coarse_search()

            # Stage 3: Refinement
            await self._stage_refinement()

            # Stage 4: Batch optimization
            await self._stage_batch_optimization()

            # Stage 5: Final validation
            await self._stage_final_validation(baseline_result)

            # Compile result
            result = self._compile_result(baseline_result)
            return result

        except Exception as e:
            logger.error("Optimization failed", model=model_id, error=str(e))
            raise
        finally:
            # Ensure model is unloaded
            await self.client.unload_model(model_id=model_id)

    async def _stage_discovery(self) -> None:
        """Stage 1: Discover model capabilities."""
        logger.info("Stage 1: Capability discovery", model=self.state.model_id)
        self.state.stage = OptimizationStage.DISCOVERY

        self.state.capabilities = await self.discovery.inspect_model(self.state.model_id)
        logger.info(
            "Model capabilities",
            model=self.state.model_id,
            arch=self.state.capabilities.architecture,
            max_ctx=self.state.capabilities.max_context_length,
            is_moe=self.state.capabilities.is_moe,
            est_vram=self.state.capabilities.estimated_vram_gb,
        )

        # Save checkpoint
        await self.checkpoint.save(self.state)

    async def _stage_coarse_search(self) -> None:
        """Stage 2: Coarse search with deterministic intelligent sampling."""
        logger.info("Stage 2: Coarse search", model=self.state.model_id)
        self.state.stage = OptimizationStage.COARSE_SEARCH

        caps = self.state.capabilities
        if not caps:
            return

        # Generate coarse search configurations with deterministic sampling
        configs = self._generate_coarse_configs(caps)

        for i, (ctx, gpu_ratio, kv_gpu, flash, batch) in enumerate(configs):
            if self._should_skip_config(ctx, gpu_ratio):
                continue

            load_config = LoadConfig(
                context_length=ctx,
                gpu_ratio=gpu_ratio,
                flash_attention=flash,
                offload_kv_cache_to_gpu=kv_gpu,
                eval_batch_size=batch,
            )

            logger.info(
                "Testing coarse config",
                idx=i + 1,
                total=len(configs),
                ctx=ctx,
                gpu=gpu_ratio,
                kv_gpu=kv_gpu,
                flash=flash,
                batch=batch,
            )

            result = await self._run_config(load_config, ctx)
            if result and result.passed:
                self.state.tested_configs.append(result)
                self._update_best(result)
            elif result and not result.passed:
                self.state.failed_regions.add((ctx, gpu_ratio))

            await self.checkpoint.save(self.state)

    def _deterministic_sample(self, items: list, k: int) -> list:
        """Deterministically sample k items covering low/high, sorted."""
        if not items:
            return []
        if len(items) <= k:
            return sorted(items)
        sorted_items = sorted(items)
        indices = []
        for i in range(k):
            idx = int(round(i * (len(sorted_items) - 1) / (k - 1))) if k > 1 else 0
            indices.append(idx)
        seen = set()
        result = []
        for idx in indices:
            val = sorted_items[idx]
            if val not in seen:
                result.append(val)
                seen.add(val)
        return sorted(result)

    def _generate_coarse_configs(self, caps: ModelCapabilities) -> list[tuple]:
        """Generate coarse search configuration combinations with intelligent sampling."""
        # Use deterministic sampling to cover dimensions, not just first N
        ctx_candidates = self._deterministic_sample(
            caps.generate_context_candidates(), min(4, len(caps.generate_context_candidates()))
        )
        gpu_candidates = self._deterministic_sample(
            caps.generate_gpu_ratio_candidates(), min(3, len(caps.generate_gpu_ratio_candidates()))
        )
        kv_candidates = sorted(caps.generate_kv_cache_candidates())
        flash_candidates = sorted(caps.generate_flash_attention_candidates())
        batch_candidates = self._deterministic_sample(
            caps.generate_batch_size_candidates(),
            min(2, len(caps.generate_batch_size_candidates())),
        )

        configs = []
        for ctx in sorted(ctx_candidates):
            for gpu in sorted(gpu_candidates):
                for kv in kv_candidates:
                    for flash in flash_candidates:
                        for batch in sorted(batch_candidates):
                            configs.append((ctx, gpu, kv, flash, batch))

        # Deterministically sample to 20 if too many, evenly spaced
        configs.sort(key=lambda x: (x[0], x[1], x[4], x[3], x[2]))
        if len(configs) > 20:
            step = len(configs) / 20
            sampled = []
            for i in range(20):
                idx = int(round(i * step))
                if idx >= len(configs):
                    idx = len(configs) - 1
                sampled.append(configs[idx])
            # Dedup
            seen = set()
            deduped = []
            for c in sampled:
                if c not in seen:
                    seen.add(c)
                    deduped.append(c)
            return deduped
        return configs[:20]

    async def _stage_refinement(self) -> None:
        """Stage 3: Refine around best configurations with interaction tests."""
        logger.info("Stage 3: Refinement", model=self.state.model_id)
        self.state.stage = OptimizationStage.REFINEMENT

        if not self.state.best_config:
            logger.warning("No best config found, skipping refinement")
            return

        # Promising candidates: best + Pareto top 2
        promising = [self.state.best_config]
        pareto = self._compute_pareto_frontier(self.state.tested_configs)
        pareto_sorted = sorted(pareto, key=lambda r: r.get_avg_generation_tok_s(), reverse=True)
        for pc in pareto_sorted:
            if pc.config_id != self.state.best_config.config_id and len(promising) < 3:
                promising.append(pc)

        caps = self.state.capabilities
        tested_keys = {
            (r.context_length, r.gpu_ratio, r.eval_batch_size) for r in self.state.tested_configs
        }

        for best in promising:
            # Refine context around best
            if caps.max_context_length:
                ctx_candidates = self._refine_context(best.context_length, caps.max_context_length)
                # Also add nearest neighbors from search space
                all_ctx = caps.generate_context_candidates()
                sorted_ctx = sorted(all_ctx)
                try:
                    idx = sorted_ctx.index(best.context_length)
                    for off in [-1, 1]:
                        nidx = idx + off
                        if 0 <= nidx < len(sorted_ctx) and sorted_ctx[nidx] not in ctx_candidates:
                            ctx_candidates.append(sorted_ctx[nidx])
                except ValueError:
                    pass
                ctx_candidates = sorted(set(ctx_candidates))
            else:
                ctx_candidates = [best.context_length]

            # Refine GPU ratio around best
            gpu_candidates = self._refine_gpu_ratio(best.gpu_ratio or 0.8)
            # Filter to valid ratios from capabilities for hardware-agnostic
            valid_gpus = caps.generate_gpu_ratio_candidates()
            gpu_candidates = [g for g in gpu_candidates if g in valid_gpus] or gpu_candidates

            # Refine batch size
            batch_candidates = self._refine_batch_size(best.eval_batch_size or 256)

            # Test interactions: toggle flash/kv at best point
            for flash in caps.generate_flash_attention_candidates():
                if flash != best.flash_attention:
                    key = (best.context_length, best.gpu_ratio, best.eval_batch_size)
                    cfg = LoadConfig(
                        context_length=best.context_length,
                        gpu_ratio=best.gpu_ratio,
                        flash_attention=flash,
                        offload_kv_cache_to_gpu=best.kv_cache_gpu,
                        eval_batch_size=best.eval_batch_size,
                    )
                    if (
                        key not in tested_keys
                        and (best.context_length, best.gpu_ratio) not in self.state.failed_regions
                    ):
                        result = await self._run_config(cfg, best.context_length)
                        if result and result.passed:
                            self.state.tested_configs.append(result)
                            self._update_best(result)
                        await self.checkpoint.save(self.state)

            for ctx in ctx_candidates:
                for gpu in gpu_candidates:
                    for batch in batch_candidates:
                        key = (ctx, gpu, batch)
                        if key in tested_keys:
                            continue
                        if (ctx, gpu) in self.state.failed_regions:
                            continue
                        load_config = LoadConfig(
                            context_length=ctx,
                            gpu_ratio=gpu,
                            flash_attention=best.flash_attention,
                            offload_kv_cache_to_gpu=best.kv_cache_gpu,
                            eval_batch_size=batch,
                        )

                        result = await self._run_config(load_config, ctx)
                        if result and result.passed:
                            self.state.tested_configs.append(result)
                            self._update_best(result)
                        elif result and not result.passed:
                            self.state.failed_regions.add((ctx, gpu))

                        await self.checkpoint.save(self.state)

    def _refine_context(self, best_ctx: int, max_ctx: int) -> list[int]:
        """Generate context candidates around best."""
        candidates = [best_ctx]
        step = max(512, best_ctx // 4)

        for offset in [-step, step, -2 * step, 2 * step]:
            ctx = best_ctx + offset
            if 512 <= ctx <= max_ctx:
                candidates.append(ctx)

        return sorted(set(candidates))

    def _refine_gpu_ratio(self, best_gpu: float) -> list[float]:
        """Generate GPU ratio candidates around best."""
        candidates = [best_gpu]
        for offset in [-0.1, -0.05, 0.05, 0.1, -0.15, 0.15]:
            gpu = round(best_gpu + offset, 2)
            if 0.0 <= gpu <= 1.0:
                candidates.append(gpu)
        return sorted(set(candidates))

    def _refine_batch_size(self, best_batch: int) -> list[int]:
        """Generate batch size candidates around best."""
        candidates = [best_batch]
        for mult in [0.5, 0.75, 1.5, 2.0]:
            batch = int(best_batch * mult)
            batch = max(32, min(2048, batch))
            # Round to nearest 32
            batch = round(batch / 32) * 32
            candidates.append(batch)
        return sorted(set(candidates))

    async def _stage_batch_optimization(self) -> None:
        """Stage 4: Optimize eval batch size for best config."""
        logger.info("Stage 4: Batch optimization", model=self.state.model_id)
        self.state.stage = OptimizationStage.BATCH_OPTIMIZATION

        if not self.state.best_config:
            return

        best = self.state.best_config
        caps = self.state.capabilities
        batch_candidates = caps.generate_batch_size_candidates()

        for batch in batch_candidates:
            if batch == best.eval_batch_size:
                continue

            load_config = LoadConfig(
                context_length=best.context_length,
                gpu_ratio=best.gpu_ratio,
                flash_attention=best.flash_attention,
                offload_kv_cache_to_gpu=best.kv_cache_gpu,
                eval_batch_size=batch,
            )

            result = await self._run_config(load_config, best.context_length)
            if result and result.passed:
                self.state.tested_configs.append(result)
                self._update_best(result)

            await self.checkpoint.save(self.state)

    async def _stage_final_validation(self, baseline: BenchmarkResult | None) -> None:
        """Stage 5: Final validation with multiple runs."""
        logger.info("Stage 5: Final validation", model=self.state.model_id)
        self.state.stage = OptimizationStage.FINAL_VALIDATION

        if not self.state.best_config:
            logger.warning("No best config for final validation")
            self.state.stage = OptimizationStage.COMPLETE
            return

        # Run best config 5 more times for statistical significance
        best = self.state.best_config
        load_config = LoadConfig(
            context_length=best.context_length,
            gpu_ratio=best.gpu_ratio,
            flash_attention=best.flash_attention,
            offload_kv_cache_to_gpu=best.kv_cache_gpu,
            eval_batch_size=best.eval_batch_size,
        )

        validation_results = []
        for i in range(5):
            result = await self._run_config(load_config, best.context_length)
            if result and result.passed:
                validation_results.append(result)

        if validation_results:
            # Use median of validation runs
            self.state.best_config = self._median_result(validation_results)

        # Calculate Pareto frontier
        self.state.pareto_frontier = self._compute_pareto_frontier(self.state.tested_configs)

        # Compare with baseline if provided
        if baseline:
            self._compare_with_baseline(baseline)

        self.state.stage = OptimizationStage.COMPLETE
        await self.checkpoint.save(self.state)

    async def _run_config(
        self, load_config: LoadConfig, context_length: int
    ) -> BenchmarkResult | None:
        """Run a single configuration with error handling."""
        try:
            result = await self.runner.run_benchmark(
                self.state.model_id,
                load_config,
                context_length,
            )

            if not result.passed:
                logger.warning(
                    "Config failed", config_id=result.config_id, reason=result.failure_reason
                )
                self.state.errors.append(f"{result.config_id}: {result.failure_reason}")
                return None

            # Evaluate quality / correctness
            quality_scores = self.evaluator.evaluate_all(self.state.model_id, result.metrics)
            agg_quality = self.evaluator.aggregate_quality(quality_scores)
            result.quality_score = agg_quality.overall
            # Keep full quality object for scoring
            result._quality_obj = agg_quality

            if not self.evaluator.passes_threshold(agg_quality):
                logger.warning(
                    "Config below quality threshold",
                    config_id=result.config_id,
                    quality=agg_quality.overall,
                    checks=f"{agg_quality.checks_passed}/{agg_quality.checks_total}",
                    threshold=self.evaluator.minimum_score,
                )
                result.passed = False
                result.failure_reason = f"Quality/correctness {agg_quality.overall:.3f} ({agg_quality.checks_passed}/{agg_quality.checks_total} checks) below threshold {self.evaluator.minimum_score}"
                return None

            logger.info(
                "Config passed",
                config_id=result.config_id,
                gen_tok_s=result.get_avg_generation_tok_s(),
                quality=f"{agg_quality.checks_passed}/{agg_quality.checks_total}",
            )

            return result

        except Exception as e:
            logger.error("Config error", error=str(e))
            self.state.errors.append(f"Exception: {e!s}")
            # Ensure unload on error
            await self.client.unload_model(model_id=self.state.model_id)
            return None

    def _update_best(self, result: BenchmarkResult) -> None:
        """Update best configuration based on hardware-agnostic profile scoring."""
        if not self.state.best_config:
            self.state.best_config = result
            return

        current_score = self._score_result(self.state.best_config)
        new_score = self._score_result(result)

        if new_score > current_score:
            logger.info(
                "New best config",
                old_score=current_score,
                new_score=new_score,
                config_id=result.config_id,
            )
            self.state.best_config = result

    def _score_result(self, result: BenchmarkResult) -> float:
        """Score a result hardware-agnostically (run-relative).

        Uses observed feasible configs in current run for speed/TTFT
        normalization, model_max_context for context, and hardware-relative
        memory scoring. Never uses hardcoded 50, 1000, 2000, 32768, 6GB.
        """
        profile = self.state.profile
        weights = profile.weights

        # Collect bounds from all tested configs + this result for run-relative
        all_results = self.state.tested_configs + (
            [result] if result not in self.state.tested_configs else []
        )
        feasible = [r for r in all_results if r.passed]
        if not feasible:
            feasible = [result]

        # Extract speed bounds
        gen_vals = [
            r.get_avg_generation_tok_s() for r in feasible if r.get_avg_generation_tok_s() > 0
        ]
        prompt_vals = [r.get_avg_prompt_tok_s() for r in feasible if r.get_avg_prompt_tok_s() > 0]
        # Estimated TTFT: use estimated_ttft if available else ttft fallback
        ttft_vals = []
        for r in feasible:
            try:
                ttft_vals.append(
                    r.get_avg_estimated_ttft_ms()
                    if hasattr(r, "get_avg_estimated_ttft_ms")
                    else r.get_avg_ttft_ms()
                )
            except Exception:
                try:
                    ttft_vals.append(r.get_avg_ttft_ms())
                except Exception:
                    pass
        ttft_vals = [v for v in ttft_vals if v > 0]

        gen = result.get_avg_generation_tok_s()
        prompt = result.get_avg_prompt_tok_s()
        try:
            ttft = (
                result.get_avg_estimated_ttft_ms()
                if hasattr(result, "get_avg_estimated_ttft_ms")
                else result.get_avg_ttft_ms()
            )
        except Exception:
            ttft = result.get_avg_ttft_ms()
        quality = getattr(result, "_quality_obj", None)
        if quality and hasattr(quality, "overall"):
            quality_val = quality.overall
        elif isinstance(result.quality_score, (int, float)):
            quality_val = float(result.quality_score)
        else:
            quality_val = 1.0
        stability = result.stability_score
        ctx = result.context_length
        vram = result.peak_vram_gb

        # Run-relative normalization with zero-range safety
        def norm_higher(v, vals):
            if not vals or len(set(vals)) == 1:
                return 1.0
            mn, mx = min(vals), max(vals)
            rng = mx - mn
            if abs(rng) < 1e-9:
                return 1.0
            return max(0.0, min(1.0, (v - mn) / rng))

        def norm_lower(v, vals):
            if not vals or len(set(vals)) == 1:
                return 1.0
            mn, mx = min(vals), max(vals)
            rng = mx - mn
            if abs(rng) < 1e-9:
                return 1.0
            return max(0.0, min(1.0, (mx - v) / rng))

        norm_gen = norm_higher(gen, gen_vals)
        norm_prompt = norm_higher(prompt, prompt_vals)
        norm_ttft = norm_lower(ttft, ttft_vals)

        # Context: model-relative
        model_max = self.state.capabilities.max_context_length if self.state.capabilities else None
        norm_context = normalize_context(
            ctx, model_max, max([r.context_length for r in feasible]) if feasible else None
        )

        # Memory: hardware-relative
        total_vram = config.hardware.gpu_vram_gb
        # If not set, try to use estimated from capabilities as hint, but not as fixed constant
        # Otherwise fallback to run-relative
        if total_vram is None or total_vram == 0:
            # Fallback: run-relative lower is not always better, but we can use headroom concept with observed max
            observed_vrams = [r.peak_vram_gb for r in feasible if r.peak_vram_gb]
            if observed_vrams and vram and len(set(observed_vrams)) > 1:
                norm_vram = norm_lower(vram, observed_vrams)
            else:
                norm_vram = 0.5
        else:
            norm_vram = compute_memory_score(vram, None, total_vram, None)

        score = (
            weights.get("generation_speed", 0) * norm_gen
            + weights.get("prompt_processing", 0) * norm_prompt
            + weights.get("ttft", 0) * norm_ttft
            + weights.get("quality", 0) * quality_val
            + weights.get("stability", 0) * stability
            + weights.get("context_capacity", 0) * norm_context
            + weights.get("memory_efficiency", 0) * norm_vram
        )

        return max(0.0, min(1.0, score))

    def _compute_pareto_frontier(self, results: list[BenchmarkResult]) -> list[BenchmarkResult]:
        """Compute Pareto frontier for multi-objective optimization."""
        if not results:
            return []

        # Objectives: maximize quality, gen_speed, prompt_speed, context; minimize vram, ram
        frontier = []

        for r in results:
            if not r.passed:
                continue

            dominated = False
            for other in results:
                if not other.passed or other is r:
                    continue

                # Check if other dominates r
                other_better = (
                    (
                        other._quality_obj.overall
                        if hasattr(other, "_quality_obj")
                        else other.quality_score
                    )
                    >= (r._quality_obj.overall if hasattr(r, "_quality_obj") else r.quality_score)
                    and other.get_avg_generation_tok_s() >= r.get_avg_generation_tok_s()
                    and other.get_avg_prompt_tok_s() >= r.get_avg_prompt_tok_s()
                    and other.context_length >= r.context_length
                    and (other.peak_vram_gb or 0) <= (r.peak_vram_gb or float("inf"))
                )
                other_strictly_better = (
                    (
                        other._quality_obj.overall
                        if hasattr(other, "_quality_obj")
                        else other.quality_score
                    )
                    > (r._quality_obj.overall if hasattr(r, "_quality_obj") else r.quality_score)
                    or other.get_avg_generation_tok_s() > r.get_avg_generation_tok_s()
                    or other.get_avg_prompt_tok_s() > r.get_avg_prompt_tok_s()
                    or other.context_length > r.context_length
                    or (other.peak_vram_gb or 0) < (r.peak_vram_gb or float("inf"))
                )

                if other_better and other_strictly_better:
                    dominated = True
                    break

            if not dominated:
                frontier.append(r)

        return frontier

    def _median_result(self, results: list[BenchmarkResult]) -> BenchmarkResult:
        """Create a result with median metrics from multiple runs."""
        if not results:
            return results[0]

        base = results[0]
        # For simplicity, return the one with median generation speed
        speeds = [(r.get_avg_generation_tok_s(), r) for r in results]
        speeds.sort(key=lambda x: x[0])
        return speeds[len(speeds) // 2][1]

    def _compare_with_baseline(self, baseline: BenchmarkResult) -> None:
        """Compare best config with baseline using actual measurements."""
        if not self.state.best_config:
            return

        best = self.state.best_config
        try:
            gen_improvement = (
                (best.get_avg_generation_tok_s() - baseline.get_avg_generation_tok_s())
                / baseline.get_avg_generation_tok_s()
                if baseline.get_avg_generation_tok_s()
                else 0
            )
        except Exception:
            gen_improvement = 0
        self.state.best_result = best  # Store for result compilation

    def _should_skip_config(self, context: int, gpu_ratio: float) -> bool:
        """Heuristic to skip unlikely configurations - hardware-aware."""
        caps = self.state.capabilities
        if not caps:
            return False

        # Skip if context > max
        if caps.max_context_length and context > caps.max_context_length:
            return True

        # Skip if GPU ratio too high for estimated VRAM relative to available VRAM
        # Hardware-relative, not hardcoded 6GB
        total_vram = config.hardware.gpu_vram_gb
        if caps.estimated_vram_gb and total_vram:
            if gpu_ratio > 0.8 and caps.estimated_vram_gb > total_vram * 0.9:
                return True

        return False

    def _compile_result(self, baseline: BenchmarkResult | None) -> OptimizationResult:
        """Compile final optimization result with percentage changes from actual measurements."""
        best = self.state.best_config
        if not best:
            raise RuntimeError("No valid configuration found")

        recommended_config = LoadConfig(
            context_length=best.context_length,
            gpu_ratio=best.gpu_ratio,
            flash_attention=best.flash_attention,
            offload_kv_cache_to_gpu=best.kv_cache_gpu,
            eval_batch_size=best.eval_batch_size,
        )

        opt_time = (datetime.now() - self.state.start_time).total_seconds()

        speedup = None
        quality_change = None
        if (
            baseline
            and best.get_avg_generation_tok_s() > 0
            and baseline.get_avg_generation_tok_s() > 0
        ):
            speedup = (
                best.get_avg_generation_tok_s() - baseline.get_avg_generation_tok_s()
            ) / baseline.get_avg_generation_tok_s()
            # Quality change from actual measurements
            base_q = (
                baseline._quality_obj.overall
                if hasattr(baseline, "_quality_obj")
                else baseline.quality_score
            )
            best_q = (
                best._quality_obj.overall if hasattr(best, "_quality_obj") else best.quality_score
            )
            if (
                isinstance(base_q, (int, float))
                and isinstance(best_q, (int, float))
                and base_q != 0
            ):
                quality_change = (best_q - base_q) / base_q

        # Compute score breakdown for best with final bounds
        try:
            bounds = compute_norm_bounds(
                self.state.tested_configs,
                None,
                caps.max_context_length if self.state.capabilities else None,
            )
            # Need to handle BenchmarkResult vs ConfigurationResult for scoring
            # For engine, create temporary hardware stub
            breakdown = None
        except Exception:
            breakdown = None

        return OptimizationResult(
            model_id=self.state.model_id,
            profile=self.state.profile,
            recommended_config=recommended_config,
            recommended_context=best.context_length,
            best_result=best,
            pareto_frontier=self.state.pareto_frontier,
            all_results=self.state.tested_configs,
            optimization_time_s=opt_time,
            speedup_vs_baseline=speedup,
            quality_change=quality_change,
            score_breakdown=breakdown,
        )
