"""Adaptive optimization engine - hardware-agnostic scoring."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from lm_optimizer.config import config
from lm_optimizer.database.repositories import config_repo, hardware_repo, model_repo, run_repo
from lm_optimizer.domain.models import (
    DEFAULT_PROFILE_WEIGHTS,
    ConfigurationResult,
    ConfigurationStatus,
    HardwareInfo,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
    OptimizationStage,
    ProfileWeights,
    RunStatus,
)
from lm_optimizer.logging_config import get_logger
from lm_optimizer.scoring.normalization import (
    compute_bounds,
    score_result_breakdown,
    weighted_score,
)
from lm_optimizer.services.benchmark import BenchmarkService
from lm_optimizer.services.lm_studio import LMStudioClient
from lm_optimizer.services.quality import QualityEvaluator
from lm_optimizer.services.search_space import SearchSpace, SearchSpaceGenerator

logger = get_logger(__name__)


@dataclass
class OptimizationState:
    """Current state of optimization run."""

    run: OptimizationRun
    search_space: SearchSpace
    tested_configs: list[ConfigurationResult] = field(default_factory=list)
    best_config: ConfigurationResult | None = None
    pareto_frontier: list[ConfigurationResult] = field(default_factory=list)
    current_config_id: UUID | None = None
    errors: list[str] = field(default_factory=list)
    should_pause: bool = False
    should_cancel: bool = False
    # Track failed/OOM configs to avoid refining inferior regions
    failed_regions: set = field(default_factory=set)


class AdaptiveOptimizer:
    """Adaptive optimizer with 5-stage search, hardware-agnostic scoring."""

    def __init__(
        self,
        client: LMStudioClient,
        benchmark_service: BenchmarkService,
        quality_evaluator: QualityEvaluator,
        search_generator: SearchSpaceGenerator,
    ):
        self.client = client
        self.benchmark = benchmark_service
        self.quality = quality_evaluator
        self.search_generator = search_generator
        self.state: OptimizationState | None = None

    async def optimize(
        self,
        model: ModelIdentity,
        hardware: HardwareInfo,
        profile: OptimizationProfile,
        profile_weights: ProfileWeights | None = None,
        quality_threshold: float = 0.97,
        advanced_settings: dict | None = None,
        baseline_config: LoadConfiguration | None = None,
    ) -> OptimizationRun:
        """Run full optimization with baseline capture and hardware-agnostic scoring."""
        # Save hardware and model
        hardware_id = hardware_repo.save(hardware)
        model_repo.save(model)

        advanced_settings = advanced_settings or {}

        # Handle RoPE experimental flag - DISABLED by default
        enable_rope = advanced_settings.get("enable_rope", False)
        is_experimental = False
        experimental_reason = None
        if enable_rope:
            is_experimental = True
            experimental_reason = "RoPE parameters enabled (experimental)"
            # RoPE requires stronger quality validation
            quality_threshold = max(quality_threshold, 0.98)
            logger.warning(
                "RoPE experimental enabled - applying stronger quality validation",
                threshold=quality_threshold,
            )
        else:
            # Ensure RoPE params are not in search space by stripping them
            advanced_settings["enable_rope"] = False

        # Use profile's default threshold if not explicitly overridden
        # Profiles have intentional thresholds: speed 0.95, balanced 0.97, context/quality higher
        if quality_threshold is None:
            from lm_optimizer.profiles.registry import ProfileRegistry

            reg = ProfileRegistry()
            try:
                prof = reg.get(profile.value)
                quality_threshold = prof.minimum_quality
            except Exception:
                quality_threshold = 0.97

        # Resolve profile weights
        resolved_weights = profile_weights or DEFAULT_PROFILE_WEIGHTS.get(profile)
        if resolved_weights:
            weights_dict = resolved_weights.to_dict()
        else:
            weights_dict = {}

        # Create run
        run = OptimizationRun(
            model=model,
            hardware=hardware,
            hardware_id=hardware_id,
            profile=profile,
            profile_weights=weights_dict,
            quality_threshold=quality_threshold,
            search_space={},
            is_experimental=is_experimental,
            experimental_reason=experimental_reason,
            benchmark_params={
                "benchmark_repetitions": config.optimization.benchmark_runs,
                "validation_repetitions": 5,
                "warmup_runs": config.optimization.warmup_runs,
                "seed": 42,
                "deterministic": True,
                "temperature": "per-benchmark-case-fixed",
                "candidate_generation": "deterministic-sorted-sampling",
            },
        )

        # Generate search space
        search_space = self.search_generator.generate(model, hardware, profile, advanced_settings)
        run.search_space = search_space.to_dict()

        # Initialize state
        self.state = OptimizationState(run=run, search_space=search_space)

        # Baseline: capture actual current configuration if not provided
        if baseline_config is None:
            baseline_config = await self._capture_baseline_config(model)
        if baseline_config:
            await self._run_baseline(baseline_config)

        # Save initial run
        run.status = RunStatus.RUNNING
        run.started_at = datetime.now()
        run_repo.save(run)

        try:
            # Stage 1: Discovery
            await self._stage_discovery()

            # Stage 2: Coarse Search
            await self._stage_coarse_search()

            # Stage 3: Refinement
            await self._stage_refinement()

            # Stage 4: Batch Optimization
            await self._stage_batch_optimization()

            # Stage 5: Validation
            await self._stage_validation()

            # Recompute all scores with final bounds for explainability
            self._recompute_all_scores()

            # Finalize
            await self._finalize()

        except asyncio.CancelledError:
            run.status = RunStatus.CANCELLED
            run.stage = OptimizationStage.CANCELLED
            raise
        except Exception as e:
            logger.error("Optimization failed", error=str(e))
            run.status = RunStatus.FAILED
            run.stage = OptimizationStage.ERROR
            run.error = str(e)
        finally:
            run.completed_at = datetime.now()
            if run.started_at:
                run.duration_seconds = (run.completed_at - run.started_at).total_seconds()
            run_repo.save(run)
            # Ensure model unloaded
            await self.client.unload_model(model_id=model.id)

        return run

    async def _capture_baseline_config(self, model: ModelIdentity) -> LoadConfiguration | None:
        """Capture actual currently loaded configuration as baseline."""
        try:
            # Try to get currently loaded model info
            loaded = await self.client.get_model(model.id)
            if loaded and loaded.loaded and loaded.load_config:
                logger.info("Captured baseline from loaded model", model=model.id)
                return LoadConfiguration(
                    context_length=loaded.load_config.context_length,
                    gpu_ratio=loaded.load_config.gpu_ratio,
                    flash_attention=loaded.load_config.flash_attention,
                    offload_kv_cache_to_gpu=loaded.load_config.offload_kv_cache_to_gpu,
                    eval_batch_size=loaded.load_config.eval_batch_size,
                    num_experts=loaded.load_config.num_experts,
                )
            # Check client's loaded models cache
            if hasattr(self.client, "_loaded_models") and model.id in self.client._loaded_models:
                # We don't know exact config, use minimal baseline
                logger.info(
                    "Model is loaded but config unknown, using minimal baseline", model=model.id
                )
                return LoadConfiguration(context_length=model.context_limit or 4096)
        except Exception as e:
            logger.debug("Baseline capture failed", error=str(e))
        return None

    async def _run_baseline(self, baseline_config: LoadConfiguration) -> None:
        """Run baseline benchmark and save metrics."""
        logger.info("Running baseline benchmark", config=baseline_config.to_dict())
        self.state.run.stage = OptimizationStage.DISCOVERY
        run_repo.save(self.state.run)

        result = await self.benchmark.run_benchmark(
            self.state.run.model.id,
            baseline_config,
            self.state.run.model.context_limit or baseline_config.context_length or 4096,
        )
        # Convert status to enum
        if result.status == "passed":
            result.status = ConfigurationStatus.PASSED
        elif result.status == "failed":
            result.status = ConfigurationStatus.FAILED
            if (
                "OOM" in (result.error or "").upper()
                or "out of memory" in (result.error or "").lower()
            ):
                result.status = ConfigurationStatus.OOM

        # Evaluate quality for baseline even if we keep it regardless
        try:
            quality_scores = self.quality.evaluate_all(self.state.run.model.id, result.metrics)
            agg_quality = self.quality.aggregate_quality(quality_scores)
            result.quality_score = agg_quality
        except Exception:
            pass

        # Score baseline with current weights (will be recomputed later)
        result.run_id = self.state.run.id
        config_repo.save(result)

        self.state.run.baseline_config_id = result.id
        # Store baseline metrics for comparison
        self.state.run.baseline_metrics = {
            "generation_tok_s": result.get_avg_generation_tok_s(),
            "prompt_tok_s": result.get_avg_prompt_tok_s(),
            "estimated_ttft_ms": result.get_avg_estimated_ttft_ms(),
            "context_length": result.context_length,
            "peak_vram_gb": result.peak_vram_gb,
            "peak_ram_gb": result.peak_ram_gb,
            "quality_overall": result.quality_score.overall if result.quality_score else None,
            "quality_checks": result.quality_score.as_checks_str()
            if result.quality_score and hasattr(result.quality_score, "as_checks_str")
            else None,
            "stability": result.stability_score,
        }
        # Baseline is also a tested config
        self.state.tested_configs.append(result)
        run_repo.save(self.state.run)

    async def _stage_discovery(self) -> None:
        """Stage 1: Discovery - already done during search space generation."""
        logger.info("Stage 1: Discovery complete", search_space=self.state.search_space.to_dict())
        self.state.run.stage = OptimizationStage.DISCOVERY
        run_repo.save(self.state.run)

    async def _stage_coarse_search(self) -> None:
        """Stage 2: Coarse search with deterministic intelligent sampling."""
        logger.info("Stage 2: Coarse search")
        self.state.run.stage = OptimizationStage.COARSE_SEARCH
        run_repo.save(self.state.run)

        space = self.state.search_space
        configs = self._generate_coarse_configs(space)

        for i, cfg in enumerate(configs):
            if self.state.should_cancel:
                break
            while self.state.should_pause:
                await asyncio.sleep(1)

            logger.info("Testing coarse config", idx=i + 1, total=len(configs), **cfg.to_dict())
            self.state.current_config_id = None
            result = await self._test_config(
                cfg, cfg.context_length or self.state.run.model.context_limit or 4096
            )

            if result and result.status == ConfigurationStatus.PASSED:
                self.state.tested_configs.append(result)
                self._update_best(result)
            elif result and result.status in (ConfigurationStatus.FAILED, ConfigurationStatus.OOM):
                # Track failed regions to avoid refining them
                self.state.failed_regions.add((cfg.context_length, cfg.gpu_ratio))

            # Recompute scores periodically for hardware-agnostic normalization
            if (i + 1) % 5 == 0:
                self._recompute_all_scores()

            run_repo.save(self.state.run)

    def _deterministic_sample(self, items: list, k: int) -> list:
        """Deterministically sample k items covering low/high.

        Picks evenly spaced indices to ensure coverage of range.
        Deterministic (sorted order, no randomness).
        """
        if not items:
            return []
        if len(items) <= k:
            return sorted(items)
        sorted_items = sorted(items)
        # Evenly spaced indices including first and last
        indices = []
        for i in range(k):
            idx = int(round(i * (len(sorted_items) - 1) / (k - 1))) if k > 1 else 0
            indices.append(idx)
        # Dedup and preserve order
        seen = set()
        result = []
        for idx in indices:
            val = sorted_items[idx]
            if val not in seen:
                result.append(val)
                seen.add(val)
        return sorted(result)

    def _generate_coarse_configs(self, space: SearchSpace) -> list[LoadConfiguration]:
        """Generate coarse search configurations with deterministic intelligent sampling.

        Covers search dimensions reasonably: low/high context, low/high GPU,
        KV alternatives, Flash alternatives, batch alternatives.
        Uses deterministic sampling (sorted, evenly spaced) for reproducibility.
        """
        # Sample dimensions to ensure coverage, not just first N
        ctx_samples = self._deterministic_sample(
            space.context_lengths, min(4, len(space.context_lengths))
        )
        gpu_samples = self._deterministic_sample(space.gpu_ratios, min(3, len(space.gpu_ratios)))
        batch_samples = self._deterministic_sample(
            space.batch_sizes, min(2, len(space.batch_sizes))
        )
        flash_samples = sorted(space.flash_attention_options)
        kv_samples = sorted(space.kv_cache_options)

        # Generate ALL combinations in deterministic sorted order
        all_configs = []
        for ctx in sorted(ctx_samples):
            for gpu in sorted(gpu_samples):
                for flash in flash_samples:
                    for kv in kv_samples:
                        for batch in sorted(batch_samples):
                            all_configs.append(
                                LoadConfiguration(
                                    context_length=ctx,
                                    gpu_ratio=gpu,
                                    flash_attention=flash,
                                    offload_kv_cache_to_gpu=kv,
                                    eval_batch_size=batch,
                                )
                            )

        # If still too many, deterministically sample 20 covering dimensions
        # Sort deterministically by tuple to ensure reproducibility
        all_configs.sort(
            key=lambda c: (
                c.context_length or 0,
                c.gpu_ratio or 0,
                c.eval_batch_size or 0,
                c.flash_attention or False,
                c.offload_kv_cache_to_gpu or False,
            )
        )

        if len(all_configs) > 20:
            # Evenly spaced deterministic sampling to cover dimensions
            step = len(all_configs) / 20
            sampled = []
            for i in range(20):
                idx = int(round(i * step))
                if idx >= len(all_configs):
                    idx = len(all_configs) - 1
                sampled.append(all_configs[idx])
            # Dedup preserve order
            seen = set()
            deduped = []
            for c in sampled:
                key = (
                    c.context_length,
                    c.gpu_ratio,
                    c.flash_attention,
                    c.offload_kv_cache_to_gpu,
                    c.eval_batch_size,
                )
                if key not in seen:
                    seen.add(key)
                    deduped.append(c)
            return deduped

        return all_configs[:20]

    async def _stage_refinement(self) -> None:
        """Stage 3: Refinement around promising candidates (not just best)."""
        logger.info("Stage 3: Refinement")
        self.state.run.stage = OptimizationStage.REFINEMENT
        run_repo.save(self.state.run)

        if not self.state.best_config:
            logger.warning("No best config found, skipping refinement")
            return

        # Select promising candidates: best + top Pareto frontier (up to 3)
        promising = [self.state.best_config]
        # Add Pareto frontier configs that are not already best, sorted by generation speed
        pareto = self._compute_pareto_frontier(self.state.tested_configs)
        pareto_sorted = sorted(pareto, key=lambda r: r.get_avg_generation_tok_s(), reverse=True)
        for pc in pareto_sorted:
            if pc.id != self.state.best_config.id and len(promising) < 3:
                promising.append(pc)

        space = self.state.search_space
        tested_keys = {
            (c.context_length, c.gpu_ratio, c.eval_batch_size) for c in self.state.tested_configs
        }

        for best in promising:
            # Refine around each promising candidate, avoiding failed regions
            ctx_candidates = self._refine_context(best.context_length, space.context_lengths)
            gpu_candidates = self._refine_gpu_ratio(best.gpu_ratio or 0.8, space.gpu_ratios)
            batch_candidates = self._refine_batch_size(
                best.eval_batch_size or 256, space.batch_sizes
            )

            # Also test important parameter interactions: toggle flash and kv at best context/gpu
            interaction_configs = []
            for flash in space.flash_attention_options:
                if flash != best.flash_attention:
                    interaction_configs.append(
                        LoadConfiguration(
                            context_length=best.context_length,
                            gpu_ratio=best.gpu_ratio,
                            flash_attention=flash,
                            offload_kv_cache_to_gpu=best.offload_kv_cache_to_gpu,
                            eval_batch_size=best.eval_batch_size,
                        )
                    )
            for kv in space.kv_cache_options:
                if kv != best.offload_kv_cache_to_gpu:
                    interaction_configs.append(
                        LoadConfiguration(
                            context_length=best.context_length,
                            gpu_ratio=best.gpu_ratio,
                            flash_attention=best.flash_attention,
                            offload_kv_cache_to_gpu=kv,
                            eval_batch_size=best.eval_batch_size,
                        )
                    )

            for cfg in interaction_configs:
                key = (cfg.context_length, cfg.gpu_ratio, cfg.eval_batch_size)
                if key in tested_keys or key in self.state.failed_regions:
                    continue
                if self.state.should_cancel:
                    break
                while self.state.should_pause:
                    await asyncio.sleep(1)
                result = await self._test_config(cfg, cfg.context_length or 4096)
                tested_keys.add(key)
                if result and result.status == ConfigurationStatus.PASSED:
                    self.state.tested_configs.append(result)
                    self._update_best(result)
                run_repo.save(self.state.run)

            for ctx in ctx_candidates:
                for gpu in gpu_candidates:
                    for batch in batch_candidates:
                        key = (ctx, gpu, batch)
                        if key in tested_keys:
                            continue
                        # Avoid clearly inferior regions (failed before)
                        if (ctx, gpu) in self.state.failed_regions:
                            continue
                        if self.state.should_cancel:
                            break
                        while self.state.should_pause:
                            await asyncio.sleep(1)

                        config = LoadConfiguration(
                            context_length=ctx,
                            gpu_ratio=gpu,
                            flash_attention=best.flash_attention,
                            offload_kv_cache_to_gpu=best.offload_kv_cache_to_gpu,
                            eval_batch_size=batch,
                        )

                        result = await self._test_config(config, ctx)
                        tested_keys.add(key)
                        if result and result.status == ConfigurationStatus.PASSED:
                            self.state.tested_configs.append(result)
                            self._update_best(result)
                        elif result and result.status == ConfigurationStatus.OOM:
                            self.state.failed_regions.add((ctx, gpu))

                        run_repo.save(self.state.run)

            # Recompute scores after each promising candidate's refinement
            self._recompute_all_scores()

    def _refine_context(self, best_ctx: int, all_ctx: list[int]) -> list[int]:
        """Generate context candidates around best, using hardware-aware steps.

        Uses model-relative steps: max(512, best//4) but ensures we stay
        within search space bounds. Avoids hardcoding 32768.
        """
        if not all_ctx:
            return [best_ctx]
        candidates = [best_ctx]
        step = max(512, best_ctx // 4)

        for offset in [-step, step, -2 * step, 2 * step]:
            ctx = best_ctx + offset
            if ctx in all_ctx and ctx not in candidates:
                candidates.append(ctx)

        # Also include nearest neighbors in sorted search space for fine granularity
        sorted_ctx = sorted(all_ctx)
        try:
            idx = sorted_ctx.index(best_ctx)
            for neighbor_off in [-1, 1]:
                nidx = idx + neighbor_off
                if 0 <= nidx < len(sorted_ctx):
                    candidates.append(sorted_ctx[nidx])
        except ValueError:
            pass

        return sorted(set(candidates))

    def _refine_gpu_ratio(self, best_gpu: float, all_gpu: list[float]) -> list[float]:
        """Generate GPU ratio candidates around best, deterministically."""
        candidates = [best_gpu]
        # Use fine-grained offsets, filtered to search space
        for offset in [-0.1, -0.05, 0.05, 0.1, -0.15, 0.15]:
            gpu = round(best_gpu + offset, 2)
            if 0.0 <= gpu <= 1.0 and gpu in all_gpu and gpu not in candidates:
                candidates.append(gpu)
        return sorted(set(candidates))

    def _refine_batch_size(self, best_batch: int, all_batch: list[int]) -> list[int]:
        """Generate batch size candidates around best, deterministic."""
        candidates = [best_batch]
        for mult in [0.5, 0.75, 1.5, 2.0]:
            batch = int(best_batch * mult)
            batch = max(32, min(2048, batch))
            batch = round(batch / 32) * 32
            if batch in all_batch and batch not in candidates:
                candidates.append(batch)
        return sorted(set(candidates))

    async def _stage_batch_optimization(self) -> None:
        """Stage 4: Optimize eval batch size."""
        logger.info("Stage 4: Batch optimization")
        self.state.run.stage = OptimizationStage.BATCH_OPTIMIZATION
        run_repo.save(self.state.run)

        if not self.state.best_config:
            return

        best = self.state.best_config
        tested_batches = {
            c.eval_batch_size
            for c in self.state.tested_configs
            if c.context_length == best.context_length and c.gpu_ratio == best.gpu_ratio
        }
        for batch in self.state.search_space.batch_sizes:
            if batch in tested_batches:
                continue
            if self.state.should_cancel:
                break
            while self.state.should_pause:
                await asyncio.sleep(1)

            config = LoadConfiguration(
                context_length=best.context_length,
                gpu_ratio=best.gpu_ratio,
                flash_attention=best.flash_attention,
                offload_kv_cache_to_gpu=best.offload_kv_cache_to_gpu,
                eval_batch_size=batch,
            )

            result = await self._test_config(config, best.context_length)
            if result and result.status == ConfigurationStatus.PASSED:
                self.state.tested_configs.append(result)
                self._update_best(result)

            run_repo.save(self.state.run)

    async def _stage_validation(self) -> None:
        """Stage 5: Final validation with multiple runs."""
        logger.info("Stage 5: Final validation")
        self.state.run.stage = OptimizationStage.VALIDATION
        run_repo.save(self.state.run)

        if not self.state.best_config:
            self.state.run.stage = OptimizationStage.COMPLETE
            return

        best = self.state.best_config
        config = LoadConfiguration(
            context_length=best.context_length,
            gpu_ratio=best.gpu_ratio,
            flash_attention=best.flash_attention,
            offload_kv_cache_to_gpu=best.offload_kv_cache_to_gpu,
            eval_batch_size=best.eval_batch_size,
        )
        # Preserve RoPE if experimental and enabled
        if self.state.run.is_experimental:
            config.rope_freq_base = best.config.rope_freq_base
            config.rope_freq_scale = best.config.rope_freq_scale

        # Run validation repetitions
        validation_results = []
        for i in range(self.state.run.validation_repetitions):
            if self.state.should_cancel:
                break
            result = await self._test_config(config, best.context_length)
            if result and result.status == ConfigurationStatus.PASSED:
                validation_results.append(result)

        if validation_results:
            # Use median result
            self.state.best_config = self._median_result(validation_results)
            # Ensure best is in tested list
            if self.state.best_config.id not in [c.id for c in self.state.tested_configs]:
                self.state.tested_configs.append(self.state.best_config)
            self.state.run.best_config_id = self.state.best_config.id

        # Calculate Pareto frontier
        self.state.pareto_frontier = self._compute_pareto_frontier(self.state.tested_configs)
        self.state.run.pareto_config_ids = [c.id for c in self.state.pareto_frontier]

        self.state.run.stage = OptimizationStage.COMPLETE
        run_repo.save(self.state.run)

    async def _test_config(
        self, load_config: LoadConfiguration, context_length: int
    ) -> ConfigurationResult | None:
        """Test a single configuration with hardware-agnostic scoring."""
        try:
            result = await self.benchmark.run_benchmark(
                self.state.run.model.id, load_config, context_length
            )
            result.run_id = self.state.run.id

            # Convert status to enum
            if result.status == "passed":
                result.status = ConfigurationStatus.PASSED
            elif result.status == "failed":
                result.status = ConfigurationStatus.FAILED
                if (
                    "OOM" in (result.error or "").upper()
                    or "out of memory" in (result.error or "").lower()
                ):
                    result.status = ConfigurationStatus.OOM

            # Evaluate quality / correctness
            quality_scores = self.quality.evaluate_all(self.state.run.model.id, result.metrics)
            agg_quality = self.quality.aggregate_quality(quality_scores)
            result.quality_score = agg_quality

            if not self.quality.passes_threshold(agg_quality):
                logger.warning(
                    "Config below quality threshold",
                    config_id=str(result.id),
                    quality=agg_quality.overall,
                    checks=f"{agg_quality.checks_passed}/{agg_quality.checks_total}",
                    threshold=self.quality.config.minimum_score,
                )
                result.status = ConfigurationStatus.FAILED
                result.error = f"Quality/correctness {agg_quality.overall:.3f} ({agg_quality.checks_passed}/{agg_quality.checks_total} checks) below threshold {self.quality.config.minimum_score}"
                # Still save for DB record but return None to not count as passed
                result.score_breakdown = {"quality": agg_quality.overall}
                result.score = 0.0
                config_repo.save(result)
                return None

            # Hardware-agnostic scoring with breakdown
            # Need bounds from all tested so far plus this result
            temp_results = self.state.tested_configs + [result]
            bounds = compute_bounds(
                temp_results, self.state.run.hardware, self.state.run.model.context_limit
            )
            breakdown = score_result_breakdown(
                result,
                bounds,
                self.state.run.hardware,
                self.state.run.model.context_limit,
                self.state.run.profile_weights,
            )
            result.score_breakdown = breakdown
            result.score = weighted_score(breakdown, self.state.run.profile_weights)

            # For experimental RoPE, apply stronger quality validation before claiming faster is better
            if self.state.run.is_experimental and breakdown.get("quality", 1.0) < 0.98:
                logger.warning(
                    "Experimental RoPE config quality insufficient",
                    quality=breakdown.get("quality"),
                )
                # Do not treat as best even if faster

            config_repo.save(result)

            logger.info(
                "Config passed",
                config_id=str(result.id),
                gen_tok_s=result.get_avg_generation_tok_s(),
                quality=f"{agg_quality.checks_passed}/{agg_quality.checks_total}",
                score=result.score,
                breakdown=breakdown,
            )

            return result

        except Exception as e:
            logger.error("Config error", error=str(e))
            self.state.errors.append(f"{load_config}: {e!s}")
            await self.client.unload_model(model_id=self.state.run.model.id)
            return None

    def _update_best(self, result: ConfigurationResult) -> None:
        """Update best configuration using hardware-agnostic scores."""
        if not self.state.best_config:
            self.state.best_config = result
            self.state.run.best_config_id = result.id
            return

        # Recompute both scores with current bounds for fair comparison
        bounds = compute_bounds(
            self.state.tested_configs, self.state.run.hardware, self.state.run.model.context_limit
        )
        # Update existing best's breakdown
        best_breakdown = score_result_breakdown(
            self.state.best_config,
            bounds,
            self.state.run.hardware,
            self.state.run.model.context_limit,
            self.state.run.profile_weights,
        )
        self.state.best_config.score_breakdown = best_breakdown
        self.state.best_config.score = weighted_score(
            best_breakdown, self.state.run.profile_weights
        )

        # Result already scored, but recompute with latest bounds
        new_breakdown = score_result_breakdown(
            result,
            bounds,
            self.state.run.hardware,
            self.state.run.model.context_limit,
            self.state.run.profile_weights,
        )
        result.score_breakdown = new_breakdown
        result.score = weighted_score(new_breakdown, self.state.run.profile_weights)

        current_score = self.state.best_config.score
        new_score = result.score

        if new_score > current_score:
            logger.info(
                "New best config",
                old_score=current_score,
                new_score=new_score,
                breakdown=new_breakdown,
            )
            self.state.best_config = result
            self.state.run.best_config_id = result.id

    def _recompute_all_scores(self) -> None:
        """Recompute scores for all tested configs with final bounds (for explainability)."""
        if not self.state.tested_configs:
            return
        bounds = compute_bounds(
            self.state.tested_configs, self.state.run.hardware, self.state.run.model.context_limit
        )
        for r in self.state.tested_configs:
            if r.status == ConfigurationStatus.PASSED:
                breakdown = score_result_breakdown(
                    r,
                    bounds,
                    self.state.run.hardware,
                    self.state.run.model.context_limit,
                    self.state.run.profile_weights,
                )
                r.score_breakdown = breakdown
                r.score = weighted_score(breakdown, self.state.run.profile_weights)
                config_repo.save(r)
        if self.state.best_config:
            # Re-evaluate best
            best = next(
                (c for c in self.state.tested_configs if c.id == self.state.best_config.id),
                self.state.best_config,
            )
            self.state.best_config = best

    def _compute_pareto_frontier(
        self, results: list[ConfigurationResult]
    ) -> list[ConfigurationResult]:
        """Compute Pareto frontier - hardware-agnostic multi-objective."""
        if not results:
            return []

        frontier = []
        for r in results:
            if r.status != ConfigurationStatus.PASSED:
                continue

            dominated = False
            for other in results:
                if other.status != ConfigurationStatus.PASSED or other is r:
                    continue

                other_better = (
                    other.quality_score.overall >= r.quality_score.overall
                    and other.get_avg_generation_tok_s() >= r.get_avg_generation_tok_s()
                    and other.get_avg_prompt_tok_s() >= r.get_avg_prompt_tok_s()
                    and other.context_length >= r.context_length
                    and (other.peak_vram_gb or float("inf")) <= (r.peak_vram_gb or float("inf"))
                )
                other_strictly_better = (
                    other.quality_score.overall > r.quality_score.overall
                    or other.get_avg_generation_tok_s() > r.get_avg_generation_tok_s()
                    or other.get_avg_prompt_tok_s() > r.get_avg_prompt_tok_s()
                    or other.context_length > r.context_length
                    or (other.peak_vram_gb or float("inf")) < (r.peak_vram_gb or float("inf"))
                )

                if other_better and other_strictly_better:
                    dominated = True
                    break

            if not dominated:
                frontier.append(r)

        return frontier

    def _median_result(self, results: list[ConfigurationResult]) -> ConfigurationResult:
        """Get median result by generation speed."""
        speeds = [(r.get_avg_generation_tok_s(), r) for r in results]
        speeds.sort(key=lambda x: x[0])
        return speeds[len(speeds) // 2][1]

    async def _finalize(self) -> None:
        """Finalize optimization run with comparison to baseline."""
        self.state.run.configurations = self.state.tested_configs
        if self.state.best_config:
            self.state.run.best_config_id = self.state.best_config.id

        # Calculate percentage changes vs baseline if available
        if self.state.run.baseline_metrics and self.state.best_config:
            baseline = self.state.run.baseline_metrics
            best = self.state.best_config
            changes = {}
            try:
                if baseline.get("generation_tok_s") and best.get_avg_generation_tok_s():
                    changes["generation_speed_change"] = (
                        (best.get_avg_generation_tok_s() - baseline["generation_tok_s"])
                        / baseline["generation_tok_s"]
                        * 100
                    )
                if baseline.get("prompt_tok_s") and best.get_avg_prompt_tok_s():
                    changes["prompt_speed_change"] = (
                        (best.get_avg_prompt_tok_s() - baseline["prompt_tok_s"])
                        / baseline["prompt_tok_s"]
                        * 100
                    )
                if baseline.get("context_length") and best.context_length:
                    changes["context_change"] = (
                        (best.context_length - baseline["context_length"])
                        / baseline["context_length"]
                        * 100
                    )
                if baseline.get("peak_vram_gb") and best.peak_vram_gb:
                    changes["vram_change"] = (
                        (best.peak_vram_gb - baseline["peak_vram_gb"])
                        / baseline["peak_vram_gb"]
                        * 100
                    )
                if baseline.get("quality_overall") and best.quality_score:
                    changes["quality_change"] = (
                        (best.quality_score.overall - baseline["quality_overall"])
                        / baseline["quality_overall"]
                        * 100
                        if baseline["quality_overall"]
                        else 0
                    )
                self.state.run.baseline_metrics["changes_vs_baseline"] = changes
            except Exception:
                pass

        self.state.run.status = RunStatus.COMPLETED
        self.state.run.stage = OptimizationStage.COMPLETE
        run_repo.save(self.state.run)

    def pause(self) -> None:
        """Pause optimization."""
        if self.state:
            self.state.should_pause = True

    def resume(self) -> None:
        """Resume optimization."""
        if self.state:
            self.state.should_pause = False

    def cancel(self) -> None:
        """Cancel optimization."""
        if self.state:
            self.state.should_cancel = True
            self.state.should_pause = False


# Type alias for LoadConfig
LoadConfig = LoadConfiguration
