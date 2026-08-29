"""Benchmark runner for executing tests and collecting metrics."""

import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from lm_optimizer.api.client import ChatMessage, LMStudioClient, LoadConfig
from lm_optimizer.benchmark.suite import BenchmarkCase, create_benchmark_cases
from lm_optimizer.config import config
from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkMetrics:
    """Metrics from a single benchmark run.

    estimated_ttft_ms is ESTIMATED (total_time *0.1) because LM Studio
    API does not expose streaming TTFT via non-streaming endpoint.
    """

    test_name: str
    category: str
    success: bool
    load_time_ms: float = 0.0
    estimated_ttft_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_processing_ms: float = 0.0
    generation_ms: float = 0.0
    prompt_tok_s: float = 0.0
    generation_tok_s: float = 0.0
    error: str | None = None
    output_text: str = ""

    @property
    def ttft_ms(self) -> float:
        return self.estimated_ttft_ms

    @ttft_ms.setter
    def ttft_ms(self, value: float) -> None:
        self.estimated_ttft_ms = value


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a configuration."""

    config_id: str
    model_id: str
    load_config: LoadConfig
    context_length: int
    gpu_ratio: float | None
    flash_attention: bool | None
    kv_cache_gpu: bool | None
    eval_batch_size: int | None
    num_experts: int | None
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: list[BenchmarkMetrics] = field(default_factory=list)
    peak_vram_gb: float | None = None
    peak_ram_gb: float | None = None
    stability_score: float = 1.0
    quality_score: float = 1.0
    passed: bool = True
    failure_reason: str | None = None


class BenchmarkRunner:
    """Runs benchmarks against LM Studio."""

    def __init__(self, client: LMStudioClient):
        self.client = client
        self.current_config_id: str | None = None

    async def run_benchmark(
        self,
        model_id: str,
        load_config: LoadConfig,
        context_length: int,
        runs: int | None = None,
        warmup_runs: int | None = None,
    ) -> BenchmarkResult:
        """Run full benchmark suite for a configuration."""
        config_id = str(uuid.uuid4())[:8]
        self.current_config_id = config_id

        runs = runs or config.optimization.benchmark_runs
        warmup_runs = warmup_runs or config.optimization.warmup_runs

        logger.info(
            "Starting benchmark", config_id=config_id, model=model_id, context=context_length
        )

        # Load model with configuration
        load_start = time.perf_counter()
        load_result = await self.client.load_model(model_id, load_config)
        load_time_ms = (time.perf_counter() - load_start) * 1000

        if not load_result.success:
            return BenchmarkResult(
                config_id=config_id,
                model_id=model_id,
                load_config=load_config,
                context_length=context_length,
                gpu_ratio=load_config.gpu_ratio,
                flash_attention=load_config.flash_attention,
                kv_cache_gpu=load_config.offload_kv_cache_to_gpu,
                eval_batch_size=load_config.eval_batch_size,
                num_experts=load_config.num_experts,
                passed=False,
                failure_reason=f"Model load failed: {load_result.error}",
            )

        # Verify actual loaded configuration
        actual_config = load_result.load_config or load_config

        # Create benchmark cases for this context length
        cases = create_benchmark_cases(context_length)

        all_metrics: list[BenchmarkMetrics] = []

        try:
            # Warmup runs
            for i in range(warmup_runs):
                logger.debug("Warmup run", run=i + 1, total=warmup_runs)
                for case in cases:
                    await self._run_single_case(model_id, case)

            # Measured runs
            for run_idx in range(runs):
                logger.debug("Measured run", run=run_idx + 1, total=runs)
                for case in cases:
                    metrics = await self._run_single_case(model_id, case)
                    all_metrics.append(metrics)

            # Aggregate metrics
            result = self._aggregate_results(
                config_id=config_id,
                model_id=model_id,
                load_config=load_config,
                actual_config=actual_config,
                context_length=context_length,
                metrics=all_metrics,
                load_time_ms=load_time_ms,
            )

        finally:
            # Always unload model
            await self.client.unload_model(model_id=model_id)

        logger.info(
            "Benchmark complete",
            config_id=config_id,
            passed=result.passed,
            gen_tok_s=result.get_avg_generation_tok_s(),
        )

        return result

    async def _run_single_case(self, model_id: str, case: BenchmarkCase) -> BenchmarkMetrics:
        """Run a single benchmark case."""
        start_time = time.perf_counter()
        first_token_time = None

        try:
            # Use streaming to measure TTFT
            messages = [ChatMessage(role="user", content=case.prompt)]

            # For TTFT measurement, we need to stream
            # But LM Studio API might not expose streaming in /api/v1/chat/completions
            # Fall back to non-streaming and estimate
            response = await self.client.chat_completion(
                model=model_id,
                messages=messages,
                temperature=case.temperature,
                max_tokens=case.max_tokens,
                seed=42,  # Fixed seed for reproducibility
                stop=case.stop_sequences,
            )

            total_time_ms = (time.perf_counter() - start_time) * 1000

            # Extract metrics from response
            usage = response.usage
            choice = response.choices[0] if response.choices else None
            output_text = choice.message.content if choice else ""

            # Estimated TTFT - see class docstring
            estimated_ttft_ms = total_time_ms * 0.1 if usage.prompt_tokens > 0 else total_time_ms

            prompt_processing_ms = total_time_ms - (
                usage.completion_tokens / max(usage.completion_tokens, 1) * total_time_ms
            )
            generation_ms = total_time_ms - prompt_processing_ms

            prompt_tok_s = (
                (usage.prompt_tokens / (prompt_processing_ms / 1000))
                if prompt_processing_ms > 0
                else 0
            )
            generation_tok_s = (
                (usage.completion_tokens / (generation_ms / 1000)) if generation_ms > 0 else 0
            )

            return BenchmarkMetrics(
                test_name=case.name,
                category=case.category,
                success=True,
                estimated_ttft_ms=estimated_ttft_ms,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                prompt_processing_ms=prompt_processing_ms,
                generation_ms=generation_ms,
                prompt_tok_s=prompt_tok_s,
                generation_tok_s=generation_tok_s,
                output_text=output_text,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.warning("Benchmark case failed", case=case.name, error=str(e))
            return BenchmarkMetrics(
                test_name=case.name,
                category=case.category,
                success=False,
                error=str(e),
            )

    def _aggregate_results(
        self,
        config_id: str,
        model_id: str,
        load_config: LoadConfig,
        actual_config: LoadConfig,
        context_length: int,
        metrics: list[BenchmarkMetrics],
        load_time_ms: float,
    ) -> BenchmarkResult:
        """Aggregate metrics across runs."""
        # Group by test name
        by_test: dict[str, list[BenchmarkMetrics]] = {}
        for m in metrics:
            by_test.setdefault(m.test_name, []).append(m)

        aggregated: list[BenchmarkMetrics] = []
        all_passed = True

        for test_name, test_metrics in by_test.items():
            successful = [m for m in test_metrics if m.success]
            if not successful:
                all_passed = False
                aggregated.append(
                    BenchmarkMetrics(
                        test_name=test_name,
                        category=test_metrics[0].category,
                        success=False,
                        error="All runs failed",
                    )
                )
                continue

            # Use median for robustness
            agg = BenchmarkMetrics(
                test_name=test_name,
                category=successful[0].category,
                success=True,
                load_time_ms=load_time_ms,
                estimated_ttft_ms=statistics.median([m.estimated_ttft_ms for m in successful]),
                prompt_tokens=int(statistics.median([m.prompt_tokens for m in successful])),
                completion_tokens=int(statistics.median([m.completion_tokens for m in successful])),
                total_tokens=int(statistics.median([m.total_tokens for m in successful])),
                prompt_processing_ms=statistics.median(
                    [m.prompt_processing_ms for m in successful]
                ),
                generation_ms=statistics.median([m.generation_ms for m in successful]),
                prompt_tok_s=statistics.median([m.prompt_tok_s for m in successful]),
                generation_tok_s=statistics.median([m.generation_tok_s for m in successful]),
                output_text=successful[0].output_text,  # Use first run's output for quality check
            )
            aggregated.append(agg)

        # Calculate stability score (inverse of coefficient of variation)
        gen_speeds = [
            m.generation_tok_s for m in aggregated if m.success and m.generation_tok_s > 0
        ]
        stability_score = 1.0
        if len(gen_speeds) > 1:
            cv = statistics.stdev(gen_speeds) / statistics.mean(gen_speeds)
            stability_score = max(0.0, 1.0 - cv)

        return BenchmarkResult(
            config_id=config_id,
            model_id=model_id,
            load_config=load_config,
            context_length=context_length,
            gpu_ratio=actual_config.gpu_ratio,
            flash_attention=actual_config.flash_attention,
            kv_cache_gpu=actual_config.offload_kv_cache_to_gpu,
            eval_batch_size=actual_config.eval_batch_size,
            num_experts=actual_config.num_experts,
            metrics=aggregated,
            stability_score=stability_score,
            passed=all_passed,
        )

    def get_avg_generation_tok_s(self, result: BenchmarkResult) -> float:
        """Get average generation tokens/sec across successful tests."""
        speeds = [
            m.generation_tok_s for m in result.metrics if m.success and m.generation_tok_s > 0
        ]
        return statistics.median(speeds) if speeds else 0.0

    def get_avg_prompt_tok_s(self, result: BenchmarkResult) -> float:
        """Get average prompt processing tokens/sec."""
        speeds = [m.prompt_tok_s for m in result.metrics if m.success and m.prompt_tok_s > 0]
        return statistics.median(speeds) if speeds else 0.0

    def get_avg_ttft_ms(self, result: BenchmarkResult) -> float:
        """Backward compat."""
        return self.get_avg_estimated_ttft_ms(result)

    def get_avg_estimated_ttft_ms(self, result: BenchmarkResult) -> float:
        """Get average estimated time to first token."""
        ttfts = [
            m.estimated_ttft_ms for m in result.metrics if m.success and m.estimated_ttft_ms > 0
        ]
        return statistics.median(ttfts) if ttfts else 0.0
