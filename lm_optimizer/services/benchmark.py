"""Benchmark service with deterministic test suite."""

import statistics
import time
from dataclasses import dataclass

# Single source of truth for benchmark suite — import from benchmark/suite.py
from lm_optimizer.benchmark.suite import BENCHMARK_SUITE as _BENCHMARK_SUITE
from lm_optimizer.config import config
from lm_optimizer.domain.models import (
    BenchmarkCase,
    BenchmarkMetrics,
    ConfigurationResult,
    LoadConfiguration,
)
from lm_optimizer.logging_config import get_logger
from lm_optimizer.services.lm_studio import LMStudioClient

logger = get_logger(__name__)

# Re-export for backward compat — authoritative definition is in benchmark/suite.py
BENCHMARK_SUITE = _BENCHMARK_SUITE


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""

    repetitions: int = 3
    warmup_repetitions: int = 1
    timeout_seconds: float = 300.0
    load_timeout_seconds: float = 60.0
    generation_timeout_seconds: float = 120.0


class BenchmarkService:
    """Runs benchmarks against LM Studio."""

    def __init__(self, client: LMStudioClient, benchmark_config: BenchmarkConfig | None = None):
        self.client = client
        self.config = benchmark_config or BenchmarkConfig(
            repetitions=config.optimization.benchmark_runs,
            warmup_repetitions=config.optimization.warmup_runs,
        )

    def create_cases_for_context(self, context_length: int) -> list[BenchmarkCase]:
        """Create benchmark cases adapted for the given context length."""
        cases = []
        for bc in BENCHMARK_SUITE:
            max_tokens = min(bc.max_tokens, context_length // 4)
            cases.append(
                BenchmarkCase(
                    name=bc.name,
                    category=bc.category,
                    prompt=bc.prompt,
                    max_tokens=max_tokens,
                    temperature=bc.temperature,
                    stop_sequences=bc.stop_sequences,
                )
            )
        return cases

    async def run_benchmark(
        self,
        model_id: str,
        load_config: LoadConfiguration,
        context_length: int,
    ) -> ConfigurationResult:
        """Run full benchmark suite for a configuration."""
        from uuid import uuid4

        config_id = uuid4()
        result = ConfigurationResult(
            id=config_id,
            config=load_config,
            context_length=context_length,
            status="running",
        )

        logger.info("Starting benchmark", config_id=str(config_id), model=model_id)

        # Load model
        load_start = time.perf_counter()
        load_result = await self.client.load_model(model_id, load_config)
        load_time_ms = (time.perf_counter() - load_start) * 1000

        if not load_result.success:
            result.status = "failed"
            result.error = f"Model load failed: {load_result.error}"
            result.tested_at = datetime.now()
            return result

        # Use actual loaded config
        actual_config = load_result.loaded_config or load_config

        try:
            cases = self.create_cases_for_context(context_length)
            all_metrics: list[BenchmarkMetrics] = []

            # Warmup runs
            for i in range(self.config.warmup_repetitions):
                logger.debug("Warmup run", run=i + 1)
                for case in cases:
                    await self._run_single_case(model_id, case)

            # Measured runs
            for run_idx in range(self.config.repetitions):
                logger.debug("Measured run", run=run_idx + 1)
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

        except Exception as e:
            logger.error("Benchmark failed", error=str(e))
            result.status = "failed"
            result.error = str(e)
        finally:
            # Always unload
            await self.client.unload_model(model_id=model_id)

        logger.info("Benchmark complete", config_id=str(config_id), status=result.status)
        return result

    async def _run_single_case(self, model_id: str, case: BenchmarkCase) -> BenchmarkMetrics:
        """Run a single benchmark case."""
        start_time = time.perf_counter()

        try:
            messages = [{"role": "user", "content": case.prompt}]

            response = await self.client.chat_completion(
                model=model_id,
                messages=messages,
                temperature=case.temperature,
                max_tokens=case.max_tokens,
                seed=42,
                stop=case.stop_sequences,
            )

            total_time_ms = (time.perf_counter() - start_time) * 1000

            # Extract metrics
            usage = response.get("usage", {})
            choices = response.get("choices", [])
            choice = choices[0] if choices else {}
            message = choice.get("message", {})
            output_text = message.get("content", "")

            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # Estimated TTFT: rough approximation without streaming support.
            # LM Studio API does not expose true TTFT via /api/v1/chat/completions
            # without streaming. We estimate as 10% of total time for docs.
            # Labeled as estimated_ttft_ms in reports/UI.
            estimated_ttft_ms = total_time_ms * 0.1 if prompt_tokens > 0 else total_time_ms
            prompt_processing_ms = total_time_ms * (prompt_tokens / max(total_tokens, 1))
            generation_ms = total_time_ms - prompt_processing_ms

            prompt_tok_s = (
                (prompt_tokens / (prompt_processing_ms / 1000)) if prompt_processing_ms > 0 else 0
            )
            generation_tok_s = (
                (completion_tokens / (generation_ms / 1000)) if generation_ms > 0 else 0
            )

            return BenchmarkMetrics(
                test_name=case.name,
                category=case.category,
                success=True,
                load_time_ms=0,
                estimated_ttft_ms=estimated_ttft_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
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
        config_id,
        model_id: str,
        load_config: LoadConfiguration,
        actual_config: LoadConfiguration,
        context_length: int,
        metrics: list[BenchmarkMetrics],
        load_time_ms: float,
    ) -> ConfigurationResult:
        """Aggregate metrics across runs."""

        result = ConfigurationResult(
            id=config_id,
            config=load_config,
            context_length=context_length,
        )

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
                output_text=successful[0].output_text,
            )
            aggregated.append(agg)

        # Calculate stability score
        gen_speeds = [
            m.generation_tok_s for m in aggregated if m.success and m.generation_tok_s > 0
        ]
        stability_score = 1.0
        if len(gen_speeds) > 1:
            cv = statistics.stdev(gen_speeds) / statistics.mean(gen_speeds)
            stability_score = max(0.0, 1.0 - cv)

        result.metrics = aggregated
        result.stability_score = stability_score
        result.status = "passed" if all_passed else "failed"
        if not all_passed:
            result.error = "Some tests failed"
        result.tested_at = datetime.now()

        return result


# Import at end to avoid circular
from datetime import datetime
