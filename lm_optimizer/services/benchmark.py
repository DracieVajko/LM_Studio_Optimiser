"""Benchmark service with deterministic test suite."""

import asyncio
import statistics
import time
from dataclasses import dataclass

# Single source of truth for benchmark suite - import from benchmark/suite.py
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

# Re-export for backward compat - authoritative definition is in benchmark/suite.py
BENCHMARK_SUITE = _BENCHMARK_SUITE

# Benchmark style presets: category -> temperature override.
# precise: code/math need determinism; creative: summarization benefits from
# variance; balanced: suite defaults. Recorded in ConfigurationResult.generation.
STYLE_TEMPERATURE_OVERRIDES: dict[str, dict[str, float]] = {
    "precise": {"coding": 0.1, "reasoning": 0.1},
    "creative": {"context": 0.9},
    "balanced": {},
}

VALID_STYLES = ("precise", "balanced", "creative")


def _sample_mem() -> dict:
    """One VRAM/RAM snapshot; {} when unavailable (never fabricated)."""
    try:
        from lm_optimizer.services.hostguard import gpu_free_mb, mem_free_gb

        out: dict = {}
        try:
            gpus = gpu_free_mb() or []
            primary = next((g for g in gpus if isinstance(g, dict) and "free_mb" in g), None)
            if primary:
                out["vram_free_mb"] = float(primary["free_mb"])
                out["vram_total_mb"] = float(primary["total_mb"])
        except Exception:
            pass
        try:
            mem = mem_free_gb() or {}
            if "ram_free_gb" in mem:
                out["ram_free_gb"] = float(mem["ram_free_gb"])
                out["ram_total_gb"] = float(mem["total_ram_gb"])
        except Exception:
            pass
        return out
    except Exception:
        return {}


def _peak_usage(before: dict | None, after: dict | None) -> tuple[float | None, float | None]:
    """Peak GB used = total - min(free) over snapshots; None if unmeasurable."""
    peak_vram = peak_ram = None
    try:
        frees = [
            float(s.get("vram_free_mb"))
            for s in (before, after)
            if isinstance(s, dict) and s.get("vram_free_mb") is not None
        ]
        totals = [
            float(s.get("vram_total_mb"))
            for s in (before, after)
            if isinstance(s, dict) and s.get("vram_total_mb")
        ]
        if frees and totals:
            peak_vram = round((totals[0] - min(frees)) / 1024, 2)
    except (TypeError, ValueError):
        pass
    try:
        frees = [
            float(s.get("ram_free_gb"))
            for s in (before, after)
            if isinstance(s, dict) and s.get("ram_free_gb") is not None
        ]
        totals = [
            float(s.get("ram_total_gb"))
            for s in (before, after)
            if isinstance(s, dict) and s.get("ram_total_gb")
        ]
        if frees and totals:
            peak_ram = round(totals[0] - min(frees), 2)
    except (TypeError, ValueError):
        pass
    return peak_vram, peak_ram


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""

    repetitions: int = 3
    warmup_repetitions: int = 1
    timeout_seconds: float = 300.0
    load_timeout_seconds: float = 60.0
    generation_timeout_seconds: float = 120.0
    # Cost control for slow models: scales every case's max_tokens
    # (0.25-1.0). Below 1.0 risks truncation-failed quality gates.
    max_tokens_scale: float = 1.0


class BenchmarkService:
    """Runs benchmarks against LM Studio."""

    def __init__(
        self,
        client: LMStudioClient,
        benchmark_config: BenchmarkConfig | None = None,
        style: str = "balanced",
        reasoning: str | None = "off",
        gpu_via_cli: bool = False,
    ):
        self.client = client
        self.config = benchmark_config or BenchmarkConfig(
            repetitions=config.optimization.benchmark_runs,
            warmup_repetitions=config.optimization.warmup_runs,
        )
        if style not in VALID_STYLES:
            raise ValueError(f"Unknown style {style!r}, expected one of {VALID_STYLES}")
        self.style = style
        # GPU offload the REST API cannot express goes through `lms load
        # --gpu` when enabled (and available); otherwise REST with the ratio
        # recorded-but-not-applied (never silently claimed as applied).
        self.gpu_via_cli = gpu_via_cli
        # Benchmarks score the ANSWER, not the thinking trace (R32). The native
        # chat endpoint exposes reasoning off/low/medium/high/on; default off
        # keeps outputs deterministic and scorable.
        self.reasoning = reasoning

    async def _channel_load(
        self, model_id: str, cfg: LoadConfiguration
    ) -> tuple[bool, str | None, object | None, str, dict | None, str]:
        """Load via the authoritative channel.

        Returns (ok, identifier_or_error, loaded_config, channel, applied,
        load_error). CLI is used only for gpu_ratio REST cannot express and
        only when enabled + available; CLI failure falls back to REST.
        """
        from lm_optimizer.services import control_adapter as _ca

        if (
            self.gpu_via_cli
            and getattr(cfg, "gpu_ratio", None) is not None
            and _ca.lms_cli.lms_available()
        ):
            result, channel, applied = await _ca.load_with_channel(
                self.client, model_id, cfg, gpu_via_cli=True
            )
            if getattr(result, "success", False):
                return True, getattr(result, "identifier", None), None, channel, applied, ""
            err = getattr(result, "error", "") or "CLI load failed"
            return False, None, None, channel, applied, f"Model load failed: {err}"
        res = await self.client.load_model(model_id, cfg)
        if not res.success:
            return False, None, None, "REST", None, f"Model load failed: {res.error}"
        applied = None
        if getattr(res, "loaded_config", None) is not None:
            try:
                applied = res.loaded_config.to_dict()
            except Exception:
                applied = None
        return (
            True,
            getattr(res, "identifier", None),
            getattr(res, "loaded_config", None),
            "REST",
            applied,
            "",
        )

    async def smoke_test(
        self,
        model_id: str,
        load_config: LoadConfiguration | None = None,
        prompt: str = "Say hi in 5 words.",
        max_tokens: int = 30,
    ) -> tuple[bool, float, str]:
        """Quick load + single-chat gate. Returns (ok, tok_s, error). Always unloads."""
        cfg = load_config or LoadConfiguration(
            context_length=2048, flash_attention=True, offload_kv_cache_to_gpu=True
        )
        try:
            ok, _ident, _loaded, _channel, _applied, err = await self._channel_load(model_id, cfg)
            if not ok:
                return False, 0.0, err
            out = await self.client.chat_completion(
                model=model_id,
                input_text=prompt,
                temperature=0.3,
                max_output_tokens=max_tokens,
                reasoning=self.reasoning,
            )
            stats = out.get("_stats", {}) or {}
            return True, stats.get("tokens_per_second", 0) or 0.0, ""
        except Exception as e:
            return False, 0.0, f"{type(e).__name__}: {e}"
        finally:
            try:
                if not await self.client.ensure_unloaded(model_id):
                    logger.warning("Smoke left model loaded", model=model_id)
            except Exception:
                pass

    async def measure_throughput(
        self,
        model_id: str,
        load_config: LoadConfiguration,
        parallelism: int,
        prompt: str = "Summarize: concurrency probe.",
        max_tokens: int = 32,
        temperature: float = 0.3,
        reasoning: str | None = None,
    ) -> dict[str, object]:
        """Execute N concurrent requests and MEASURE aggregate throughput.

        Returns the recorded block (parallelism, total tokens, wall-clock
        elapsed, aggregate tok/s = total/wall, per-request tok/s, failures,
        queue/wait). Always unloads. One shared load serves all N requests.
        """
        import time as _time

        from lm_optimizer.services.workload import _RequestOutcome, summarize_throughput

        active_reasoning = self.reasoning if reasoning is None else reasoning
        parallel = max(1, parallelism)
        try:
            ok, _ident, _loaded, channel, _applied, load_err = await self._channel_load(
                model_id, load_config
            )
            if not ok:
                failed = summarize_throughput(
                    parallel,
                    [
                        _RequestOutcome(
                            tokens=0, elapsed_s=0.0, ok=False, error=f"load failed: {load_err}"
                        )
                        for _ in range(parallel)
                    ],
                    wall_s=0.0,
                )
                out = failed.to_dict()
                out["load_channel"] = channel
                return out

            async def _one() -> _RequestOutcome:
                start = _time.perf_counter()
                try:
                    response = await self.client.chat_completion(
                        model=model_id,
                        input_text=prompt,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        reasoning=active_reasoning,
                    )
                    usage, _text, _stats = self._extract_response(response)
                    elapsed = _time.perf_counter() - start
                    tokens = int(usage.get("completion_tokens", 0))
                    if not _text.strip() or tokens <= 0:
                        return _RequestOutcome(
                            tokens=0, elapsed_s=elapsed, ok=False, error="Empty output"
                        )
                    return _RequestOutcome(tokens=tokens, elapsed_s=elapsed, ok=True)
                except Exception as e:
                    elapsed = _time.perf_counter() - start
                    return _RequestOutcome(
                        tokens=0,
                        elapsed_s=elapsed,
                        ok=False,
                        error=f"{type(e).__name__}: {e}"[:160],
                    )

            wall_start = _time.perf_counter()
            outcomes = list(await asyncio.gather(*(_one() for _ in range(parallel))))
            wall_s = _time.perf_counter() - wall_start
            out = summarize_throughput(parallel, outcomes, wall_s).to_dict()
            out["load_channel"] = channel
            return out
        finally:
            try:
                if not await self.client.ensure_unloaded(model_id):
                    logger.warning("Throughput probe left model loaded", model=model_id)
            except Exception:
                pass

    def create_cases_for_context(
        self, context_length: int, style: str | None = None
    ) -> list[BenchmarkCase]:
        """Create benchmark cases adapted for the given context length and style."""
        from lm_optimizer.benchmark.suite import create_benchmark_cases

        active_style = style or self.style
        overrides = STYLE_TEMPERATURE_OVERRIDES.get(active_style, {})
        try:
            scale = float(getattr(self.config, "max_tokens_scale", 1.0) or 1.0)
        except (TypeError, ValueError):
            scale = 1.0
        scale = min(1.0, max(0.25, scale))
        return [
            BenchmarkCase(
                name=c.name,
                category=c.category,
                prompt=c.prompt,
                max_tokens=max(32, int(c.max_tokens * scale)),
                temperature=overrides.get(c.category, c.temperature),
                stop_sequences=c.stop_sequences,
            )
            for c in create_benchmark_cases(context_length)
        ]

    async def run_benchmark(
        self,
        model_id: str,
        load_config: LoadConfiguration,
        context_length: int,
        style: str | None = None,
        reasoning: str | None = None,
    ) -> ConfigurationResult:
        """Run full benchmark suite for a configuration."""
        cases = self.create_cases_for_context(context_length, style=style or self.style)
        return await self.run_cases(
            model_id, load_config, context_length, cases, style=style, reasoning=reasoning
        )

    async def run_cases(
        self,
        model_id: str,
        load_config: LoadConfiguration,
        context_length: int,
        cases: list[BenchmarkCase] | None = None,
        repetitions: int | None = None,
        warmup_repetitions: int | None = None,
        style: str | None = None,
        reasoning: str | None = None,
    ) -> ConfigurationResult:
        """Run an explicit case list (full suite or a recovery subset).

        Same load → verify → preheat → measured → unload skeleton as the
        former run_benchmark body; only the case list varies. `warmup=0`
        disables preheat chats (measured runs still execute).
        """
        from uuid import uuid4

        active_style = style or self.style
        active_reasoning = self.reasoning if reasoning is None else reasoning
        reps = self.config.repetitions if repetitions is None else repetitions
        warmups = (
            self.config.warmup_repetitions if warmup_repetitions is None else warmup_repetitions
        )
        config_id = uuid4()
        result = ConfigurationResult(
            id=config_id,
            config=load_config,
            context_length=context_length,
            status="running",
        )

        logger.info("Starting benchmark", config_id=str(config_id), model=model_id)

        load_time_ms = 0.0
        try:
            # Load model inside try: any failure from here on still unloads.
            from lm_optimizer.services import control_adapter as _ca

            load_start = time.perf_counter()
            ok, _ident, loaded_cfg, channel, applied, load_err = await self._channel_load(
                model_id, load_config
            )
            load_time_ms = (time.perf_counter() - load_start) * 1000

            if not ok:
                result.status = "failed"
                result.error = load_err
                result.tested_at = datetime.now()
                return result

            # Use actual loaded config
            actual_config = loaded_cfg or load_config

            active_cases = (
                cases
                if cases is not None
                else self.create_cases_for_context(context_length, style=active_style)
            )
            generation = {
                "style": active_style,
                "temperature_overrides": STYLE_TEMPERATURE_OVERRIDES.get(active_style, {}),
                "reasoning": active_reasoning,
                "load_channel": channel,
                # REQUESTED vs APPLIED (§13): never report requested as verified.
                "load_verification": _ca.verify_applied(load_config.to_api_params(), applied),
                "max_tokens_scale": getattr(self.config, "max_tokens_scale", 1.0),
                "mem_before": _sample_mem(),
            }
            all_metrics: list[BenchmarkMetrics] = []

            # PREHEAT phase (LOAD done above; VERIFY+PREHEAT identical across
            # candidates, timed separately, never contaminates load_time_ms).
            from lm_optimizer.services.preheat import PreheatConfig, run_preheat_phase

            if warmups and warmups > 0:
                preheat = await run_preheat_phase(
                    self.client,
                    model_id,
                    PreheatConfig(repetitions=warmups),
                    reasoning=active_reasoning,
                )
            else:
                preheat = await run_preheat_phase(
                    self.client,
                    model_id,
                    PreheatConfig(enabled=False),
                    reasoning=active_reasoning,
                )
            generation["preheat"] = preheat.to_dict()
            if not preheat.ok and preheat.error == "cancelled during preheat":
                result.status = "failed"
                result.error = "cancelled during preheat"
                result.tested_at = datetime.now()
                return result
            logger.debug("Preheat complete", **preheat.to_dict())

            # Measured runs
            for run_idx in range(reps):
                logger.debug("Measured run", run=run_idx + 1)
                for case in active_cases:
                    metrics = await self._run_single_case(model_id, case, active_reasoning)
                    all_metrics.append(metrics)

            # Peak memory from before/after snapshots (real measurement;
            # None when the host cannot report it — never estimated).
            mem_after = _sample_mem()
            # Aggregate metrics
            result = self._aggregate_results(
                config_id=config_id,
                model_id=model_id,
                load_config=load_config,
                actual_config=actual_config,
                context_length=context_length,
                metrics=all_metrics,
                load_time_ms=load_time_ms,
                generation=generation,
            )
            peak_vram, peak_ram = _peak_usage(generation.get("mem_before"), mem_after)
            result.peak_vram_gb = peak_vram
            result.peak_ram_gb = peak_ram

        except Exception as e:
            logger.error("Benchmark failed", error=str(e))
            result.status = "failed"
            result.error = str(e)
        finally:
            # Always unload and verify server state
            if not await self.client.ensure_unloaded(model_id):
                logger.warning("Model may still be loaded", model=model_id)

        logger.info("Benchmark complete", config_id=str(config_id), status=result.status)
        return result

    async def run_speed_probe(
        self,
        model_id: str,
        load_config: LoadConfiguration,
        context_length: int,
        repetitions: int = 1,
        warmup: bool = True,
        reasoning: str | None = None,
    ) -> ConfigurationResult:
        """Phase-A cheap speed probe: one short prompt, no quality evaluation.

        Never runs the full 5-test suite. Result carries the generation
        phase marker (OptimizationPhase.SPEED) and stays scoreless (score
        None until quality passes).
        """
        from lm_optimizer.domain.models import OptimizationPhase
        from lm_optimizer.services.speed_probe import (
            SPEED_PROBE_MAX_TOKENS,
            speed_probe_case,
        )

        result = await self.run_cases(
            model_id,
            load_config,
            context_length,
            [speed_probe_case()],
            repetitions=repetitions,
            warmup_repetitions=None if warmup else 0,
            reasoning=reasoning,
        )
        gen = dict(result.generation or {})
        gen["phase"] = OptimizationPhase.SPEED.value
        gen["probe"] = True
        gen["probe_max_tokens"] = SPEED_PROBE_MAX_TOKENS
        result.generation = gen
        return result

    def _model_known_no_reasoning(self, model_id: str) -> bool:
        """Whether the model already rejected the reasoning key (duck-typed)."""
        check = getattr(self.client, "model_known_no_reasoning", None)
        return bool(check(model_id)) if callable(check) else False

    async def precision_probe(
        self,
        model_id: str,
        load_config: LoadConfiguration | None = None,
        low_temp: float = 0.2,
    ) -> dict:
        """Compare suite-default temps vs a low temp for max-precision users.

        Runs coding + structured + reasoning tests at both temps (one load).
        Returns {tests: {name: {default_temp, default_text, low_temp, low_text}}}
        plus optional error. Caller evaluates quality. Always unloads.
        """
        from lm_optimizer.benchmark.suite import BENCHMARK_SUITE

        names = ("coding_task", "structured_output", "medium_reasoning")
        prompts = {p.name: p for p in BENCHMARK_SUITE if p.name in names}
        cfg = load_config or LoadConfiguration(
            context_length=2048, flash_attention=True, offload_kv_cache_to_gpu=True
        )
        out: dict = {"tests": {}}
        try:
            res = await self.client.load_model(model_id, cfg)
            if not res.success:
                out["error"] = f"Model load failed: {res.error}"
                return out
            for name in names:
                prompt = prompts[name]
                entry: dict = {"default_temp": prompt.temperature, "low_temp": low_temp}
                for tag, temp in (("default_text", prompt.temperature), ("low_text", low_temp)):
                    response = await self.client.chat_completion(
                        model=model_id,
                        input_text=prompt.prompt,
                        temperature=temp,
                        max_output_tokens=prompt.max_tokens,
                        reasoning=self.reasoning,
                    )
                    usage, text, _stats = self._extract_response(response)
                    entry[tag] = text
                out["tests"][name] = entry
            return out
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
            return out
        finally:
            try:
                await self.client.ensure_unloaded(model_id)
            except Exception:
                pass

    @staticmethod
    def _extract_response(response: dict) -> tuple[dict, str, dict]:
        """Split a converted chat dict into (usage, output_text, stats)."""
        usage = response.get("usage", {})
        choices = response.get("choices", [])
        choice = choices[0] if choices else {}
        output_text = choice.get("message", {}).get("content", "")
        stats = response.get("_stats", {}) or {}
        return usage, output_text, stats

    async def _run_single_case(
        self, model_id: str, case: BenchmarkCase, reasoning: str | None = None
    ) -> BenchmarkMetrics:
        """Run a single benchmark case via the native chat endpoint."""
        start_time = time.perf_counter()

        try:
            response = await self.client.chat_completion(
                model=model_id,
                input_text=case.prompt,
                temperature=case.temperature,
                max_output_tokens=case.max_tokens,
                reasoning=reasoning,
            )
            usage, output_text, stats = self._extract_response(response)
            if (
                not output_text.strip()
                and (reasoning in (None, "off"))
                and not self._model_known_no_reasoning(model_id)
            ):
                # Model may need thinking to answer: exactly one retry with
                # reasoning on (widest-supported thinking mode; sets vary, R36)
                logger.debug("Empty output, retrying with reasoning", case=case.name)
                response = await self.client.chat_completion(
                    model=model_id,
                    input_text=case.prompt,
                    temperature=case.temperature,
                    max_output_tokens=case.max_tokens,
                    reasoning="on",
                )
                usage, output_text, stats = self._extract_response(response)
            if not output_text.strip():
                # Degenerate generation (e.g. immediate EOS): must not poison
                # speed stats with absurd tok/s from ~0 tokens.
                return BenchmarkMetrics(
                    test_name=case.name,
                    category=case.category,
                    success=False,
                    error="Empty output",
                )

            total_time_ms = (time.perf_counter() - start_time) * 1000

            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # Real TTFT/tok/s from native stats when available (fallback: old heuristic)
            ttft_s = stats.get("time_to_first_token_seconds") or 0
            server_tok_s = stats.get("tokens_per_second") or 0
            if ttft_s > 0:
                estimated_ttft_ms = ttft_s * 1000
            else:
                estimated_ttft_ms = total_time_ms * 0.1 if prompt_tokens > 0 else total_time_ms
            if server_tok_s > 0 and completion_tokens > 0:
                generation_ms = completion_tokens / server_tok_s * 1000
                prompt_processing_ms = max(0.0, total_time_ms - generation_ms)
                generation_tok_s = server_tok_s
            else:
                prompt_processing_ms = total_time_ms * (prompt_tokens / max(total_tokens, 1))
                generation_ms = total_time_ms - prompt_processing_ms
                generation_tok_s = (
                    (completion_tokens / (generation_ms / 1000)) if generation_ms > 0 else 0
                )

            prompt_tok_s = (
                (prompt_tokens / (prompt_processing_ms / 1000)) if prompt_processing_ms > 0 else 0
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
        generation: dict | None = None,
    ) -> ConfigurationResult:
        """Aggregate metrics across runs."""

        result = ConfigurationResult(
            id=config_id,
            config=load_config,
            context_length=context_length,
            generation=generation,
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
            # Quality is evaluated on the most complete sample (max completion
            # tokens): single-sample truncation flakes must not fail good configs (R33)
            best_text_run = max(successful, key=lambda m: m.completion_tokens)
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
                output_text=best_text_run.output_text,
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
