"""Dedicated tests for the workload model (INTERACTIVE / THROUGHPUT)."""

import pytest

from lm_optimizer.domain.models import (
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
)
from lm_optimizer.services.workload import (
    ThroughputMeasurement,
    WorkloadType,
    _RequestOutcome,
    aggregate_throughput,
    apply_to_space,
    effective_throughput,
    normalize_workload,
    recorded_aggregate,
    summarize_throughput,
    workload_tiebreak_key,
)


def _result(gen=40.0, ttft=100.0, parallel=1):
    metrics = [
        BenchmarkMetrics(
            test_name="t",
            category="instruction",
            success=True,
            generation_tok_s=gen,
            prompt_tok_s=500,
            estimated_ttft_ms=ttft,
            completion_tokens=50,
            output_text="ok",
        )
    ]
    cfg = LoadConfiguration(context_length=4096, parallel=parallel)
    return ConfigurationResult(
        config=cfg, context_length=4096, status=ConfigurationStatus.PASSED, metrics=metrics
    )


class TestTypes:
    def test_enum_values(self):
        assert WorkloadType.INTERACTIVE.value == "interactive"
        assert WorkloadType.THROUGHPUT.value == "throughput"

    def test_normalize(self):
        assert normalize_workload(None) == "interactive"
        assert normalize_workload("THROUGHPUT") == "throughput"
        assert normalize_workload("batch") == "throughput"
        assert normalize_workload("interactive") == "interactive"


class TestSpace:
    def test_interactive_fixes_parallel(self):
        assert apply_to_space([1, 2, 4], "interactive") == [1]

    def test_interactive_respects_explicit_override(self):
        assert apply_to_space([1, 2, 4], "interactive", explicit_override=[2, 4]) == [1, 2, 4]

    def test_throughput_keeps_full(self):
        assert apply_to_space([1, 2, 4], "throughput") == [1, 2, 4]


class TestMetrics:
    def test_legacy_estimate_documented(self):
        # Legacy estimate kept for compat only; the decision path uses
        # effective_throughput (measured-first). Do not use for decisions.
        assert aggregate_throughput(_result(gen=40.0, parallel=4)) == pytest.approx(160.0)
        assert aggregate_throughput(_result(gen=40.0, parallel=1)) == pytest.approx(40.0)

    def test_tiebreak_interactive_prefers_ttft(self):
        low_ttft = _result(gen=38.0, ttft=50.0, parallel=1)
        high_ttft = _result(gen=30.0, ttft=200.0, parallel=4)
        assert workload_tiebreak_key("interactive", low_ttft) > workload_tiebreak_key(
            "interactive", high_ttft
        )

    def test_tiebreak_throughput_prefers_measured_aggregate(self):
        # P1.5 fix: throughput decisions use MEASURED aggregate, never
        # per_request_tps * parallel. Without a measurement the fallback is
        # the single-request rate (no multiplication).
        slow_single = _result(gen=38.0, ttft=50.0, parallel=1)
        fast_measured = _result(gen=30.0, ttft=200.0, parallel=4)
        fast_measured.generation = {
            "throughput_measured": {"aggregate_tok_s": 100.0, "parallelism": 4}
        }
        assert workload_tiebreak_key("throughput", fast_measured) > workload_tiebreak_key(
            "throughput", slow_single
        )

    def test_tiebreak_throughput_fallback_is_not_multiplied(self):
        a = _result(gen=38.0, ttft=50.0, parallel=1)
        b = _result(gen=30.0, ttft=200.0, parallel=4)
        assert effective_throughput(b) == pytest.approx(30.0)
        assert workload_tiebreak_key("throughput", a) > workload_tiebreak_key("throughput", b)


class TestMeasuredAggregate:
    def test_summarize_formula(self):
        outcomes = [
            _RequestOutcome(tokens=10, elapsed_s=0.5, ok=True),
            _RequestOutcome(tokens=10, elapsed_s=0.5, ok=True),
            _RequestOutcome(tokens=10, elapsed_s=0.5, ok=True),
            _RequestOutcome(tokens=10, elapsed_s=0.5, ok=True),
        ]
        m = summarize_throughput(4, outcomes, wall_s=1.0)
        assert isinstance(m, ThroughputMeasurement)
        assert m.parallelism == 4
        assert m.total_tokens == 40
        assert m.aggregate_tok_s == pytest.approx(40.0)  # total / wall
        assert m.failures == 0
        assert m.per_request_tok_s == pytest.approx([20.0] * 4)

    def test_summarize_counts_failures(self):
        outcomes = [
            _RequestOutcome(tokens=10, elapsed_s=0.5, ok=True),
            _RequestOutcome(tokens=0, elapsed_s=0.1, ok=False, error="boom"),
        ]
        m = summarize_throughput(2, outcomes, wall_s=0.5)
        assert m.total_tokens == 10
        assert m.failures == 1
        assert m.errors == ["boom"]
        assert m.aggregate_tok_s == pytest.approx(20.0)

    def test_summarize_zero_wall_is_safe(self):
        m = summarize_throughput(2, [_RequestOutcome(tokens=5, elapsed_s=0.1)], wall_s=0.0)
        assert m.aggregate_tok_s == 0.0

    def test_recorded_and_effective(self):
        r = _result(gen=30.0, parallel=4)
        assert recorded_aggregate(r) is None
        assert effective_throughput(r) == pytest.approx(30.0)
        r.generation = {"throughput_measured": {"aggregate_tok_s": 95.5, "parallelism": 4}}
        assert recorded_aggregate(r) == pytest.approx(95.5)
        assert effective_throughput(r) == pytest.approx(95.5)

    def test_measurement_serializes(self):
        m = summarize_throughput(2, [_RequestOutcome(tokens=8, elapsed_s=0.4)], wall_s=0.4)
        d = m.to_dict()
        for key in (
            "parallelism",
            "total_tokens",
            "elapsed_s",
            "aggregate_tok_s",
            "per_request_tok_s",
            "failures",
            "queue_wait_ms",
        ):
            assert key in d


class TestConcurrentProbe:
    def _svc(self, chat):
        from unittest.mock import AsyncMock, MagicMock

        from lm_optimizer.services.benchmark import BenchmarkService

        c = MagicMock()
        load = MagicMock()
        load.success = True
        load.loaded_config = None
        c.load_model = AsyncMock(return_value=load)
        c.ensure_unloaded = AsyncMock(return_value=True)
        c.chat_completion = chat
        return BenchmarkService(c)

    def test_concurrent_aggregate_is_total_over_wall(self):
        import asyncio
        from unittest.mock import AsyncMock

        async def _chat(**_kw):
            import asyncio as _aio

            # 50 ms >> 0.5 ms rounding quantum: recomputation from the
            # rounded elapsed_s stays within tolerance, deterministically.
            await _aio.sleep(0.05)
            return {
                "choices": [{"message": {"content": "ok response text"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
                "_stats": {},
            }

        chat = AsyncMock(side_effect=_chat)
        svc = self._svc(chat)
        from lm_optimizer.domain.models import LoadConfiguration

        out = asyncio.run(svc.measure_throughput("m", LoadConfiguration(context_length=2048), 4))
        assert out["parallelism"] == 4
        assert out["total_tokens"] == 40
        assert out["failures"] == 0
        assert chat.await_count == 4
        # aggregate == total / wall (exact formula is unit-tested in
        # TestMeasuredAggregate; here wall rounding adds noise -> tolerance).
        assert out["aggregate_tok_s"] == pytest.approx(
            out["total_tokens"] / out["elapsed_s"], rel=0.05
        )
        assert len(out["per_request_tok_s"]) == 4

    def test_concurrent_failures_recorded(self):
        import asyncio
        from unittest.mock import AsyncMock

        chat = AsyncMock(
            side_effect=[
                {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"completion_tokens": 8},
                    "_stats": {},
                },
                RuntimeError("server busy"),
            ]
        )
        svc = self._svc(chat)
        from lm_optimizer.domain.models import LoadConfiguration

        out = asyncio.run(svc.measure_throughput("m", LoadConfiguration(context_length=2048), 2))
        assert out["failures"] == 1
        assert out["total_tokens"] == 8
        assert len(out["errors"]) == 1

    def test_load_failure_marks_all_failed(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from lm_optimizer.services.benchmark import BenchmarkService

        c = MagicMock()
        load = MagicMock()
        load.success = False
        load.error = "refused"
        c.load_model = AsyncMock(return_value=load)
        c.ensure_unloaded = AsyncMock(return_value=True)
        svc = BenchmarkService(c)
        from lm_optimizer.domain.models import LoadConfiguration

        out = asyncio.run(svc.measure_throughput("m", LoadConfiguration(context_length=2048), 3))
        assert out["failures"] == 3
        assert out["total_tokens"] == 0


class TestPersistence:
    def test_workload_in_run_metadata(self):
        from lm_optimizer.domain.models import ModelIdentity, HardwareInfo, OptimizationRun

        hw = HardwareInfo(
            os="T",
            cpu_name="C",
            cpu_cores_physical=1,
            cpu_cores_logical=2,
            total_ram_gb=8,
            gpu_count=0,
        )
        run = OptimizationRun(model=ModelIdentity(id="m", name="m"), hardware=hw)
        run.benchmark_params["workload_type"] = normalize_workload("throughput")
        assert run.benchmark_params["workload_type"] == "throughput"

    def test_search_space_workload_aware(self):
        from unittest.mock import MagicMock
        from lm_optimizer.domain.models import ModelIdentity, HardwareInfo, OptimizationProfile
        from lm_optimizer.services.lm_studio import LMStudioCapabilities
        from lm_optimizer.services.search_space import SearchSpaceGenerator

        client = MagicMock()
        cap = LMStudioCapabilities()
        cap.supports_context_length = True
        cap.supports_gpu_ratio = True
        cap.supports_flash_attention = True
        cap.supports_kv_cache_placement = True
        cap.supports_eval_batch_size = True
        cap.supports_physical_batch_size = True
        cap.supports_parallel = True
        client.capabilities = cap
        gen = SearchSpaceGenerator(client)
        model = ModelIdentity(id="m", name="m", context_limit=8192)
        hw = HardwareInfo(
            os="T",
            cpu_name="C",
            cpu_cores_physical=1,
            cpu_cores_logical=2,
            total_ram_gb=8,
            gpu_count=0,
            gpus=[],
        )
        # CPU-only would force [0.0] gpu; give it a fake GPU via override path:
        from lm_optimizer.domain.models import GPUInfo

        hw.gpus = [GPUInfo(index=0, name="G", vram_gb=24, vendor="NVIDIA")]
        hw.gpu_count = 1
        s_inter = gen.generate(
            model, hw, OptimizationProfile.BALANCED, {"workload_type": "interactive"}
        )
        s_thr = gen.generate(
            model, hw, OptimizationProfile.BALANCED, {"workload_type": "throughput"}
        )
        assert s_inter.parallels == [1]
        assert s_thr.parallels == [1, 2, 4]
