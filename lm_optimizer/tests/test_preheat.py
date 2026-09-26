"""Dedicated tests for the real preheat abstraction (LOAD/VERIFY/PREHEAT/MEASURE)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from lm_optimizer.services.preheat import (
    PreheatConfig,
    PreheatPhase,
    run_preheat_phase,
)


def _client(text="hi there"):
    c = MagicMock()
    c.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": text}}], "usage": {}, "_stats": {}}
    )
    return c


class TestLifecycle:
    def test_phases_enum(self):
        assert PreheatPhase.LOAD.value == "load"
        assert PreheatPhase.VERIFY.value == "verify"
        assert PreheatPhase.PREHEAT.value == "preheat"
        assert PreheatPhase.MEASURE.value == "measure"

    def test_verify_then_preheat(self):
        res = asyncio.run(run_preheat_phase(_client(), "m"))
        assert res.ok is True
        assert PreheatPhase.VERIFY.value in res.phases
        assert PreheatPhase.PREHEAT.value in res.phases

    def test_identical_across_candidates(self):
        cfg = PreheatConfig(prompt="identical prompt", max_tokens=10)
        c1, c2 = _client(), _client()
        asyncio.run(run_preheat_phase(c1, "m", cfg))
        asyncio.run(run_preheat_phase(c2, "m", cfg))
        for call in list(c1.chat_completion.call_args_list) + list(
            c2.chat_completion.call_args_list
        ):
            kwargs = call.kwargs
            assert kwargs["input_text"] == "identical prompt"
            assert kwargs["max_output_tokens"] == 10


class TestMetrics:
    def test_load_time_not_contaminated(self):
        # Preheat returns only warmup wall-clock; caller keeps load_time_ms.
        res = asyncio.run(run_preheat_phase(_client(), "m"))
        assert res.warmup_time_ms >= 0.0
        assert not hasattr(res, "load_time_ms")

    def test_recorded_separately(self):
        res = asyncio.run(run_preheat_phase(_client(), "m"))
        d = res.to_dict()
        assert "warmup_time_ms" in d and "phases" in d
        assert "estimated_ttft_ms" not in d  # never faked TTFT

    def test_configurable(self):
        cfg = PreheatConfig(prompt="custom", max_tokens=5, temperature=0.1, repetitions=2)
        c = _client()
        res = asyncio.run(run_preheat_phase(c, "m", cfg))
        assert res.ok is True
        # 1 verify + 2 preheat calls.
        assert c.chat_completion.await_count == 3

    def test_disabled_is_noop(self):
        c = _client()
        res = asyncio.run(run_preheat_phase(c, "m", PreheatConfig(enabled=False)))
        assert res.ok is True and res.warmup_time_ms == 0.0
        assert c.chat_completion.await_count == 0


class TestControl:
    def test_cancellable(self):
        c = _client()
        res = asyncio.run(
            run_preheat_phase(c, "m", PreheatConfig(repetitions=5), is_cancelled=lambda: True)
        )
        assert res.ok is False and res.error == "cancelled during preheat"

    def test_error_reported(self):
        c = MagicMock()
        c.chat_completion = AsyncMock(side_effect=RuntimeError("boom"))
        res = asyncio.run(run_preheat_phase(c, "m"))
        assert res.ok is False and "boom" in (res.error or "")


class TestNoContamination:
    def test_warmup_excluded_from_metrics(self):
        # Warmup outputs must never leak into measured metrics; load_time
        # covers only the load call; TTFT comes only from measured stats.
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from lm_optimizer.services.benchmark import BenchmarkService

        async def _chat(**kwargs):
            text = (
                "PREHEAT-TEXT" if kwargs.get("input_text") == "Say hi in 5 words." else "MEASURED"
            )
            n = 3 if text == "PREHEAT-TEXT" else 10
            return {
                "choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": n, "total_tokens": 5 + n},
                "_stats": {"tokens_per_second": 20.0, "time_to_first_token_seconds": 0.05},
            }

        c = MagicMock()
        load = MagicMock()
        load.success = True
        load.loaded_config = None
        c.load_model = AsyncMock(return_value=load)
        c.ensure_unloaded = AsyncMock(return_value=True)
        c.chat_completion = AsyncMock(side_effect=_chat)
        svc = BenchmarkService(c)
        from lm_optimizer.domain.models import LoadConfiguration

        result = asyncio.run(svc.run_benchmark("m", LoadConfiguration(context_length=2048), 2048))
        texts = [m.output_text for m in result.metrics]
        assert texts, "expected measured metrics"
        assert all("PREHEAT-TEXT" not in t for t in texts)
        # Metrics count == repetitions x suite size (warmup excluded).
        from lm_optimizer.benchmark.suite import BENCHMARK_SUITE

        assert len(result.metrics) == len(BENCHMARK_SUITE)
        assert result.generation["preheat"]["warmup_time_ms"] >= 0.0


class TestIntegration:
    def test_benchmark_records_preheat(self):
        from lm_optimizer.services.benchmark import BenchmarkService

        c = MagicMock()
        load = MagicMock()
        load.success = True
        load.loaded_config = None
        c.load_model = AsyncMock(return_value=load)
        c.ensure_unloaded = AsyncMock(return_value=True)
        c.chat_completion = AsyncMock(
            return_value={
                "choices": [{"message": {"content": "hello world answer"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
                "_stats": {"tokens_per_second": 20.0, "time_to_first_token_seconds": 0.05},
            }
        )
        svc = BenchmarkService(c)
        from lm_optimizer.domain.models import LoadConfiguration

        result = asyncio.run(svc.run_benchmark("m", LoadConfiguration(context_length=2048), 2048))
        assert result.generation is not None
        assert "preheat" in result.generation
        assert "warmup_time_ms" in result.generation["preheat"]
