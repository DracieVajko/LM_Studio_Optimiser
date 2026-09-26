"""Speed probe service path: cheap probe, subset runner (mission §5, §16).

TDD Batch 2: mock client, real BenchmarkService code.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from lm_optimizer.benchmark.suite import BENCHMARK_SUITE
from lm_optimizer.domain.models import BenchmarkCase, LoadConfiguration
from lm_optimizer.services.benchmark import BenchmarkConfig, BenchmarkService
from lm_optimizer.services.speed_probe import SPEED_PROBE_MAX_TOKENS, SPEED_PROBE_PROMPT


def _client(tok_s: float = 50.0):
    async def _chat(**kwargs):
        return {
            "choices": [{"message": {"content": "hash tables map keys fast."}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            "_stats": {"tokens_per_second": tok_s, "time_to_first_token_seconds": 0.05},
        }

    c = MagicMock()
    load = MagicMock()
    load.success = True
    load.loaded_config = None
    c.load_model = AsyncMock(return_value=load)
    c.ensure_unloaded = AsyncMock(return_value=True)
    c.chat_completion = AsyncMock(side_effect=_chat)
    return c


def _suite_prompts() -> set[str]:
    return {c.prompt for c in BENCHMARK_SUITE}


class TestSpeedProbe:
    def test_probe_never_runs_full_suite(self):
        c = _client()
        svc = BenchmarkService(c)
        asyncio.run(
            svc.run_speed_probe("m", LoadConfiguration(context_length=2048), 2048)
        )
        texts = [call.kwargs.get("input_text", "") for call in c.chat_completion.call_args_list]
        assert SPEED_PROBE_PROMPT in texts
        assert not (_suite_prompts() & set(texts)), "full suite must not run during speed probe"

    def test_probe_result_shape(self):
        c = _client(tok_s=50.0)
        svc = BenchmarkService(c)
        result = asyncio.run(
            svc.run_speed_probe("m", LoadConfiguration(context_length=2048), 2048)
        )
        assert result.status == "passed"
        assert result.metrics, "expected probe metrics"
        assert all(m.test_name == "speed_probe" for m in result.metrics)
        assert all(m.success for m in result.metrics)
        assert result.get_avg_generation_tok_s() == 50.0
        assert result.generation["phase"] == "speed"
        assert result.generation["probe_max_tokens"] == SPEED_PROBE_MAX_TOKENS
        c.ensure_unloaded.assert_called()

    def test_probe_repetitions_respected(self):
        c = _client()
        svc = BenchmarkService(
            c, benchmark_config=BenchmarkConfig(repetitions=3, warmup_repetitions=1)
        )
        result = asyncio.run(
            svc.run_speed_probe(
                "m", LoadConfiguration(context_length=2048), 2048, repetitions=2
            )
        )
        assert len(result.metrics) == 1  # reps aggregate to one median metric
        # 2 preheat chats (1 verify + 1 warmup) + 2 measured probe chats.
        assert c.chat_completion.await_count == 4

    def test_probe_load_failure(self):
        c = _client()
        bad = MagicMock()
        bad.success = False
        bad.error = "boom"
        c.load_model = AsyncMock(return_value=bad)
        svc = BenchmarkService(c)
        result = asyncio.run(
            svc.run_speed_probe("m", LoadConfiguration(context_length=2048), 2048)
        )
        assert result.status == "failed"
        assert "boom" in (result.error or "")


class TestRunCases:
    def test_subset_runner_counts(self):
        c = _client()
        svc = BenchmarkService(
            c, benchmark_config=BenchmarkConfig(repetitions=2, warmup_repetitions=0)
        )
        cases = [
            BenchmarkCase(name="a", category="x", prompt="pa", max_tokens=16),
            BenchmarkCase(name="b", category="x", prompt="pb", max_tokens=16),
        ]
        result = asyncio.run(
            svc.run_cases("m", LoadConfiguration(context_length=2048), 2048, cases)
        )
        assert result.status == "passed"
        assert len(result.metrics) == 2  # one median metric per case
        assert {m.test_name for m in result.metrics} == {"a", "b"}
        # warmup_repetitions=0 -> no preheat chats, only measured (2x2).
        assert c.chat_completion.await_count == 4
