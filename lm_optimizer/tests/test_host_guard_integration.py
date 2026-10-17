"""Host-guard integration: foreign loads are cleared before a measurement."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from lm_optimizer.domain.models import (
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    HardwareInfo,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
    QualityScore,
)
from lm_optimizer.services.foreign_watchdog import BACKEND_LMSTUDIO
from lm_optimizer.services.optimizer import AdaptiveOptimizer, OptimizationState
from lm_optimizer.services.search_space import SearchSpace


def _opt(instances, *, allow_unload=True):
    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    opt.client = MagicMock()
    opt.client.ensure_unloaded = AsyncMock(return_value=True)
    opt.client.unload_model = AsyncMock(return_value=True)

    async def _instances():
        return instances

    opt.client.get_loaded_instances = _instances
    opt.client._models_cache = [
        ModelIdentity(id="prism-ml/bonsai-27b", name="b", size_bytes=4_730_000_000),
        ModelIdentity(id="model-under-test", name="m", size_bytes=9_120_000_000),
    ]

    opt.benchmark = MagicMock()
    opt.benchmark.gpu_via_cli = False
    opt.benchmark.run_speed_probe = AsyncMock()
    opt.benchmark.run_cases = AsyncMock()
    opt.benchmark.style = "balanced"
    opt.quality = MagicMock()
    opt.state = OptimizationState(
        run=OptimizationRun(
            model=ModelIdentity(id="model-under-test", name="m", context_limit=32768),
            hardware=HardwareInfo(
                os="T", cpu_name="C", cpu_cores_physical=8, cpu_cores_logical=16,
                total_ram_gb=32.0, gpu_count=0,
            ),
            profile=OptimizationProfile.BALANCED,
        ),
        search_space=SearchSpace(),
    )
    opt.state.run.benchmark_params = {"auto_unload_foreign": allow_unload}
    opt.state.quality_context = 32768
    opt.state.speed_context = 4096
    opt._checkpoint = lambda reason="x": None
    opt._broadcast = AsyncMock()
    return opt


def _inst(model, iid):
    return {"model": model, "instance_id": iid, "config": {}}


class TestGuardHost:
    @pytest.mark.asyncio
    async def test_unloads_foreign_before_measuring(self):
        opt = _opt([_inst("model-under-test", "i1"), _inst("qwen/qwen3-coder-30b", "i2")])

        out = await opt.guard_host()

        assert out["inspected"] is True
        assert out["foreign"] == ["qwen/qwen3-coder-30b"]
        assert out["unloaded"] == ["qwen/qwen3-coder-30b"]
        opt.client.unload_model.assert_awaited_once_with(
            model_id="qwen/qwen3-coder-30b", identifier="i2"
        )
        # Evidence recorded on the run so reports can explain a failure.
        assert any("foreign_model_loaded" in e for e in opt.state.errors)
        events = [e["message"] for e in opt.state.event_log]
        assert "FOREIGN_MODEL_DETECTED" in events

    @pytest.mark.asyncio
    async def test_never_unloads_the_model_under_test(self):
        opt = _opt([_inst("model-under-test", "i1"), _inst("other", "i2")])

        await opt.guard_host()

        for call in opt.client.unload_model.await_args_list:
            assert call.kwargs["model_id"] != "model-under-test"

    @pytest.mark.asyncio
    async def test_echo_probe_model_protected(self):
        """The smallest model is loaded by connect(); never treat it as foreign."""
        opt = _opt([_inst("model-under-test", "i1"), _inst("prism-ml/bonsai-27b", "i2")])

        out = await opt.guard_host()

        assert out["foreign"] == []
        assert out["unloaded"] == []

    @pytest.mark.asyncio
    async def test_unload_can_be_disabled(self):
        opt = _opt(
            [_inst("model-under-test", "i1"), _inst("rogue", "i2")],
            allow_unload=False,
        )

        out = await opt.guard_host()

        assert out["foreign"] == ["rogue"]
        assert out["unloaded"] == []
        opt.client.unload_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_unreachable_host_does_not_abort_the_run(self):
        opt = _opt([])

        async def _boom():
            raise ConnectionError("server down")

        opt.client.get_loaded_instances = _boom

        out = await opt.guard_host()  # must not raise

        assert out["inspected"] is False
        assert out["foreign"] == []

    @pytest.mark.asyncio
    async def test_repeated_calls_reuse_one_watchdog(self):
        opt = _opt([_inst("model-under-test", "i1")])

        await opt.guard_host()
        first = opt._watchdog
        await opt.guard_host()

        assert opt._watchdog is first


class TestGuardRunsBeforeMeasurement:
    @pytest.mark.asyncio
    async def test_speed_probe_path_guards_first(self):
        opt = _opt([_inst("model-under-test", "i1"), _inst("rogue", "i2")])
        order = []

        async def _instances():
            order.append("inspect")
            return [_inst("model-under-test", "i1"), _inst("rogue", "i2")]

        opt.client.get_loaded_instances = _instances

        from lm_optimizer.services.speed_search import SpeedPlanItem

        cfg = LoadConfiguration(context_length=4096)
        item = SpeedPlanItem(config=cfg, stage="s0", kind="speed", reason="test")
        opt._probe_completed = lambda k: False

        async def _probe(*a, **k):
            order.append("probe")
            return ConfigurationResult(
                config=cfg, context_length=4096,
                status=ConfigurationStatus.PASSED,
                metrics=[BenchmarkMetrics(
                    test_name="s", category="speed", success=True, generation_tok_s=10.0,
                )],
            )

        opt.benchmark.run_speed_probe = AsyncMock(side_effect=_probe)
        opt._record_probe_result = AsyncMock(side_effect=lambda r, c: r)

        await opt._probe_speed_item(item, 4096, set())

        assert order.index("inspect") < order.index("probe"), (
            "host must be guarded BEFORE the probe runs"
        )

    @pytest.mark.asyncio
    async def test_test_config_path_guards_first(self):
        opt = _opt([_inst("model-under-test", "i1"), _inst("rogue", "i2")])
        order = []

        async def _instances():
            order.append("inspect")
            return [_inst("model-under-test", "i1"), _inst("rogue", "i2")]

        opt.client.get_loaded_instances = _instances

        async def _bench(*a, **k):
            order.append("bench")
            res = ConfigurationResult(
                config=LoadConfiguration(context_length=4096),
                context_length=4096,
                status=ConfigurationStatus.PASSED,
                metrics=[BenchmarkMetrics(
                    test_name="t", category="instruction", success=True, generation_tok_s=12.0,
                )],
            )
            return res

        opt.benchmark.run_benchmark = AsyncMock(side_effect=_bench)
        opt.quality.evaluate_all = MagicMock(return_value={})
        opt.quality.aggregate_quality = MagicMock(
            return_value=QualityScore(
                overall=0.99, task_completion=1.0, factual_consistency=1.0,
                format_compliance=1.0, coding_correctness=1.0, no_truncation=1.0,
                no_malformed=1.0, checks_passed=30, checks_total=30,
            )
        )
        opt.quality.passes_threshold = MagicMock(return_value=True)
        opt.state.tested_configs = []

        res = await opt._test_config(LoadConfiguration(context_length=4096), 4096)

        assert order.index("inspect") < order.index("bench"), (
            "host must be guarded BEFORE the benchmark runs"
        )
        assert res is not None