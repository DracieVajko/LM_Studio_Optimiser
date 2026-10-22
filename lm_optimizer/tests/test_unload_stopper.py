"""Regression tests: an unverified unload stops before the next heavy load.

Audit case: the optimizer must refuse to load another memory-intensive
candidate when the previous model's unload could not be verified (leftover
still holding memory or unverifiable endpoint state). ``cancel()``
checkpoints the run — results stay available for resume/inspection.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from lm_optimizer.api.client import LMStudioClient
from lm_optimizer.domain.models import (
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    HardwareInfo,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
)
from lm_optimizer.services.optimizer import AdaptiveOptimizer, OptimizationState
from lm_optimizer.services.search_space import SearchSpace
from lm_optimizer.services.speed_search import SpeedPlanItem


@pytest.fixture
def opt():
    o = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    o.client = MagicMock()
    o.client.ensure_unloaded = AsyncMock(return_value=True)
    o.client.unload_model = AsyncMock()

    async def _instances():
        return []

    o.client.get_loaded_instances = _instances
    o.client._models_cache = []
    o.benchmark = MagicMock()
    o.benchmark.gpu_via_cli = False
    o.benchmark.style = "balanced"
    o.benchmark.run_speed_probe = AsyncMock()
    o.benchmark.run_benchmark = AsyncMock()
    o.quality = MagicMock()
    o.state = OptimizationState(
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
    o.state.run.benchmark_params = {"auto_unload_foreign": True}
    o.state.quality_context = 32768
    o.state.speed_context = 4096
    o._checkpoint = lambda reason="x": None
    o._broadcast = AsyncMock()
    return o


class TestOptimizerStopper:
    @pytest.mark.asyncio
    async def test_probe_blocked_and_cancels(self, opt):
        opt.benchmark.last_unverified_unload = "model-under-test"
        opt._probe_completed = lambda k: False

        item = SpeedPlanItem(
            config=LoadConfiguration(context_length=4096),
            stage="s0", kind="speed", reason="t",
        )
        out = await opt._probe_speed_item(item, 4096, set())

        assert out is None
        assert opt.state.should_cancel is True
        assert opt.benchmark.run_speed_probe.await_count == 0
        assert opt.benchmark.last_unverified_unload is None
        events = [e["message"] for e in opt.state.event_log]
        assert "UNLOAD_UNVERIFIED" in events
        assert any("unload_not_verified" in e for e in opt.state.errors)

    @pytest.mark.asyncio
    async def test_full_suite_attempt_blocked(self, opt):
        opt.benchmark.last_unverified_unload = "model-under-test"

        res = await opt._test_config(LoadConfiguration(context_length=4096), 4096)

        assert res is None
        assert opt.state.should_cancel is True
        assert opt.benchmark.run_benchmark.await_count == 0

    @pytest.mark.asyncio
    async def test_no_block_when_flag_verified(self, opt):
        opt.benchmark.last_unverified_unload = None
        opt._probe_completed = lambda k: False
        opt._record_probe_result = AsyncMock(side_effect=lambda r, c: r)

        async def _probe(*a, **k):
            return ConfigurationResult(
                config=LoadConfiguration(context_length=4096),
                context_length=4096,
                status=ConfigurationStatus.PASSED,
                metrics=[BenchmarkMetrics(
                    test_name="s", category="speed", success=True,
                    generation_tok_s=10.0,
                )],
            )

        opt.benchmark.run_speed_probe = AsyncMock(side_effect=_probe)
        item = SpeedPlanItem(
            config=LoadConfiguration(context_length=4096),
            stage="s0", kind="speed", reason="t",
        )
        out = await opt._probe_speed_item(item, 4096, set())

        assert out is not None
        assert opt.state.should_cancel is False

    def test_magicmock_benchmark_flag_ignored(self, opt):
        # Plain MagicMock auto-attributes are truthy but not str: the stopper
        # must ignore them so legacy mocks never trigger a spurious cancel.
        assert opt._unverified_unload_block() is False
        assert opt.state.should_cancel is False


class TestEnsureUnloaded:
    def _mk(self, instances_fn):
        c = LMStudioClient.__new__(LMStudioClient)
        c._loaded_models = {}
        c._no_reasoning_models = set()
        c.get_loaded_instances = instances_fn
        c.unload_model = AsyncMock()
        return c

    @pytest.mark.asyncio
    async def test_no_unload_rpc_when_positively_not_loaded(self):
        async def _empty():
            return []

        c = self._mk(_empty)
        ok = await c.ensure_unloaded("m")
        assert ok is True
        c.unload_model.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unloads_leftover_by_instance_id(self):
        seq = iter([
            [{"model": "m", "instance_id": "i1", "config": {}}],
            [],
        ])

        async def _seq():
            try:
                return next(seq)
            except StopIteration:
                return []

        c = self._mk(_seq)
        ok = await c.ensure_unloaded("m")
        assert ok is True
        c.unload_model.assert_awaited_once_with(identifier="i1")

    @pytest.mark.asyncio
    async def test_leftover_persists_returns_false(self):
        left = [{"model": "m", "instance_id": "i1", "config": {}}]

        async def _always():
            return left

        c = self._mk(_always)
        ok = await c.ensure_unloaded("m")
        assert ok is False
        assert c.unload_model.await_count >= 2
