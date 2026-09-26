"""Phase B freeze + adaptive final validation + checkpoint resume state.

TDD Batch 4: mocked sweeps/benchmark, real optimizer phase code.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from lm_optimizer.domain.models import (
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
    ModelIdentity,
    OptimizationRun,
)
from lm_optimizer.services.optimizer import AdaptiveOptimizer, OptimizationState
from lm_optimizer.services.search_space import SearchSpace


def _model():
    return ModelIdentity(id="m", name="m", context_limit=32768)


def _opt():
    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    opt.client = MagicMock()
    opt.benchmark = MagicMock()
    opt.benchmark.smoke_test = AsyncMock(return_value=(True, 60.0, ""))
    opt.quality = MagicMock()
    opt.search_generator = MagicMock()
    opt.state = OptimizationState(run=OptimizationRun(model=_model()), search_space=SearchSpace())
    opt.state.speed_context = 2048
    opt.state.quality_context = 8192
    opt.state.phase_b_enabled = True
    opt.state.run.phase_b_enabled = True
    opt._progress_cb = None
    opt._checkpoint = lambda reason="x": None  # type: ignore
    opt._broadcast = AsyncMock()  # type: ignore
    return opt


def _no_db(monkeypatch):
    from lm_optimizer.database import repositories as _repo

    monkeypatch.setattr(_repo.config_repo, "save", lambda c: str(c.id))
    monkeypatch.setattr(_repo.run_repo, "save", lambda r: str(r.id))


def _winner():
    cfg = LoadConfiguration(
        context_length=8192,
        flash_attention=True,
        offload_kv_cache_to_gpu=True,
        eval_batch_size=512,
    )
    m = BenchmarkMetrics(
        test_name="short_instruction", category="instruction", success=True,
        completion_tokens=100, generation_tok_s=60.0,
    )
    return ConfigurationResult(
        id=uuid4(), config=cfg, context_length=8192,
        status=ConfigurationStatus.PASSED, metrics=[m],
    )


class TestPhaseB:
    def test_runtime_frozen_only_context_changes(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        seen_cfgs = []

        async def _smoke(model_id, cfg):
            seen_cfgs.append(dict(cfg.to_dict()))
            return True, 60.0, ""

        opt.benchmark.smoke_test = AsyncMock(side_effect=_smoke)

        async def _fake_sweep(probe, min_ctx=2048, max_ctx=65536, **kw):
            for ctx in (min_ctx, max_ctx):
                await probe(ctx)
            return {"maximum_stable_context": min_ctx, "performance_optimal": None,
                    "balanced_recommended": None}

        result = asyncio.run(opt._phase_context_optional(_winner(), sweep_fn=_fake_sweep))
        assert result is not None
        gpu_cfgs = [c for c in seen_cfgs]
        assert gpu_cfgs, "expected Phase B probes"
        # GPU-only pass: every probe identical except context_length.
        first, rest = gpu_cfgs[0], gpu_cfgs[1:]
        runtime_keys = [k for k in first if k != "context_length"]
        for other in rest[:1]:
            assert all(other.get(k) == first.get(k) for k in runtime_keys)

    def test_gpu_only_vs_system_max_reported(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()

        async def _fake_sweep(probe, min_ctx=2048, max_ctx=65536, **kw):
            return {"maximum_stable_context": 16384, "performance_optimal": None,
                    "balanced_recommended": None}

        result = asyncio.run(opt._phase_context_optional(_winner(), sweep_fn=_fake_sweep))
        assert result["gpu_only_max"] == 16384
        assert result["system_max"] == 16384
        assert result["model_limit"] == 32768
        assert result["hardware_stable_limit"] == 16384
        assert result["final_capacity"] == 16384
        assert opt.state.phase_b_result is result

    def test_disabled_phase_b_is_noop(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        opt.state.phase_b_enabled = False
        assert asyncio.run(opt._phase_context_optional(_winner())) is None
        assert opt.benchmark.smoke_test.await_count == 0


class TestFinalValidation:
    def test_stable_config_gets_minimum_reps(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        best = _winner()
        best.stability_score = 0.99
        opt.state.best_config = best
        opt.state.tested_configs = [best]
        calls = []

        async def _tc(cfg, ctx, **kw):
            calls.append(1)
            r = _winner()
            r.score = 0.9
            return r

        opt._test_config = AsyncMock(side_effect=_tc)  # type: ignore
        asyncio.run(opt._phase_final_validation())
        assert len(calls) == 2

    def test_no_best_is_safe(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        opt.state.best_config = None
        asyncio.run(opt._phase_final_validation())  # must not raise


class TestCheckpointPhase:
    def test_payload_carries_phase_state(self):
        from lm_optimizer.storage.run_checkpoint import build_checkpoint_payload

        opt = _opt()
        opt.state.phase = "recovery"
        opt.state.speed_context = 2048
        opt.state.quality_context = 8192
        opt.state.budget_class = "SLOW"
        opt.state.speed_finalists = ["abc"]
        opt.state.recovery_log = [{"event": "reconfirm"}]
        opt.state.excluded_notes = ["mmap note"]
        opt.state.phase_b_enabled = True
        payload = build_checkpoint_payload(opt.state, reason="test")
        pab = payload["phase_ab"]
        assert pab["phase"] == "recovery"
        assert pab["speed_context"] == 2048
        assert pab["quality_context"] == 8192
        assert pab["budget_class"] == "SLOW"
        assert pab["speed_finalist_ids"] == ["abc"]
        assert pab["recovery_log"] == [{"event": "reconfirm"}]
        assert pab["excluded_notes"] == ["mmap note"]
        assert pab["phase_b_enabled"] is True
