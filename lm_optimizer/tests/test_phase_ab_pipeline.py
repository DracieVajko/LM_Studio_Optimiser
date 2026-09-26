"""Phase A/B pipeline on the optimizer (mission §3-§22, tests A-P).

TDD Batch 3: mocked benchmark/client seams, real pipeline + frontier logic.
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
    OptimizationProfile,
    OptimizationRun,
    QualityScore,
)
from lm_optimizer.services.optimizer import AdaptiveOptimizer, OptimizationState
from lm_optimizer.services.search_space import SearchSpace


def _model():
    return ModelIdentity(id="m", name="m", context_limit=131072)


def _run():
    return OptimizationRun(model=_model(), profile=OptimizationProfile.SPEED)


def _probe_result(speed: float, status=ConfigurationStatus.PASSED, **cfg_kw) -> ConfigurationResult:
    cfg = LoadConfiguration(context_length=2048, **cfg_kw)
    m = BenchmarkMetrics(
        test_name="speed_probe",
        category="speed",
        success=status == ConfigurationStatus.PASSED,
        completion_tokens=40,
        generation_tok_s=speed,
        error=None if status == ConfigurationStatus.PASSED else "load boom",
    )
    return ConfigurationResult(
        id=uuid4(), config=cfg, context_length=2048, status=status, metrics=[m]
    )


def _opt():
    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    opt.client = MagicMock()
    from lm_optimizer.domain.models import LMStudioCapabilities

    opt.client.capabilities = LMStudioCapabilities()
    opt.benchmark = MagicMock()
    opt.benchmark.gpu_via_cli = False
    opt.benchmark.run_speed_probe = AsyncMock()
    opt.benchmark.run_cases = AsyncMock()
    opt.benchmark.measure_throughput = AsyncMock(
        return_value={"aggregate_tok_s": 10.0, "parallelism": 2}
    )
    opt.quality = MagicMock()
    opt.search_generator = MagicMock()
    opt.state = OptimizationState(run=_run(), search_space=SearchSpace())
    opt.state.speed_context = 2048
    opt.state.quality_context = 8192
    opt.state.phase = "speed"
    opt._progress_cb = None
    opt._checkpoint = lambda reason="x": None  # type: ignore
    opt._broadcast = AsyncMock()  # type: ignore
    return opt


def _no_db(monkeypatch):
    from lm_optimizer.database import repositories as _repo

    monkeypatch.setattr(_repo.config_repo, "save", lambda c: str(c.id))
    monkeypatch.setattr(_repo.run_repo, "save", lambda r: str(r.id))


class TestSpeedPhase:
    def test_probes_share_frozen_context_and_skip_full_suite(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        opt.benchmark.run_speed_probe = AsyncMock(
            side_effect=lambda model, cfg, ctx, **kw: _probe_result(
                50.0,
                **{k: v for k, v in cfg.to_dict().items() if k != "context_length"},
            )
        )
        front = asyncio.run(opt._phase_speed())
        assert front["winner"] is not None
        for call in opt.benchmark.run_speed_probe.call_args_list:
            assert call.args[2] == 2048
        assert not opt.benchmark.run_cases.await_count
        # Bounded: no Cartesian explosion.
        assert opt.benchmark.run_speed_probe.await_count <= 25

    def test_slow_model_continues(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        opt.benchmark.run_speed_probe = AsyncMock(return_value=_probe_result(0.5))
        opt._test_config = AsyncMock(return_value=None)  # type: ignore
        front = asyncio.run(opt._phase_speed())
        assert front["winner"].get_avg_generation_tok_s() == 0.5
        assert opt.state.budget_class == "VERY_SLOW"
        asyncio.run(opt._phase_quality(front))
        assert opt._test_config.await_count >= 1  # quality still runs

    def test_throughput_kept_separate(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        seen = []

        async def _probe(model, cfg, ctx, **kw):
            r = _probe_result(50.0)
            r.config = cfg
            seen.append(cfg.parallel)
            return r

        opt.benchmark.run_speed_probe = AsyncMock(side_effect=_probe)
        asyncio.run(opt._phase_speed())
        assert opt.benchmark.measure_throughput.await_count >= 1
        stored = [
            c.generation.get("throughput_measured")
            for c in opt.state.tested_configs
            if c.generation and c.generation.get("throughput_measured")
        ]
        assert stored, "throughput block must be recorded separately"

    def test_manual_only_recorded_not_probed(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        opt.benchmark.run_speed_probe = AsyncMock(return_value=_probe_result(50.0))
        asyncio.run(opt._phase_speed())
        assert any("mmap" in n for n in opt.state.excluded_notes)


class TestQualityPhase:
    def _passed(self, speed=60.0, **cfg_kw):
        r = _probe_result(speed, **cfg_kw)
        r.quality_score = QualityScore(
            overall=0.99,
            task_completion=1.0,
            factual_consistency=1.0,
            format_compliance=1.0,
            coding_correctness=1.0,
            no_truncation=1.0,
            no_malformed=1.0,
            checks_passed=30,
            checks_total=30,
        )
        r.score = 0.9
        return r

    def test_finalists_reach_quality_at_frozen_context(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        results = [
            self._passed(70.0),
            self._passed(68.9),
            self._passed(67.8),
            self._passed(61.0, offload_kv_cache_to_gpu=False),
        ]

        def _key(cfg):
            return {k: v for k, v in cfg.to_dict().items() if k != "context_length"}

        async def _tc(cfg, ctx, **kw):
            assert ctx == 8192
            match = next(r for r in results if _key(r.config) == _key(cfg))
            return match

        opt._test_config = AsyncMock(side_effect=_tc)  # type: ignore
        front = {
            "winner": results[0],
            "finalists": results[:3],
            "contrarian": results[3],
        }
        safe, failed = asyncio.run(opt._phase_quality(front))
        assert len(safe) == 4  # band + contrarian all quality-tested
        assert failed == []
        assert opt.state.best_config is not None

    def test_failed_fastest_cannot_win(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        fast_bad = self._passed(99.0, eval_batch_size=1024)
        fast_bad.status = ConfigurationStatus.QUALITY_FAILED
        fast_bad.score = None
        slow_good = self._passed(50.0, eval_batch_size=512)

        def _key(cfg):
            return {k: v for k, v in cfg.to_dict().items() if k != "context_length"}

        async def _tc(cfg, ctx, **kw):
            return fast_bad if _key(cfg) == _key(fast_bad.config) else slow_good

        opt._test_config = AsyncMock(side_effect=_tc)  # type: ignore
        front = {"winner": fast_bad, "finalists": [fast_bad, slow_good], "contrarian": None}
        safe, failed = asyncio.run(opt._phase_quality(front))
        assert failed == [fast_bad]
        assert opt.state.best_config is slow_good
        assert opt.state.raw_fastest is fast_bad


class TestRecovery:
    def test_single_param_rollback_and_subset_recheck(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        last_good = LoadConfiguration(context_length=8192, eval_batch_size=512)
        failed_cfg = LoadConfiguration(
            context_length=8192, eval_batch_size=1024, flash_attention=False
        )
        failed = _probe_result(70.0)
        failed.config = failed_cfg
        failed.context_length = 8192
        failed.status = ConfigurationStatus.QUALITY_FAILED
        failed.generation = {
            "quality_by_test": {
                "coding_task": {"overall": 0.5, "checks_passed": 2, "checks_total": 6},
                "short_instruction": {"overall": 1.0, "checks_passed": 6, "checks_total": 6},
            }
        }
        from lm_optimizer.domain.models import BenchmarkCase

        opt.benchmark.create_cases_for_context = MagicMock(
            return_value=[
                BenchmarkCase(name="coding_task", category="coding", prompt="p", max_tokens=64),
                BenchmarkCase(
                    name="short_instruction", category="instruction", prompt="q", max_tokens=32
                ),
            ]
        )
        subset_calls = []

        async def _cases(model, cfg, ctx, cases, **kw):
            subset_calls.append(([c.name for c in cases], cfg.to_dict()))
            r = _probe_result(69.0)
            r.config = cfg
            return r

        opt.benchmark.run_cases = AsyncMock(side_effect=_cases)
        good_q = QualityScore(
            overall=0.99,
            task_completion=1.0,
            factual_consistency=1.0,
            format_compliance=1.0,
            coding_correctness=1.0,
            no_truncation=1.0,
            no_malformed=1.0,
            checks_passed=30,
            checks_total=30,
        )
        opt.quality.evaluate_all = MagicMock(return_value={"coding_task": good_q})
        opt.quality.aggregate_quality = MagicMock(return_value=good_q)
        # Reconfirm fails (deterministic), first rollback passes.
        opt.quality.passes_threshold = MagicMock(side_effect=[False, True])
        admitted = _probe_result(68.0)
        admitted.score = 0.9
        opt._test_config = AsyncMock(return_value=admitted)  # type: ignore

        recovered, culprit = asyncio.run(opt._run_recovery(failed, last_good))
        # Only failed tests rechecked (not the full suite).
        assert subset_calls, "expected subset recheck"
        assert subset_calls[0][0] == ["coding_task"]
        # First probe rolls back exactly one (highest-risk) param.
        first_rolled = subset_calls[1][1] if len(subset_calls) > 1 else None
        assert first_rolled is not None
        base_dict = failed_cfg.to_dict()
        diff = {
            k
            for k in set(first_rolled) | set(base_dict)
            if first_rolled.get(k) != base_dict.get(k)
        }
        assert len(diff) == 1
        assert recovered is not None
        assert culprit in ("flash_attention", "eval_batch_size")
        assert opt.state.recovery_log, "recovery must be recorded"
