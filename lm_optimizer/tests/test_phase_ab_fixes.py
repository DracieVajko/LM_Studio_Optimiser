"""Fix pass for fresh-review findings 1-7 (RED first, one pass)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from lm_optimizer.domain.models import (
    BenchmarkCase,
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
    ModelIdentity,
    OptimizationRun,
    QualityScore,
)
from lm_optimizer.services.optimizer import AdaptiveOptimizer, OptimizationState
from lm_optimizer.services.search_space import SearchSpace


def _model():
    return ModelIdentity(id="m", name="m", context_limit=131072)


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
    opt.benchmark.create_cases_for_context = MagicMock(
        return_value=[
            BenchmarkCase(name="coding_task", category="coding", prompt="p", max_tokens=64),
        ]
    )
    opt.quality = MagicMock()
    opt.search_generator = MagicMock()
    opt.state = OptimizationState(run=OptimizationRun(model=_model()), search_space=SearchSpace())
    opt.state.speed_context = 2048
    opt.state.quality_context = 8192
    opt._progress_cb = None
    opt._checkpoint = lambda reason="x": None  # type: ignore
    opt._broadcast = AsyncMock()  # type: ignore
    return opt


def _no_db(monkeypatch):
    from lm_optimizer.database import repositories as _repo

    monkeypatch.setattr(_repo.config_repo, "save", lambda c: str(c.id))
    monkeypatch.setattr(_repo.run_repo, "save", lambda r: str(r.id))


def _probe_result(speed: float, **cfg_kw) -> ConfigurationResult:
    cfg = LoadConfiguration(context_length=2048, **cfg_kw)
    m = BenchmarkMetrics(
        test_name="speed_probe", category="speed", success=True,
        completion_tokens=40, generation_tok_s=speed,
    )
    return ConfigurationResult(
        id=uuid4(), config=cfg, context_length=2048,
        status=ConfigurationStatus.PASSED, metrics=[m],
    )


def _good_q():
    return QualityScore(
        overall=0.99, task_completion=1.0, factual_consistency=1.0,
        format_compliance=1.0, coding_correctness=1.0, no_truncation=1.0,
        no_malformed=1.0, checks_passed=30, checks_total=30,
    )


class TestReconfirmAdmission:
    def test_reconfirm_pass_admits_noisy_config(self, monkeypatch):
        """Finding 1: noisy (reconfirm-passed) config must be admitted, not dropped."""
        _no_db(monkeypatch)
        opt = _opt()
        failed = _probe_result(70.0)
        failed.status = ConfigurationStatus.QUALITY_FAILED
        failed.context_length = 8192
        failed.generation = {
            "quality_by_test": {
                "coding_task": {"overall": 0.5, "checks_passed": 2, "checks_total": 6},
            }
        }
        opt.benchmark.run_cases = AsyncMock(return_value=_probe_result(69.0))
        opt.quality.evaluate_all = MagicMock(return_value={"coding_task": _good_q()})
        opt.quality.aggregate_quality = MagicMock(return_value=_good_q())
        opt.quality.passes_threshold = MagicMock(return_value=True)
        admitted = _probe_result(69.5)
        admitted.score = 0.9
        opt._test_config = AsyncMock(return_value=admitted)  # type: ignore
        last_good = LoadConfiguration(context_length=8192, eval_batch_size=512)

        recovered, culprit = asyncio.run(opt._run_recovery(failed, last_good))
        assert recovered is admitted
        assert culprit is None
        assert opt._test_config.await_count == 1  # full admission, not silent drop


class TestResumeBudget:
    def test_s0_skip_keeps_measured_budget(self, monkeypatch):
        """Finding 2: resumed S0 skip must reuse the measured budget, not VERY_SLOW."""
        _no_db(monkeypatch)
        opt = _opt()
        s0_cfg = opt._default_anchor()
        s0_cfg.context_length = 2048
        seeded = _probe_result(72.0)
        seeded.config = s0_cfg
        seeded.generation = {"phase": "speed"}
        opt.state.tested_configs = [seeded]
        opt.benchmark.run_speed_probe = AsyncMock(
            side_effect=lambda m, cfg, ctx, **kw: _probe_result(10.0)
        )
        asyncio.run(opt._phase_speed())
        assert opt.state.budget_class == "FAST"
        assert opt.state.probe_repetitions == 3

    def test_baseline_and_reps_persist(self):
        """Findings 2+3+5: checkpoint carries reps + baseline config; restore rebuilds."""
        from lm_optimizer.storage.run_checkpoint import build_checkpoint_payload

        opt = _opt()
        opt.state.probe_repetitions = 2
        opt.state.baseline_result = ConfigurationResult(
            config=LoadConfiguration(context_length=2048, eval_batch_size=512),
            context_length=2048,
            status=ConfigurationStatus.SKIPPED,
        )
        payload = build_checkpoint_payload(opt.state, reason="test")
        assert payload["phase_ab"]["probe_repetitions"] == 2
        assert payload["phase_ab"]["baseline_config"]["eval_batch_size"] == 512
        opt2 = _opt()
        opt2._restore_phase_state(payload)
        assert opt2.state.probe_repetitions == 2
        assert opt2.state.baseline_result is not None
        assert opt2.state.baseline_result.config.eval_batch_size == 512


class TestRefPriority:
    def test_safe_beats_baseline(self):
        """Finding 3: fastest safe is the reference; baseline only as fallback."""
        opt = _opt()
        safe_cfg = LoadConfiguration(context_length=8192, eval_batch_size=256)
        safe = _probe_result(60.0)
        safe.config = safe_cfg
        cfg, kind = opt._recovery_reference([safe])
        assert cfg is safe_cfg and kind == "safe"
        base_cfg = LoadConfiguration(context_length=8192, eval_batch_size=512)
        opt.state.baseline_result = ConfigurationResult(
            config=base_cfg, context_length=2048, status=ConfigurationStatus.SKIPPED
        )
        cfg, kind = opt._recovery_reference([])
        assert cfg is base_cfg and kind == "baseline"
        opt.state.baseline_result = None
        cfg, kind = opt._recovery_reference([])
        assert cfg is None and kind == "none"


class TestRevalidateReset:
    def test_revalidate_clears_phase_state(self):
        """Finding 4: revalidate must clear finalists/raw/recovery/budget."""
        opt = _opt()
        opt.state.speed_finalists = ["x"]
        opt.state.raw_fastest = _probe_result(50.0)
        opt.state.recovery_log = [{"event": "reconfirm"}]
        opt.state.budget_class = "FAST"
        opt.state.probe_repetitions = 3
        opt.state.phase_b_result = {"gpu_only_max": 1}
        opt.state.baseline_result = _probe_result(10.0)
        opt._reset_phase_for_revalidate()
        assert opt.state.speed_finalists == []
        assert opt.state.raw_fastest is None
        assert opt.state.recovery_log == []
        assert opt.state.budget_class is None
        assert opt.state.probe_repetitions == 1
        assert opt.state.phase_b_result is None
        assert opt.state.baseline_result is None


class TestCrashGuards:
    def test_probe_crash_does_not_kill_phase(self, monkeypatch):
        """Finding 6a: one raising probe is logged, the phase completes."""
        _no_db(monkeypatch)
        opt = _opt()
        calls = []

        async def _flaky(m, cfg, ctx, **kw):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("probe boom")
            return _probe_result(50.0)

        opt.benchmark.run_speed_probe = AsyncMock(side_effect=_flaky)
        front = asyncio.run(opt._phase_speed())
        assert front["winner"] is not None
        assert any("Probe error" in e.get("message", "") for e in opt.state.event_log)

    def test_evaluator_crash_means_recheck_fails(self, monkeypatch):
        """Finding 6b: evaluator crash fails the recheck, never kills recovery."""
        _no_db(monkeypatch)
        opt = _opt()
        opt.benchmark.run_cases = AsyncMock(return_value=_probe_result(50.0))
        opt.quality.evaluate_all = MagicMock(side_effect=RuntimeError("eval boom"))
        ok, agg, res = asyncio.run(
            opt._recheck_quality(LoadConfiguration(context_length=8192), 8192, ["coding_task"])
        )
        assert ok is False

    def test_phase_b_crash_is_contained(self, monkeypatch):
        """Finding 6c: optional Phase B can never fail the run."""
        _no_db(monkeypatch)
        opt = _opt()
        opt.state.phase_b_enabled = True

        async def _boom(probe, **kw):
            raise RuntimeError("sweep boom")

        winner = _probe_result(60.0)
        result = asyncio.run(opt._phase_context_optional(winner, sweep_fn=_boom))
        assert result is not None and "error" in result


class TestThroughputKey:
    def test_standard_key_feeds_tiebreak(self, monkeypatch):
        """Finding 7: S3 measurements use the standard throughput_measured key."""
        _no_db(monkeypatch)
        opt = _opt()

        async def _probe(m, cfg, ctx, **kw):
            r = _probe_result(50.0)
            r.config = cfg
            return r

        opt.benchmark.run_speed_probe = AsyncMock(side_effect=_probe)
        asyncio.run(opt._phase_speed())
        from lm_optimizer.services.workload import effective_throughput

        measured = [
            c for c in opt.state.tested_configs if (c.generation or {}).get("throughput_measured")
        ]
        assert measured, "expected standard-key throughput blocks"
        assert effective_throughput(measured[0]) == 10.0
