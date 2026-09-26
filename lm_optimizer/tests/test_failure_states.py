"""Failure state machine: failed loads never score, never win (§1-§4, §25)."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

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
from lm_optimizer.services.failure_states import (
    classify_error,
    is_hard,
    is_soft,
    is_terminal_failure,
)


def _hw():
    return HardwareInfo(
        os="T", cpu_name="C", cpu_cores_physical=1, cpu_cores_logical=2, total_ram_gb=8, gpu_count=0
    )


def _optimizer(bench_result=None):
    from lm_optimizer.services.optimizer import AdaptiveOptimizer, OptimizationState
    from lm_optimizer.services.search_space import SearchSpace

    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    opt.client = MagicMock()
    opt.client.ensure_unloaded = AsyncMock(return_value=True)
    opt.benchmark = MagicMock()
    if bench_result is not None:
        opt.benchmark.run_benchmark = AsyncMock(return_value=bench_result)
    opt.quality = MagicMock()
    run = OptimizationRun(
        model=ModelIdentity(id="m", name="m"), hardware=_hw(), profile=OptimizationProfile.BALANCED
    )
    run.profile_weights = {"generation_speed": 1.0}
    run.benchmark_params = {"selection_threshold": 0.05}
    opt.state = OptimizationState(
        run=run,
        search_space=SearchSpace(
            context_lengths=[4096],
            gpu_ratios=[1.0],
            flash_attention_options=[True],
            kv_cache_options=[True],
            batch_sizes=[256],
        ),
    )
    opt.state.tested_configs = []
    opt.state.started_at = datetime.now()
    # Persist fixtures: config saves carry a FK to runs(models).
    from lm_optimizer.database import repositories as _repo

    hw_id = _repo.hardware_repo.save(opt.state.run.hardware)
    opt.state.run.hardware_id = hw_id
    _repo.model_repo.save(opt.state.run.model)
    _repo.run_repo.save(opt.state.run)
    return opt


def _bench(status="passed", error=None, gen=40.0):
    metrics = (
        [
            BenchmarkMetrics(
                test_name="t",
                category="instruction",
                success=True,
                generation_tok_s=gen,
                prompt_tok_s=500,
                estimated_ttft_ms=100,
                completion_tokens=50,
                output_text="measured answer",
            )
        ]
        if status == "passed"
        else []
    )
    return ConfigurationResult(
        config=LoadConfiguration(context_length=4096),
        context_length=4096,
        status=status,
        metrics=metrics,
        tested_at=datetime.now(),
        error=error,
    )


def _quality(overall=0.98, passed=6, total=6):
    return QualityScore(
        overall=overall,
        task_completion=1.0,
        factual_consistency=1.0,
        format_compliance=1.0,
        coding_correctness=1.0,
        no_truncation=1.0,
        no_malformed=1.0,
        checks_passed=passed,
        checks_total=total,
    )


class TestClassification:
    def test_oom_signals(self):
        assert classify_error("CUDA OOM") == ConfigurationStatus.OOM
        assert classify_error("out of memory") == ConfigurationStatus.OOM

    def test_timeout(self):
        assert classify_error("TimeoutException: timed out") == ConfigurationStatus.TIMEOUT

    def test_hard_incompatible(self):
        assert classify_error("400 unrecognized_keys") == ConfigurationStatus.INCOMPATIBLE
        assert classify_error("unsupported parameter") == ConfigurationStatus.INCOMPATIBLE
        assert classify_error("model_not_found") == ConfigurationStatus.INCOMPATIBLE

    def test_load_failed(self):
        assert classify_error("Model load failed: refused") == ConfigurationStatus.LOAD_FAILED
        assert classify_error("connection refused") == ConfigurationStatus.LOAD_FAILED

    def test_empty_and_generic(self):
        assert classify_error(None) == ConfigurationStatus.BENCHMARK_FAILED
        assert classify_error("weird boom") == ConfigurationStatus.BENCHMARK_FAILED

    def test_hard_vs_soft(self):
        assert is_hard(ConfigurationStatus.INCOMPATIBLE) is True
        assert is_hard(ConfigurationStatus.OOM) is False
        assert is_soft(ConfigurationStatus.OOM) is True
        assert is_soft(ConfigurationStatus.TIMEOUT) is True
        assert is_soft(ConfigurationStatus.LOAD_FAILED) is True
        assert is_soft(ConfigurationStatus.INCOMPATIBLE) is False
        assert is_terminal_failure(ConfigurationStatus.PASSED) is False
        assert is_terminal_failure(ConfigurationStatus.QUALITY_FAILED) is True


class TestFailedLoadNeverScores:
    def test_load_failed_state(self):
        opt = _optimizer(_bench(status="failed", error="Model load failed: refused"))
        opt.quality.evaluate_all = MagicMock(return_value={})
        result = asyncio.run(opt._test_config(LoadConfiguration(context_length=4096), 4096))
        assert result.status == ConfigurationStatus.LOAD_FAILED
        assert result.score is None
        assert result.quality_score is None
        assert result.score_breakdown is None
        opt.quality.evaluate_all.assert_not_called()

    def test_no_config_passed_event_for_failures(self):
        opt = _optimizer(_bench(status="failed", error="Model load failed: x"))
        opt.quality.evaluate_all = MagicMock(return_value={})
        asyncio.run(opt._test_config(LoadConfiguration(context_length=4096), 4096))
        kinds = [e.get("message") for e in opt.state.event_log]
        assert "CONFIG_ELIGIBLE" not in kinds
        assert "Candidate accepted" not in kinds
        assert any("LOAD_FAILED" in (k or "") for k in kinds)

    def test_oom_class(self):
        opt = _optimizer(_bench(status="failed", error="CUDA OOM"))
        opt.quality.evaluate_all = MagicMock(return_value={})
        result = asyncio.run(opt._test_config(LoadConfiguration(context_length=4096), 4096))
        assert result.status == ConfigurationStatus.OOM
        assert result.score is None

    def test_failed_result_saved_but_not_appended(self, monkeypatch):
        from lm_optimizer.database import repositories as _repo

        saved = []
        monkeypatch.setattr(_repo.config_repo, "save", lambda c: saved.append(c) or str(c.id))
        opt = _optimizer(_bench(status="failed", error="Model load failed: x"))
        opt.quality.evaluate_all = MagicMock(return_value={})
        result = asyncio.run(opt._test_config(LoadConfiguration(context_length=4096), 4096))
        assert result.score is None
        assert saved and saved[0].score is None
        assert opt.state.tested_configs == []
        assert opt.state.best_config is None


class TestQualityFailed:
    def test_quality_failed_preserves_metrics_no_score(self):
        opt = _optimizer(_bench(status="passed", gen=40.0))
        opt.quality.evaluate_all = MagicMock(return_value={"t": _quality(0.5, 3, 6)})
        opt.quality.aggregate_quality = MagicMock(return_value=_quality(0.5, 3, 6))
        opt.quality.passes_threshold = MagicMock(return_value=False)
        opt.quality.config = MagicMock()
        opt.quality.config.minimum_score = 0.97
        result = asyncio.run(opt._test_config(LoadConfiguration(context_length=4096), 4096))
        assert result.status == ConfigurationStatus.QUALITY_FAILED
        assert result.metrics  # preserved
        assert result.quality_score is not None  # evidence preserved
        assert result.score is None
        assert opt.state.best_config is None

    def test_quality_failed_never_winner(self):
        from lm_optimizer.services.run_summary import best_passed

        good = _bench(status="passed", gen=30.0)
        good.status = ConfigurationStatus.PASSED
        good.score = 0.4
        bad = _bench(status="passed", gen=99.0)
        bad.status = ConfigurationStatus.QUALITY_FAILED
        bad.score = None
        assert best_passed([good, bad]).get_avg_generation_tok_s() == 30.0


class TestWinnerEligibility:
    def test_update_best_rejects_unscored(self):
        opt = _optimizer()
        good = _bench(status="passed", gen=40.0)
        good.status = ConfigurationStatus.PASSED
        good.score = 0.8
        opt.state.best_config = good
        bad = _bench(status="failed", error="OOM")
        bad.status = ConfigurationStatus.OOM
        bad.score = None
        opt._update_best(bad)
        assert opt.state.best_config is good

    def test_pareto_never_empty_for_valid_set(self):
        opt = _optimizer()
        a = _bench(status="passed", gen=30.0)
        a.status = ConfigurationStatus.PASSED
        a.score, a.quality_score = 0.5, _quality(0.98)
        a.peak_vram_gb = 4.0
        b = _bench(status="passed", gen=50.0)
        b.status = ConfigurationStatus.PASSED
        b.score, b.quality_score = 0.7, _quality(0.99)
        b.peak_vram_gb = 8.0
        front = opt._compute_pareto_frontier([a, b])
        assert len(front) >= 1

    def test_pareto_excludes_failed_and_unscored(self):
        opt = _optimizer()
        ok = _bench(status="passed", gen=40.0)
        ok.status = ConfigurationStatus.PASSED
        ok.score, ok.quality_score = 0.6, _quality(0.98)
        ok.peak_vram_gb = 4.0
        bad = _bench(status="failed", error="x")
        bad.status = ConfigurationStatus.LOAD_FAILED
        bad.score = None
        front = opt._compute_pareto_frontier([ok, bad])
        assert bad not in front and ok in front


class TestQualityDeterminism:
    def test_same_text_same_score(self):
        from lm_optimizer.services.quality import QualityEvaluator

        ev = QualityEvaluator()
        text = '{"name": "Ada", "age": 36, "skills": ["a", "b", "c"], "address": {"city": "X", "country": "Y"}}'
        s1 = ev.evaluate("m", "structured_output", text)
        s2 = ev.evaluate("m", "structured_output", text)
        assert s1.overall == s2.overall
        assert (s1.checks_passed, s1.checks_total) == (s2.checks_passed, s2.checks_total)

    def test_suite_temperatures_frozen(self):
        from lm_optimizer.services.benchmark import BenchmarkService

        c = MagicMock()
        first = BenchmarkService(c).create_cases_for_context(4096)
        second = BenchmarkService(c).create_cases_for_context(4096)
        assert [(x.name, x.temperature) for x in first] == [(x.name, x.temperature) for x in second]
