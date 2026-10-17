"""Task 7: duration_ms persistence on every started attempt."""
import asyncio
import itertools
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

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


def _hw():
    return HardwareInfo(
        os="T", cpu_name="C", cpu_cores_physical=1, cpu_cores_logical=2,
        total_ram_gb=8, gpu_count=0,
    )


def _qs(overall=0.99):
    return QualityScore(
        overall=overall, task_completion=1.0, factual_consistency=1.0,
        format_compliance=1.0, coding_correctness=1.0, no_truncation=1.0,
        no_malformed=1.0, checks_passed=6, checks_total=6,
    )


def _metric(gen=40.0):
    return BenchmarkMetrics(
        test_name="t", category="instruction", success=True,
        generation_tok_s=gen, prompt_tok_s=500, estimated_ttft_ms=100,
        completion_tokens=50, output_text="measured answer",
    )


def _optimizer(bench_behavior):
    from lm_optimizer.services.optimizer import AdaptiveOptimizer, OptimizationState
    from lm_optimizer.services.search_space import SearchSpace

    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    opt.client = MagicMock()
    opt.client.ensure_unloaded = AsyncMock(return_value=True)
    opt.benchmark = MagicMock()
    if isinstance(bench_behavior, BaseException):
        opt.benchmark.run_benchmark = AsyncMock(side_effect=bench_behavior)
    else:
        opt.benchmark.run_benchmark = AsyncMock(return_value=bench_behavior)
    opt.quality = MagicMock()
    opt.quality.evaluate_all = MagicMock(return_value={"t": _qs()})
    opt.quality.aggregate_quality = MagicMock(return_value=_qs())
    opt.quality.passes_threshold = MagicMock(return_value=True)
    opt.quality.config = MagicMock()
    opt.quality.config.minimum_score = 0.97
    run = OptimizationRun(
        model=ModelIdentity(id="m", name="m"), hardware=_hw(),
        profile=OptimizationProfile.BALANCED,
    )
    run.profile_weights = {"generation_speed": 1.0}
    run.benchmark_params = {"selection_threshold": 0.05}
    run.advanced_settings = {}
    opt.state = OptimizationState(
        run=run,
        search_space=SearchSpace(
            context_lengths=[4096], gpu_ratios=[1.0],
            flash_attention_options=[True], kv_cache_options=[True],
            batch_sizes=[256],
        ),
    )
    opt.state.tested_configs = []
    return opt


def _clock(monkeypatch, start=100.0, end=100.5):
    import lm_optimizer.services.optimizer as _opt

    vals = itertools.chain([start, end], itertools.repeat(end))
    monkeypatch.setattr(_opt.time, "perf_counter", lambda: next(vals))


def _bench_passed():
    return ConfigurationResult(
        config=LoadConfiguration(context_length=4096),
        context_length=4096, status="passed", metrics=[_metric()],
        tested_at=datetime.now(),
    )


def _bench_failed():
    return ConfigurationResult(
        config=LoadConfiguration(context_length=4096),
        context_length=4096, status="failed", metrics=[],
        tested_at=datetime.now(), error="Model load failed: refused",
    )


class TestDurationStamping:
    def test_success_stamps_duration(self, monkeypatch):
        from lm_optimizer.database import repositories as _repo

        monkeypatch.setattr(_repo.config_repo, "save", lambda c: str(c.id))
        _clock(monkeypatch)
        opt = _optimizer(_bench_passed())
        result = asyncio.run(opt._test_config(LoadConfiguration(context_length=4096), 4096))
        assert result is not None
        assert result.duration_ms == 500.0

    def test_load_failure_stamps_duration(self, monkeypatch):
        from lm_optimizer.database import repositories as _repo

        monkeypatch.setattr(_repo.config_repo, "save", lambda c: str(c.id))
        _clock(monkeypatch)
        opt = _optimizer(_bench_failed())
        result = asyncio.run(opt._test_config(LoadConfiguration(context_length=4096), 4096))
        assert result is not None
        assert result.status == ConfigurationStatus.LOAD_FAILED
        assert result.score is None
        assert result.duration_ms == 500.0

    def test_exception_path_stamps_duration(self, monkeypatch):
        from lm_optimizer.database import repositories as _repo

        monkeypatch.setattr(_repo.config_repo, "save", lambda c: str(c.id))
        _clock(monkeypatch)
        opt = _optimizer(RuntimeError("boom"))
        result = asyncio.run(opt._test_config(LoadConfiguration(context_length=4096), 4096))
        assert result is not None  # stamped failed record, never a winner
        assert result.score is None
        assert result.duration_ms == 500.0
