"""A recovery rollback must be MEASURED at the rolled-back settings.

Production incident (run 0c3f1671, ministral-3-14b-reasoning, 7.6 h):

    rollback context_length 32768 -> 4096   quality_after 0.749
    rollback gpu_ratio      0.8   -> null   quality_after 0.667

Both took 39-52 minutes and both scored *worse* than the config they replaced,
which is impossible if the rollback had actually been applied. Tracing the code
showed `_try_rollback_suspect` builds a `rolled` config with context_length=4096
but then passes the ORIGINAL `ctx` (32768) to `_recheck_quality`, which uses
that argument - not `cfg.context_length` - to build cases and measure.

Effect: the single most valuable rollback was structurally incapable of working,
and 3.5 h of a 7.6 h run was spent re-measuring at the context it had just
decided to abandon.

These tests pin the contract: the recheck runs at the rolled context.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from lm_optimizer.domain.models import (
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
    QualityScore,
)
from lm_optimizer.services.optimizer import AdaptiveOptimizer

FAILED_CTX = 32768
GOOD_CTX = 4096
FAILED_TESTS = ["coding_task", "structured_output"]


class _Case:
    def __init__(self, name):
        self.name = name


def _result(ctx: int) -> ConfigurationResult:
    return ConfigurationResult(
        config=LoadConfiguration(context_length=ctx, gpu_ratio=0.8),
        context_length=ctx,
        status=ConfigurationStatus.QUALITY_FAILED,
        metrics=[],
        generation={"quality_by_test": {t: {"score": 0.0} for t in FAILED_TESTS}},
    )


def _optimizer():
    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    opt.benchmark = MagicMock()
    opt.benchmark.create_cases_for_context = MagicMock(
        side_effect=lambda ctx: [_Case(n) for n in FAILED_TESTS]
    )
    opt.benchmark.run_cases = AsyncMock(
        return_value=SimpleMetrics()
    )
    opt.quality = MagicMock()
    opt.quality.evaluate_all = MagicMock(return_value={})
    opt.quality.aggregate_quality = MagicMock(
        return_value=QualityScore(
            overall=0.99, task_completion=1.0, factual_consistency=1.0,
            format_compliance=1.0, coding_correctness=1.0, no_truncation=1.0,
            no_malformed=1.0, checks_passed=30, checks_total=30,
        )
    )
    opt.quality.passes_threshold = MagicMock(return_value=False)
    opt.state = MagicMock()
    opt.state.recovery_log = []
    opt.state.run.advanced_settings = {}
    opt._persist_recovery_step = lambda reason: None
    opt._admit_recovered = AsyncMock(return_value=None)
    return opt


class SimpleMetrics:
    metrics = []


class TestRollbackMeasuresRolledContext:
    @pytest.mark.asyncio
    async def test_context_rollback_is_measured_at_the_new_context(self):
        opt = _optimizer()
        failed = _result(FAILED_CTX)
        last_good = LoadConfiguration(context_length=GOOD_CTX, gpu_ratio=0.8)

        await opt._try_rollback_suspect(
            "context_length", failed, last_good, FAILED_CTX, FAILED_TESTS
        )

        measured_ctx = opt.benchmark.run_cases.await_args.args[2]
        assert measured_ctx == GOOD_CTX, (
            f"rollback measured at ctx={measured_ctx}, not the rolled-back "
            f"{GOOD_CTX} - the rollback is a no-op and costs a full "
            f"{FAILED_CTX}-token measurement"
        )

    @pytest.mark.asyncio
    async def test_cases_are_built_for_the_rolled_context(self):
        opt = _optimizer()

        await opt._try_rollback_suspect(
            "context_length", _result(FAILED_CTX),
            LoadConfiguration(context_length=GOOD_CTX), FAILED_CTX, FAILED_TESTS,
        )

        built_for = [c.args[0] for c in opt.benchmark.create_cases_for_context.call_args_list]
        assert GOOD_CTX in built_for, f"cases never built for {GOOD_CTX}: {built_for}"
        assert FAILED_CTX not in built_for, (
            f"cases still built for {FAILED_CTX} - that is the wasted 32k run"
        )

    @pytest.mark.asyncio
    async def test_non_context_suspect_still_uses_its_own_context(self):
        """Rolling back gpu_ratio must NOT silently change the context."""
        opt = _optimizer()
        failed = _result(FAILED_CTX)
        last_good = LoadConfiguration(context_length=FAILED_CTX, gpu_ratio=1.0)

        await opt._try_rollback_suspect(
            "gpu_ratio", failed, last_good, FAILED_CTX, FAILED_TESTS
        )

        measured_ctx = opt.benchmark.run_cases.await_args.args[2]
        assert measured_ctx == FAILED_CTX

    @pytest.mark.asyncio
    async def test_attempt_records_the_context_actually_measured(self):
        """quality_after must describe the measurement that produced it."""
        opt = _optimizer()

        await opt._try_rollback_suspect(
            "context_length", _result(FAILED_CTX),
            LoadConfiguration(context_length=GOOD_CTX), FAILED_CTX, FAILED_TESTS,
        )

        attempt = opt.state.recovery_log[-1]
        assert attempt["suspect"] == "context_length"
        assert attempt["new_value"] == GOOD_CTX
