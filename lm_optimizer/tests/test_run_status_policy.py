"""Run-level status policy (reliability fix).

Run status is never the worst candidate:
- SUCCESS: a validated winner exists, even if other candidates failed.
- COMPLETED_NO_WINNER: completed, model produced usable output, no config met
  the quality threshold (honest terminal state, not a failure).
- FAILED: nothing produced usable output (model never loaded / unusable).
"""

from datetime import datetime

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
    RunStatus,
)
from lm_optimizer.services.run_summary import (
    best_passed,
    classify_run,
    final_status_for_run,
)


def _metrics(gen=40.0):
    return [
        BenchmarkMetrics(
            test_name="t",
            category="instruction",
            success=True,
            generation_tok_s=gen,
            prompt_tok_s=500,
            estimated_ttft_ms=100,
            prompt_tokens=10,
            completion_tokens=50,
            total_tokens=60,
            output_text="measured coherent answer",
        )
    ]


def _qs(overall=0.99):
    return QualityScore(
        overall=overall,
        task_completion=1.0,
        factual_consistency=1.0,
        format_compliance=1.0,
        coding_correctness=1.0,
        no_truncation=1.0,
        no_malformed=1.0,
        checks_passed=6,
        checks_total=6,
    )


def _cfg(ctx=4096, status=ConfigurationStatus.PASSED, score=0.8, quality=True, err=None):
    return ConfigurationResult(
        config=LoadConfiguration(context_length=ctx, flash_attention=True),
        context_length=ctx,
        status=status,
        metrics=_metrics(),
        quality_score=_qs() if quality else None,
        score=score,
        error=err,
        tested_at=datetime.now(),
    )


def _run(configs):
    r = OptimizationRun(
        model=ModelIdentity(id="m", name="m"),
        hardware=HardwareInfo(
            os="T",
            cpu_name="C",
            cpu_cores_physical=1,
            cpu_cores_logical=2,
            total_ram_gb=8,
            gpu_count=0,
        ),
        profile=OptimizationProfile.BALANCED,
    )
    r.configurations = list(configs)
    return r


class TestRunStatusPolicy:
    def test_winner_plus_load_failures_is_success(self):
        cfgs = [
            _cfg(ctx=4096, score=0.8),
            _cfg(ctx=8192, status=ConfigurationStatus.LOAD_FAILED, score=None, quality=False, err="load"),
            _cfg(ctx=16384, status=ConfigurationStatus.LOAD_FAILED, score=None, quality=False, err="load"),
        ]
        state, info = classify_run(cfgs, _run(cfgs))
        assert state == "SUCCESS"
        assert info["validated_winner"] is True
        assert final_status_for_run(_run(cfgs)) == RunStatus.SUCCESS

    def test_winner_plus_quality_failures_is_success(self):
        cfgs = [
            _cfg(ctx=4096, score=0.85),
            _cfg(ctx=8192, status=ConfigurationStatus.QUALITY_FAILED, score=None, quality=True),
            _cfg(ctx=16384, status=ConfigurationStatus.QUALITY_FAILED, score=None, quality=True),
        ]
        assert classify_run(cfgs, _run(cfgs))[0] == "SUCCESS"
        assert final_status_for_run(_run(cfgs)) == RunStatus.SUCCESS

    def test_completed_no_validated_winner(self):
        cfgs = [
            _cfg(ctx=4096, status=ConfigurationStatus.QUALITY_FAILED, score=None, quality=True),
            _cfg(ctx=8192, status=ConfigurationStatus.QUALITY_FAILED, score=None, quality=True),
        ]
        state, info = classify_run(cfgs, _run(cfgs))
        assert state == "NO_WINNER"
        assert info["validated_winner"] is False
        assert info["operational"] is True
        assert final_status_for_run(_run(cfgs)) == RunStatus.COMPLETED_NO_WINNER

    def test_all_load_failed_is_failed(self):
        cfgs = [
            _cfg(ctx=4096, status=ConfigurationStatus.LOAD_FAILED, score=None, quality=False, err="refused"),
            _cfg(ctx=8192, status=ConfigurationStatus.OOM, score=None, quality=False, err="OOM"),
            _cfg(ctx=16384, status=ConfigurationStatus.INCOMPATIBLE, score=None, quality=False, err="unsupported"),
        ]
        state, _ = classify_run(cfgs, _run(cfgs))
        assert state == "FAILED"
        assert final_status_for_run(_run(cfgs)) == RunStatus.FAILED

    def test_no_configs_is_failed(self):
        assert classify_run([], _run([]))[0] == "FAILED"
        assert final_status_for_run(_run([])) == RunStatus.FAILED

    def test_unusable_output_below_viability_is_failed(self):
        # Model loaded but produced gibberish / 0 tok/s and no usable output:
        # no passed candidate and no quality evaluation evidence.
        cfg = _cfg(ctx=4096, status=ConfigurationStatus.BENCHMARK_FAILED, score=None, quality=False, err="gibberish")
        cfg.metrics = []
        assert classify_run([cfg], _run([cfg]))[0] == "FAILED"

    def test_best_passed_never_returns_failed_or_unscored(self):
        bad = _cfg(ctx=4096, status=ConfigurationStatus.FAILED, score=0.99, quality=True, err="boom")
        good = _cfg(ctx=8192, status=ConfigurationStatus.PASSED, score=0.4)
        assert best_passed([bad, good]).context_length == 8192
        assert best_passed([bad]) is None

    def test_legacy_partial_success_still_enum(self):
        assert RunStatus.PARTIAL_SUCCESS.value == "partial_success"
        assert RunStatus.COMPLETED_NO_WINNER.value == "completed_no_validated_winner"
