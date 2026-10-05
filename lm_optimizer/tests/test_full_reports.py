"""Full report visibility: all-tried-configs table (Task 1)."""

from lm_optimizer.domain.models import (
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
    QualityScore,
)


def _metric(gen=30.0, ok=True):
    return BenchmarkMetrics(
        test_name="t",
        category="instruction",
        success=ok,
        generation_tok_s=gen,
        prompt_tok_s=500,
        estimated_ttft_ms=100,
        prompt_tokens=10,
        completion_tokens=50,
        total_tokens=60,
        output_text="measured",
    )


def _qs(overall=1.0):
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


def passed_cfg(score, ctx=16384):
    return ConfigurationResult(
        config=LoadConfiguration(
            context_length=ctx,
            flash_attention=True,
            offload_kv_cache_to_gpu=True,
            eval_batch_size=512,
            physical_batch_size=512,
            parallel=4,
            context_checkpoints=32,
        ),
        context_length=ctx,
        status=ConfigurationStatus.PASSED,
        metrics=[_metric()],
        quality_score=_qs(),
        score=score,
    )


def failed_cfg():
    return ConfigurationResult(
        config=LoadConfiguration(context_length=8192),
        context_length=8192,
        status=ConfigurationStatus.LOAD_FAILED,
        metrics=[],
        quality_score=None,
        score=None,
        error="load refused: KV quant mismatch",
    )


def test_all_configs_table_lists_every_config():
    from lm_optimizer.services.reporting import _all_configs_rows

    rows = _all_configs_rows([passed_cfg(9.1), failed_cfg(), passed_cfg(7.7)], best_id=passed_cfg(9.1).id)
    text = "\n".join(rows)
    assert text.count("|") > 0
    assert text.index("9.1") < text.index("7.7") < text.index("load_failed")


def test_missing_metrics_render_na():
    from lm_optimizer.services.reporting import _all_configs_rows

    rows = _all_configs_rows([failed_cfg()], best_id=None)
    text = "\n".join(rows)
    assert "n/a" in text
    assert "load_failed" in text


def test_pipe_escaping_and_error_trim():
    from lm_optimizer.services.reporting import _all_configs_rows

    bad = failed_cfg()
    bad.error = "a|b " + "x" * 200
    rows = _all_configs_rows([bad], best_id=None)
    text = "\n".join(rows)
    assert "a\\|b" in text
    assert "x" * 200 not in text


def test_best_report_contains_all_configs_table(tmp_path):
    from lm_optimizer.domain.models import (
        HardwareInfo,
        ModelIdentity,
        OptimizationProfile,
        OptimizationRun,
    )
    from lm_optimizer.services.reporting import save_best_report

    run = OptimizationRun(
        model=ModelIdentity(id="m-1", name="M1"),
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
    hi = passed_cfg(9.1)
    lo = passed_cfg(7.7, ctx=8192)
    run.configurations = [hi, failed_cfg(), lo]
    for c in run.configurations:
        c.run_id = run.id
    run.best_config_id = hi.id
    text = save_best_report(run, out_dir=str(tmp_path)).read_text(encoding="utf-8")
    assert "All tried configurations" in text
    assert text.index("9.100") < text.index("7.700") < text.index("load_failed")
