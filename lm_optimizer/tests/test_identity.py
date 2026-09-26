"""Model/run identity: reports follow the run's own model_id (§6-§7, §33-§34)."""

from datetime import datetime
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


def _hw():
    return HardwareInfo(
        os="T", cpu_name="C", cpu_cores_physical=1, cpu_cores_logical=2, total_ram_gb=8, gpu_count=0
    )


def _run(model_id, name, quant=None, arch=None, size=None, failed=False):
    run = OptimizationRun(
        model=ModelIdentity(
            id=model_id, name=name, quantization=quant, architecture=arch, size_bytes=size
        ),
        hardware=_hw(),
        profile=OptimizationProfile.BALANCED,
    )
    metrics = [
        BenchmarkMetrics(
            test_name="t",
            category="instruction",
            success=True,
            generation_tok_s=40.0,
            prompt_tok_s=500,
            estimated_ttft_ms=100,
            completion_tokens=50,
            output_text="ok",
        )
    ]
    qs = QualityScore(
        overall=0.98,
        task_completion=1.0,
        factual_consistency=1.0,
        format_compliance=1.0,
        coding_correctness=1.0,
        no_truncation=1.0,
        no_malformed=1.0,
        checks_passed=6,
        checks_total=6,
    )
    good = ConfigurationResult(
        config=LoadConfiguration(context_length=4096),
        context_length=4096,
        status=ConfigurationStatus.PASSED,
        metrics=metrics,
        quality_score=qs,
        score=0.7,
        score_breakdown={"generation_speed": 0.8},
        tested_at=datetime.now(),
    )
    good.run_id = run.id
    run.configurations = [good]
    if failed:
        bad = ConfigurationResult(
            config=LoadConfiguration(context_length=8192),
            context_length=8192,
            status=ConfigurationStatus.LOAD_FAILED,
            error="Model load failed: refused",
            tested_at=datetime.now(),
        )
        bad.run_id = run.id
        run.configurations.append(bad)
    run.best_config_id = good.id
    return run


class TestReportIdentity:
    def test_filename_follows_run_model_not_order(self, tmp_path):
        from lm_optimizer.services.reporting import save_best_report

        runs = [
            _run("qwen3.8-4b", "Qwen3.8", quant="Q4_K_M", arch="qwen3", size=1),
            _run("mistral-7b", "Mistral", quant="Q5_K_M", arch="mistral", size=2),
            _run("qwen3.5-4b", "Qwen3.5", quant="Q4_K_M", arch="qwen3", size=3),
        ]
        # Scrambled execution order must not affect naming.
        paths = [save_best_report(r, out_dir=tmp_path) for r in reversed(runs)]
        names = sorted(p.name for p in paths)
        assert names == ["mistral-7b-best.md", "qwen3.5-4b-best.md", "qwen3.8-4b-best.md"]

    def test_qwen38_qwen35_separated(self, tmp_path):
        from lm_optimizer.services.reporting import save_best_report

        p38 = save_best_report(_run("qwen3.8-4b-sft-fable5-glint", "glint"), out_dir=tmp_path)
        p35 = save_best_report(_run("qwen3.5-4b", "qwen"), out_dir=tmp_path)
        assert p38.name == "qwen3.8-4b-sft-fable5-glint-best.md"
        assert p35.name == "qwen3.5-4b-best.md"
        assert "qwen3.8-4b-sft-fable5-glint" in p38.read_text()
        assert "qwen3.5-4b" not in p38.read_text().replace("qwen3.8-4b-sft-fable5-glint", "")

    def test_report_carries_identity_content(self, tmp_path):
        from lm_optimizer.services.reporting import save_best_report

        run = _run("m-1", "Display", quant="Q4", arch="llama", size=99, failed=True)
        path = save_best_report(
            run,
            out_dir=tmp_path,
            host={
                "storage": {
                    "drives": [],
                    "pagefile": "C:",
                    "swap_free_gb": 1.0,
                    "swap_total_gb": 2.0,
                }
            },
        )
        text = path.read_text()
        assert str(run.id) in text  # Run ID
        assert "Display" in text and "Q4" in text and "llama" in text
        assert "Control channels" in text and "Requested vs applied" in text
        assert "Maximum stable context" in text
        assert "Storage / swap" in text
        assert "LOAD_FAILED" in text or "load_failed" in text

    def test_configurations_carry_run_id(self):
        run = _run("m-9", "M", failed=True)
        assert {str(c.run_id) for c in run.configurations} == {str(run.id)}
