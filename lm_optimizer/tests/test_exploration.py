"""Task 10: fast exploration semantics (mock-only)."""
import asyncio
from datetime import datetime


def _hw():
    from lm_optimizer.domain.models import HardwareInfo

    return HardwareInfo(
        os="T", cpu_name="C", cpu_cores_physical=1, cpu_cores_logical=2,
        total_ram_gb=8, gpu_count=0,
    )


def _metric(gen=40.0):
    from lm_optimizer.domain.models import BenchmarkMetrics

    return BenchmarkMetrics(
        test_name="t", category="instruction", success=True,
        generation_tok_s=gen, prompt_tok_s=500, estimated_ttft_ms=100,
        prompt_tokens=10, completion_tokens=50, total_tokens=60,
        output_text="measured answer here",
    )


def _qs(overall=1.0):
    from lm_optimizer.domain.models import QualityScore

    return QualityScore(
        overall=overall, task_completion=1.0, factual_consistency=1.0,
        format_compliance=1.0, coding_correctness=1.0, no_truncation=1.0,
        no_malformed=1.0, checks_passed=6, checks_total=6,
    )


def _result(ctx=16384, gen=40.0, score=0.7):
    from lm_optimizer.domain.models import (
        ConfigurationResult, ConfigurationStatus, LoadConfiguration,
    )

    return ConfigurationResult(
        config=LoadConfiguration(context_length=ctx, flash_attention=True),
        context_length=ctx, status=ConfigurationStatus.PASSED,
        metrics=[_metric(gen=gen)], quality_score=_qs(),
        score=score, tested_at=datetime.now(),
        generation={"stage": "validation", "style": "balanced"},
    )


def _run(configs, skips=None):
    from lm_optimizer.domain.models import (
        ModelIdentity, OptimizationProfile, OptimizationRun,
    )

    run = OptimizationRun(
        model=ModelIdentity(id="m-9", name="M9"),
        hardware=_hw(), profile=OptimizationProfile.BALANCED,
    )
    run.configurations = configs
    for c in configs:
        c.run_id = run.id
    best = max(configs, key=lambda c: c.score or 0)
    run.best_config_id = best.id
    run.advanced_settings = {"skip_phases": list(skips or [])}
    run.benchmark_params = {"skip_phases": list(skips or [])}
    return run, best


class TestExploration:
    def test_no_quality_suite_on_explore(self):
        from lm_optimizer.services.context_sweep import pick_quality_rungs

        report = {"geometric": [{"ctx": 8192, "ok": True, "tok_s": 50.0}],
                  "refinement": []}
        assert pick_quality_rungs(report, 0) == []  # explore: suites off

    def test_skips_preserved_on_resume(self):
        from lm_optimizer.cli.main import _apply_resume_skips
        from unittest.mock import MagicMock

        run = MagicMock()
        run.advanced_settings = {"skip_phases": ["quality"]}
        run.benchmark_params = {"skip_phases": ["quality"]}
        _apply_resume_skips(run, None)  # no CLI flag → preserved
        assert run.advanced_settings["skip_phases"] == ["quality"]

    def test_exploratory_cannot_become_validated_winner(self, tmp_path):
        from lm_optimizer.services.reporting import save_best_report

        run, _ = _run([_result()], skips=["quality"])
        path = save_best_report(run, out_dir=tmp_path)
        assert path is not None and path.name.endswith("-explore.md")
        text = path.read_text(encoding="utf-8")
        assert "quality_validated: false" in text
        assert not (tmp_path / "m-9-best.md").exists()

    def test_exploratory_cannot_overwrite_validated_report(self, tmp_path):
        from lm_optimizer.services.reporting import save_best_report

        best_path = tmp_path / "m-9-best.md"
        best_path.write_text("VALIDATED WINNER ctx 32768", encoding="utf-8")
        run, _ = _run([_result()], skips=["quality"])
        out = save_best_report(run, out_dir=tmp_path)
        assert out.name.endswith("-explore.md")
        assert best_path.read_text(encoding="utf-8") == "VALIDATED WINNER ctx 32768"

    def test_rung_statuses_match_execution(self, monkeypatch):
        import lm_optimizer.cli.main as _cli
        from lm_optimizer.services import benchmark as _bench

        ok_res = _result(ctx=16384)
        calls = {"n": 0}

        class _FakeSvc:
            def __init__(self, client, benchmark_config=None):
                pass

            async def run_benchmark(self, model, cfg, ctx, style="balanced"):
                calls["n"] += 1
                if ctx == 16384:
                    return ok_res
                raise RuntimeError("boom-ctx")

        monkeypatch.setattr(_bench, "BenchmarkService", _FakeSvc)
        report = {"geometric": [{"ctx": 32768, "ok": True, "tok_s": 10.0},
                                {"ctx": 16384, "ok": True, "tok_s": 40.0}],
                  "refinement": []}
        rows = asyncio.run(
            _cli._ctx_quality_suites(
                object(), "m", report, 2, 0.0, 1, 1.0, True, True, 256,
            )
        )
        by_ctx = {r["ctx"]: r for r in rows}
        assert by_ctx[16384]["pass"] is True  # suite ran, threshold 0.0 met
        assert by_ctx[32768]["pass"] is False  # exception → recorded failure
        assert "error" in by_ctx[32768]
