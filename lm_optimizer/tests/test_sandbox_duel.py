"""Tests for sandbox-duel CLI command."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from lm_optimizer.domain.models import (
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
    RunStatus,
)


def _winner_run(monkeypatch, model_id: str, ctx: int = 8192):
    """Seed a run whose best_config_id resolves to a PASSED configuration."""
    from lm_optimizer.database import repositories as _repo
    from lm_optimizer.domain.models import HardwareInfo

    hw = HardwareInfo(
        os="windows", cpu_name="c", cpu_cores_physical=8, cpu_cores_logical=16,
        total_ram_gb=16.0, gpu_count=1,
    )
    _repo.hardware_repo.save(hw)
    _repo.model_repo.save(ModelIdentity(id=model_id, name=model_id))

    run = OptimizationRun(
        model=ModelIdentity(id=model_id, name=model_id),
        hardware=hw,
        profile=OptimizationProfile.BALANCED,
    )
    run.status = RunStatus.SUCCESS
    _repo.run_repo.save(run)

    cfg = LoadConfiguration(
        context_length=ctx, flash_attention=True, offload_kv_cache_to_gpu=True,
        eval_batch_size=2048, physical_batch_size=512, parallel=4, context_checkpoints=32,
    )
    best = ConfigurationResult(
        run_id=run.id, config=cfg, context_length=ctx,
        status=ConfigurationStatus.PASSED, score=0.9,
        metrics=[BenchmarkMetrics(
            test_name="t", category="c", success=True,
            generation_tok_s=20.0, prompt_tok_s=8.0, estimated_ttft_ms=140.0,
        )],
    )
    _repo.config_repo.save(best)
    run.best_config_id = best.id
    _repo.run_repo.save(run)
    return run


class TestLoadBestConfig:
    def test_loads_validated_winner(self, monkeypatch):
        from lm_optimizer.cli.sandbox_duel import _load_best_config

        _winner_run(monkeypatch, "duel-model-a", ctx=8192)
        cfg, ctx = _load_best_config("duel-model-a")
        assert ctx == 8192
        assert cfg.eval_batch_size == 2048

    def test_no_runs_refuses_to_guess(self, monkeypatch):
        from lm_optimizer.cli.sandbox_duel import _load_best_config

        with pytest.raises(ValueError, match="No runs for model"):
            _load_best_config("never-optimized-model")

    def test_run_without_winner_refuses(self, monkeypatch):
        """A run with no validated winner must not be dueled on a guess."""
        from lm_optimizer.cli.sandbox_duel import _load_best_config
        from lm_optimizer.database import repositories as _repo
        from lm_optimizer.domain.models import HardwareInfo

        hw = HardwareInfo(
            os="windows", cpu_name="c", cpu_cores_physical=8, cpu_cores_logical=16,
            total_ram_gb=16.0, gpu_count=1,
        )
        _repo.hardware_repo.save(hw)
        _repo.model_repo.save(ModelIdentity(id="winnerless", name="winnerless"))
        run = OptimizationRun(
            model=ModelIdentity(id="winnerless", name="winnerless"),
            hardware=hw, profile=OptimizationProfile.BALANCED,
        )
        run.status = RunStatus.COMPLETED_NO_WINNER
        _repo.run_repo.save(run)

        with pytest.raises(ValueError, match="No validated winner"):
            _load_best_config("winnerless")


class TestDuelMedians:
    def _side(self, model_id, speeds, ttfits, qualities):
        from lm_optimizer.cli.sandbox_duel import DuelModelResult, DuelRun

        side = DuelModelResult(
            model_id=model_id,
            config=LoadConfiguration(context_length=8192),
            context_length=8192,
        )
        for sp, tt, q in zip(speeds, ttfits, qualities):
            side.runs.append(DuelRun(
                ttft_ms=tt, generation_tok_s=sp, prompt_tok_s=8.0,
                quality_overall=q, passed=True,
            ))
        return side

    def test_medians_use_robust_central_value(self):
        a = self._side("m1", speeds=[10.0, 30.0, 20.0], ttfits=[300, 100, 200], qualities=[0.9, 0.9, 0.9])
        assert a.median_gen() == 20.0
        assert a.median_ttft() == 200.0
        assert a.median_quality() == 0.9

    def test_verdict_picks_faster_side(self):
        from lm_optimizer.cli.sandbox_duel import DuelResult

        a = self._side("m1", speeds=[40.0], ttfits=[100], qualities=[0.9])
        b = self._side("m2", speeds=[20.0], ttfits=[200], qualities=[0.95])
        verdict = DuelResult(prompt="p", model_a=a, model_b=b).verdict()

        assert verdict["TTFT"]["winner"] == "a"
        assert verdict["Generation speed"]["winner"] == "a"
        assert verdict["Quality"]["winner"] == "b"
        assert verdict["TTFT"]["margin_pct"] == 50.0

    def test_unmeasured_metrics_do_not_crash(self):
        """A side with no measurable runs must still produce a verdict row."""
        from lm_optimizer.cli.sandbox_duel import DuelModelResult, DuelResult

        empty = DuelModelResult(
            model_id="empty", config=LoadConfiguration(context_length=4096), context_length=4096
        )
        real = self._side("real", speeds=[30.0], ttfits=[120], qualities=[0.9])
        verdict = DuelResult(prompt="p", model_a=empty, model_b=real).verdict()
        assert verdict["TTFT"]["winner"] == "b"
        assert verdict["TTFT"]["margin_pct"] == 0.0


class TestDuelReport:
    def test_report_contains_config_results_verdict(self, tmp_path):
        from lm_optimizer.cli.sandbox_duel import (
            DuelModelResult,
            DuelResult,
            DuelRun,
            generate_duel_report,
        )

        a = DuelModelResult(
            model_id="qwen3-32b",
            config=LoadConfiguration(
                context_length=8192, flash_attention=True, offload_kv_cache_to_gpu=True,
                eval_batch_size=2048,
            ),
            context_length=8192,
        )
        a.runs.append(DuelRun(ttft_ms=140, generation_tok_s=28.4, prompt_tok_s=8.0, quality_overall=0.98, passed=True))

        b = DuelModelResult(
            model_id="llama3.1-8b",
            config=LoadConfiguration(
                context_length=4096, flash_attention=True, offload_kv_cache_to_gpu=True,
                eval_batch_size=1024,
            ),
            context_length=4096,
        )
        b.runs.append(DuelRun(ttft_ms=89, generation_tok_s=42.3, prompt_tok_s=12.0, quality_overall=0.95, passed=True))

        md = generate_duel_report(
            DuelResult(prompt="Write a fibonacci function", model_a=a, model_b=b),
            tmp_path,
            {"gpu": "RTX 3060 6.0GB", "cpu": "AMD 8P/16L", "lms": "0.3.12"},
        )

        assert "# Duel: qwen3-32b vs llama3.1-8b" in md
        assert "## Configurations" in md
        assert "## Prompt" in md
        assert "fibonacci" in md
        assert "## Results" in md
        assert "## Medians" in md
        assert "## Verdict" in md
        assert "RTX 3060" in md
        assert "llama3.1-8b" in md
        assert list(tmp_path.glob("duel-*.md")), "report file must be written"

    def test_report_marks_load_failure(self, tmp_path):
        from lm_optimizer.cli.sandbox_duel import (
            DuelModelResult,
            DuelResult,
            generate_duel_report,
        )

        broken = DuelModelResult(
            model_id="broken", config=LoadConfiguration(context_length=4096), context_length=4096
        )
        broken.error = "Load failed: OOM"
        working = DuelModelResult(
            model_id="ok", config=LoadConfiguration(context_length=4096), context_length=4096
        )

        md = generate_duel_report(
            DuelResult(prompt="p", model_a=broken, model_b=working), tmp_path, {}
        )
        assert "ERROR: Load failed: OOM" in md


class TestDuelCliSurface:
    def test_command_registered(self):
        from typer.testing import CliRunner

        from lm_optimizer.cli.main import app

        result = CliRunner().invoke(app, ["--help"])
        assert result.exit_code == 0, result.output
        assert "sandbox-duel" in result.output

    def test_missing_winner_exits_cleanly(self, monkeypatch):
        from typer.testing import CliRunner

        from lm_optimizer.cli.main import app

        result = CliRunner().invoke(
            app,
            ["sandbox-duel", "--model-a", "duel-absent-a", "--model-b", "duel-absent-b",
             "--prompt", "hi"],
        )
        assert result.exit_code == 1
        assert "No runs for model" in result.output or "No validated winner" in result.output