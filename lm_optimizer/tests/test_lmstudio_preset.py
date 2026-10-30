"""LM Studio preset export (Priority 1).

LM Studio has no direct preset API: the supported workflow is a validated,
importable JSON preset built from the run's ACTUAL winning candidate, with
recorded-only (REST-unsupported) params reported explicitly.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

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
from lm_optimizer.services.preset_export import (
    PRESET_FORMAT,
    build_lmstudio_preset,
    default_preset_filename,
    parse_lmstudio_preset,
    preset_from_run,
    render_lmstudio_preset,
    write_lmstudio_preset,
)


def _qs(overall=0.975):
    return QualityScore(
        overall=overall, task_completion=1.0, factual_consistency=1.0,
        format_compliance=1.0, coding_correctness=1.0, no_truncation=1.0,
        no_malformed=1.0, checks_passed=6, checks_total=6,
    )


def _metrics(gen=31.5):
    return [BenchmarkMetrics(
        test_name="t", category="instruction", success=True,
        generation_tok_s=gen, prompt_tok_s=500, estimated_ttft_ms=100,
        prompt_tokens=10, completion_tokens=50, total_tokens=60, output_text="ok",
    )]


def _winner(ctx=32768, gpu_ratio=0.85):
    return ConfigurationResult(
        config=LoadConfiguration(
            context_length=ctx, gpu_ratio=gpu_ratio, flash_attention=True,
            offload_kv_cache_to_gpu=True, eval_batch_size=512,
        ),
        context_length=ctx, status=ConfigurationStatus.PASSED,
        metrics=_metrics(), quality_score=_qs(), score=0.9,
        tested_at=datetime.now(),
    )


class TestBuild:
    def test_context_and_params_preserved(self):
        preset = build_lmstudio_preset(
            model_id="qwen/qwen3-coder-30b", profile="balanced",
            config=_winner().config.to_dict(), context_length=32768,
            run_id="r1", best_config_id="c1",
        )
        assert preset["model_id"] == "qwen/qwen3-coder-30b"
        assert preset["context_length"] == 32768
        assert preset["applied_params"]["context_length"] == 32768
        assert preset["applied_params"]["flash_attention"] is True
        assert preset["applied_params"]["eval_batch_size"] == 512
        assert preset["source_run_id"] == "r1"
        assert preset["best_config_id"] == "c1"

    def test_gpu_ratio_recorded_not_applied(self):
        preset = build_lmstudio_preset(
            model_id="m", profile="balanced",
            config=_winner(gpu_ratio=0.85).config.to_dict(), context_length=32768,
        )
        assert "gpu_ratio" not in preset["applied_params"]
        assert preset["recorded_only_params"]["gpu_ratio"] == 0.85
        assert "gpu_ratio" in preset["unsupported_params"]

    def test_outer_context_overrides_stale_nested(self):
        # Nested config lags (4096) while the winner's real ctx is 32768.
        cfg = _winner(ctx=4096).config.to_dict()
        preset = build_lmstudio_preset(
            model_id="m", profile="balanced", config=cfg, context_length=32768,
        )
        assert preset["applied_params"]["context_length"] == 32768

    def test_empty_config_raises(self):
        with pytest.raises(ValueError):
            build_lmstudio_preset(model_id="m", profile="balanced", config={})


class TestRoundTrip:
    def test_render_parse_round_trip(self):
        preset = build_lmstudio_preset(
            model_id="m", profile="balanced",
            config=_winner().config.to_dict(), context_length=32768,
        )
        parsed = parse_lmstudio_preset(render_lmstudio_preset(preset))
        assert parsed["applied_params"]["context_length"] == 32768
        assert parsed["format"] == PRESET_FORMAT

    def test_parse_rejects_bad_format(self):
        with pytest.raises(ValueError):
            parse_lmstudio_preset(json.dumps({"format": "nope", "applied_params": {"context_length": 1}}))

    def test_parse_rejects_missing_context(self):
        bad = json.dumps({"format": PRESET_FORMAT, "applied_params": {"flash_attention": True}})
        with pytest.raises(ValueError):
            parse_lmstudio_preset(bad)

    def test_parse_rejects_non_json(self):
        with pytest.raises(ValueError):
            parse_lmstudio_preset("not json{")

    def test_write_and_read_back(self, tmp_path):
        preset = build_lmstudio_preset(
            model_id="m/x", profile="balanced",
            config=_winner().config.to_dict(), context_length=32768,
        )
        path = write_lmstudio_preset(preset, out_dir=tmp_path)
        assert path.exists() and path.parent == tmp_path
        assert parse_lmstudio_preset(path.read_text(encoding="utf-8"))

    def test_default_filename_is_safe(self):
        name = default_preset_filename("qwen/qwen3-coder-30b", "balanced")
        assert "/" not in name and name.endswith(".preset.json")


class TestPresetFromRun:
    def test_no_winner_raises(self):
        run = OptimizationRun(
            model=ModelIdentity(id="m", name="m"),
            hardware=HardwareInfo(os="T", cpu_name="C", cpu_cores_physical=1,
                                  cpu_cores_logical=2, total_ram_gb=8, gpu_count=0),
            profile=OptimizationProfile.BALANCED,
        )
        run.configurations = []
        with pytest.raises(ValueError):
            preset_from_run(run)

    def test_winner_produces_preset(self):
        run = OptimizationRun(
            model=ModelIdentity(id="m", name="m"),
            hardware=HardwareInfo(os="T", cpu_name="C", cpu_cores_physical=1,
                                  cpu_cores_logical=2, total_ram_gb=8, gpu_count=0),
            profile=OptimizationProfile.BALANCED,
        )
        best = _winner(ctx=32768)
        run.configurations = [best]
        run.best_config_id = best.id
        preset = preset_from_run(run)
        assert preset["context_length"] == 32768
        assert preset["best_config_id"] == str(best.id)


class TestCliExport:
    def _save(self, model="m", profile="balanced", config=None):
        from lm_optimizer.database.repositories import model_repo, preset_repo

        model_repo.save(ModelIdentity(id=model, name=model))
        return preset_repo.save({
            "model_id": model, "profile": profile,
            "config": config or _winner().config.to_dict(),
            "context_length": 32768, "best_config_id": "c1",
            "metrics": {"generation_tok_s": 31.5, "quality_score": 0.975},
            "quality": {"overall": 0.975}, "run_id": None,
            "optimizer_version": "1.0.0-beta",
        })

    def test_export_writes_importable_file(self, tmp_path):
        from lm_optimizer.cli.main import app

        self._save(model="m")
        out = tmp_path / "e.json"
        r = CliRunner().invoke(app, ["preset-export", "m", "--output", str(out)])
        assert r.exit_code == 0, r.output
        assert out.exists()
        data = parse_lmstudio_preset(out.read_text(encoding="utf-8"))
        assert data["applied_params"]["context_length"] == 32768
        assert "Exported" in r.output

    def test_export_default_dir(self, tmp_path, monkeypatch):
        import lm_optimizer.services.preset_export as pe
        from lm_optimizer.cli.main import app

        monkeypatch.chdir(tmp_path)
        self._save(model="m")
        r = CliRunner().invoke(app, ["preset-export", "m"])
        assert r.exit_code == 0, r.output
        assert (tmp_path / "lmstudio-presets").exists()

    def test_export_missing_preset_exits_1(self):
        from lm_optimizer.cli.main import app

        r = CliRunner().invoke(app, ["preset-export", "nope"])
        assert r.exit_code == 1
        assert "No saved preset" in r.output

    def test_export_reports_unsupported(self, tmp_path):
        from lm_optimizer.cli.main import app

        self._save(model="g", config=_winner(gpu_ratio=0.9).config.to_dict())
        out = tmp_path / "g.json"
        r = CliRunner().invoke(app, ["preset-export", "g", "--output", str(out)])
        assert r.exit_code == 0, r.output
        assert "gpu_ratio" in r.output
