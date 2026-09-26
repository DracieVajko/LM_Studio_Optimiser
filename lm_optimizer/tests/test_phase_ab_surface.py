"""Phase A/B surface: report sections, API fields, CLI flag, schema (mission §17-18, §26-27)."""

from uuid import uuid4

from lm_optimizer.domain.models import (
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    HardwareInfo,
    LoadConfiguration,
    ModelIdentity,
    OptimizationRun,
)


def _hw():
    return HardwareInfo(
        os="windows", cpu_name="c", cpu_cores_physical=8, cpu_cores_logical=16,
        total_ram_gb=16.0, gpu_count=1,
    )


def _result(speed: float, status=ConfigurationStatus.PASSED, **cfg_kw):
    cfg = LoadConfiguration(context_length=8192, **cfg_kw)
    m = BenchmarkMetrics(
        test_name="short_instruction", category="instruction", success=True,
        completion_tokens=100, generation_tok_s=speed,
    )
    return ConfigurationResult(
        id=uuid4(), config=cfg, context_length=8192, status=status,
        metrics=[m], score=0.8 if status == ConfigurationStatus.PASSED else None,
    )


def _phase_run(tmp_note="mmap note"):
    safe = _result(68.4)
    raw = _result(72.1, status=ConfigurationStatus.QUALITY_FAILED)
    run = OptimizationRun(model=ModelIdentity(id="m", name="m"), hardware=_hw())
    run.configurations = [safe, raw]
    run.best_config_id = safe.id
    run.benchmark_params = {
        "phase_ab": {
            "phase": "final_validation",
            "speed_context": 2048,
            "quality_context": 8192,
            "budget_class": "NORMAL",
            "speed_finalist_ids": [str(safe.id)],
            "raw_fastest_id": str(raw.id),
            "recovery_log": [
                {"suspect": "flash_attention", "old_value": True,
                 "new_value": False, "recheck_passed": True, "quality_after": 0.99}
            ],
            "excluded_notes": [tmp_note],
            "phase_b_enabled": True,
            "phase_b_result": {
                "gpu_only_max": 16384, "system_max": 32768, "model_limit": 32768,
                "hardware_stable_limit": 32768, "final_capacity": 32768,
            },
        }
    }
    return run


class TestPhaseReport:
    def test_raw_vs_safe_recovery_context_sections(self, tmp_path):
        from lm_optimizer.services.reporting import save_best_report

        text = save_best_report(_phase_run(), out_dir=tmp_path).read_text(encoding="utf-8")
        assert "## Phase A" in text
        assert "72.1" in text  # raw fastest shown even though failed
        assert "68.4" in text  # quality-safe winner
        assert "flash_attention" in text  # recovery culprit
        assert "mmap note" in text  # excluded (manual-only) params
        assert "## Phase B" in text
        assert "16384" in text and "32768" in text  # gpu-only vs system max

    def test_legacy_run_has_no_phase_section(self, tmp_path):
        from lm_optimizer.services.reporting import save_best_report

        run = OptimizationRun(model=ModelIdentity(id="m", name="m"), hardware=_hw())
        run.configurations = [_result(50.0)]
        run.best_config_id = run.configurations[0].id
        text = save_best_report(run, out_dir=tmp_path).read_text(encoding="utf-8")
        assert "## Phase A" not in text


class TestApiSurface:
    def test_convert_run_exposes_phase(self):
        from lm_optimizer.api.routes import _convert_run

        resp = _convert_run(_phase_run())
        assert resp.phase == "final_validation"
        assert (
            str(resp.raw_fastest_config_id)
            == resp.benchmark_params["phase_ab"]["raw_fastest_id"]
        )
        assert resp.phase_b_enabled is True
        assert resp.phase_b_result["gpu_only_max"] == 16384

    def test_legacy_convert_defaults(self):
        from lm_optimizer.api.routes import _convert_run

        run = OptimizationRun(model=ModelIdentity(id="m", name="m"), hardware=_hw())
        resp = _convert_run(run)
        assert resp.phase == "speed"
        assert resp.raw_fastest_config_id is None
        assert resp.phase_b_enabled is False

    def test_schema_opt_in(self):
        from lm_optimizer.api.schemas import AdvancedSettingsSchema

        assert AdvancedSettingsSchema().optimize_context is False
        assert AdvancedSettingsSchema(optimize_context=True).optimize_context is True


class TestCliDisplay:
    def test_raw_vs_safe_shown(self, capsys):
        from lm_optimizer.cli.main import _display_optimization_result

        _display_optimization_result(_phase_run())
        out = capsys.readouterr().out
        assert "Raw fastest" in out
        assert "72.1" in out
        assert "Quality-safe" in out
        assert "flash_attention" in out
        assert "GPU-only max" in out

    def test_legacy_no_phase_block(self, capsys):
        from lm_optimizer.cli.main import _display_optimization_result

        run = OptimizationRun(model=ModelIdentity(id="m", name="m"), hardware=_hw())
        run.configurations = [_result(50.0)]
        run.best_config_id = run.configurations[0].id
        _display_optimization_result(run)
        out = capsys.readouterr().out
        assert "Raw fastest" not in out


class TestCliSurface:
    def test_optimize_has_context_flag(self):
        import typer

        from lm_optimizer.cli.main import app

        names = set()
        for cmd in (app.registered_commands or []):
            if getattr(cmd, "name", "") == "optimize" or (
                getattr(cmd, "callback", None)
                and getattr(cmd.callback, "__name__", "") == "optimize"
            ):
                info = typer.main.get_command(app)
                params = [p.name for p in info.commands["optimize"].params]
                names.update(params)
        assert "optimize_context" in names
