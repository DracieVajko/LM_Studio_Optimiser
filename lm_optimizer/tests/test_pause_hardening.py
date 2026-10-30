"""Task 9: pause/resume hardening.

Pause during SPEED/QUALITY/RECOVERY/validation; durable checkpoint;
same-ID resume; no re-run of completed; no duplicate workers. Mock-only.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock


def _opt():
    from lm_optimizer.domain.models import (
        HardwareInfo, ModelIdentity, OptimizationProfile, OptimizationRun,
        OptimizationStage, RunStatus,
    )
    from lm_optimizer.services import optimizer as _opt

    opt = _opt.AdaptiveOptimizer.__new__(_opt.AdaptiveOptimizer)
    run = OptimizationRun(
        model=ModelIdentity(id="m", name="m"),
        hardware=HardwareInfo(
            os="T", cpu_name="C", cpu_cores_physical=1,
            cpu_cores_logical=2, total_ram_gb=8, gpu_count=0,
        ),
        profile=OptimizationProfile.BALANCED,
    )
    run.status = RunStatus.RUNNING
    run.stage = OptimizationStage.DISCOVERY
    run.benchmark_params = {}
    run.advanced_settings = {}
    state = MagicMock()
    state.run = run
    state.tested_configs = []
    state.event_log = []
    state.errors = []
    state.phase = "speed"
    state.current_config_id = None
    state.recovery_cursor = None
    state.should_pause = False
    opt.state = state
    opt.client = MagicMock()
    opt.client.ensure_unloaded = AsyncMock(return_value=True)
    return opt, _opt


class TestPauseContract:
    def test_pause_point_raises_when_flag_set(self):
        from lm_optimizer.services.optimizer import PauseRequested

        opt, _ = _opt()
        opt.state.should_pause = True
        opt._checkpoint = MagicMock()
        opt._sync_phase_params = MagicMock()
        try:
            asyncio.run(opt._pause_point())
            raise AssertionError("should have raised")
        except PauseRequested:
            pass

    def test_pause_point_quiet_when_no_flag(self):
        opt, _ = _opt()
        asyncio.run(opt._pause_point())  # must not raise

    def test_pause_now_persists_and_marks_paused(self, monkeypatch, tmp_path):
        from lm_optimizer.domain.models import RunStatus
        from lm_optimizer.services import optimizer as _optmod
        from lm_optimizer.services.optimizer import PauseRequested

        opt, _ = _opt()
        monkeypatch.setattr(_optmod.run_repo, "save", lambda r: "r1")
        opt._checkpoint = MagicMock()
        opt._sync_phase_params = MagicMock()
        opt._broadcast = AsyncMock()
        monkeypatch.setattr(
            opt, "_pause_flag_path", lambda *a, **k: tmp_path / "pause_x.flag"
        )
        try:
            asyncio.run(opt._pause_now(reason="user_requested"))
            raise AssertionError("should have raised")
        except PauseRequested:
            pass
        assert opt.state.run.status == RunStatus.PAUSED
        opt._checkpoint.assert_called_once()
        assert "paused:user_requested" in str(opt._checkpoint.call_args)

    def test_phases_contain_pause_checks(self):
        import inspect

        from lm_optimizer.services.optimizer import AdaptiveOptimizer

        for fn in ("_phase_speed", "_phase_quality", "_recover_failed",
                   "_phase_final_validation"):
            src = inspect.getsource(getattr(AdaptiveOptimizer, fn))
            assert "_pause_point" in src or "should_pause" in src, fn

    def test_resume_keeps_same_id_no_new_run(self):
        import inspect

        from lm_optimizer.services.optimizer import AdaptiveOptimizer

        src = inspect.getsource(AdaptiveOptimizer.resume_from_checkpoint)
        assert "OptimizationRun(" not in src  # rebuilds from DB, same id
        assert "run_repo.get" in src

    def test_no_duplicate_workers_locks(self):
        import inspect

        from lm_optimizer.cli import main as _cli

        assert "_acquire_run_lock" in inspect.getsource(_cli.optimize)
        assert "_acquire_run_lock" in inspect.getsource(_cli.auto)
        assert "_acquire_run_lock" in inspect.getsource(_cli.resume)
        assert "_acquire_run_lock" in inspect.getsource(_cli._run_resume)

    def test_pause_command_writes_flag_file(self, monkeypatch, tmp_path):
        from typer.testing import CliRunner

        from lm_optimizer.cli import main as _cli

        monkeypatch.setattr(
            "lm_optimizer.storage.run_checkpoint.checkpoint_dir", lambda: tmp_path
        )
        monkeypatch.setattr(_cli, "_resolve_run_id", lambda prefix: "run-abc")
        import lm_optimizer.storage.run_checkpoint as _ckpt

        monkeypatch.setattr(_ckpt, "checkpoint_dir", lambda: tmp_path)
        result = CliRunner().invoke(_cli.app, ["pause", "run-abc"])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "pause_run-abc.flag").exists()
