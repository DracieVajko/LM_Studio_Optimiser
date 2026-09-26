"""Pause/resume mechanism regressions. No real models; mocked seams + tmp dirs."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from lm_optimizer.domain.models import (
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
    ModelIdentity,
    OptimizationRun,
    QualityScore,
)
from lm_optimizer.services.optimizer import AdaptiveOptimizer, OptimizationState
from lm_optimizer.services.search_space import SearchSpace


def _model():
    return ModelIdentity(id="m", name="m", context_limit=131072)


def _opt():
    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    opt.client = MagicMock()
    from lm_optimizer.domain.models import LMStudioCapabilities

    opt.client.capabilities = LMStudioCapabilities()
    opt.client.ensure_unloaded = AsyncMock(return_value=True)
    opt.benchmark = MagicMock()
    opt.benchmark.gpu_via_cli = False
    opt.benchmark.run_speed_probe = AsyncMock()
    opt.benchmark.run_cases = AsyncMock()
    opt.benchmark.style = "balanced"
    opt.quality = MagicMock()
    opt.search_generator = MagicMock()
    opt.state = OptimizationState(run=OptimizationRun(model=_model()), search_space=SearchSpace())
    opt.state.speed_context = 2048
    opt.state.quality_context = 8192
    opt._progress_cb = None
    opt._checkpoint = lambda reason="x": None  # type: ignore
    opt._broadcast = AsyncMock()  # type: ignore
    return opt


def _no_db(monkeypatch):
    from lm_optimizer.database import repositories as _repo

    monkeypatch.setattr(_repo.config_repo, "save", lambda c: str(c.id))
    monkeypatch.setattr(_repo.run_repo, "save", lambda r: str(r.id))


def _probe_result(speed: float, **cfg_kw) -> ConfigurationResult:
    cfg = LoadConfiguration(context_length=2048, **cfg_kw)
    m = BenchmarkMetrics(
        test_name="speed_probe", category="speed", success=True,
        completion_tokens=64, generation_tok_s=speed,
    )
    r = ConfigurationResult(
        id=uuid4(), config=cfg, context_length=2048,
        status=ConfigurationStatus.PASSED, metrics=[m],
    )
    r.generation = {"phase": "speed", "stage": "speed_discovery"}
    return r


def _quality_result(speed: float, **cfg_kw) -> ConfigurationResult:
    cfg = LoadConfiguration(context_length=8192, **cfg_kw)
    m = BenchmarkMetrics(
        test_name="short_instruction", category="instruction", success=True,
        completion_tokens=100, generation_tok_s=speed,
    )
    r = ConfigurationResult(
        id=uuid4(), config=cfg, context_length=8192,
        status=ConfigurationStatus.PASSED, metrics=[m],
    )
    r.quality_score = QualityScore(
        overall=0.99, task_completion=1.0, factual_consistency=1.0,
        format_compliance=1.0, coding_correctness=1.0, no_truncation=1.0,
        no_malformed=1.0, checks_passed=30, checks_total=30,
    )
    r.score = 0.8
    r.generation = {
        "phase": "quality", "stage": "quality_check", "score_context": False,
        "quality_completed": True,
    }
    return r


def _flag(opt, tmp_path):
    path = tmp_path / f"pause_{opt.state.run.id}.flag"
    path.write_text(json.dumps({"reason": "user_requested"}), encoding="utf-8")
    return path


def _patch_ckpt_dir(tmp_path, monkeypatch):
    from lm_optimizer.storage import run_checkpoint as rc

    monkeypatch.setattr(rc, "checkpoint_dir", lambda: tmp_path)


def _seed_db_run(monkeypatch):
    """Persisted run + model + hardware rows (satisfies FK constraints)."""
    from lm_optimizer.database import repositories as _repo
    from lm_optimizer.domain.models import HardwareInfo

    hw = HardwareInfo(
        os="windows", cpu_name="c", cpu_cores_physical=8, cpu_cores_logical=16,
        total_ram_gb=16.0, gpu_count=1,
    )
    _repo.hardware_repo.save(hw)
    _repo.model_repo.save(_model())
    run = OptimizationRun(model=_model(), hardware=hw)
    _repo.run_repo.save(run)
    return run


def _real_checkpoint(opt):
    """Rebind the real _checkpoint (file-writing) onto a mocked optimizer."""
    import types

    opt._checkpoint = types.MethodType(AdaptiveOptimizer._checkpoint, opt)
    return opt


class TestPausePhases:
    def test_pause_during_speed(self, tmp_path, monkeypatch):
        import pytest

        from lm_optimizer.services.optimizer import PauseRequested

        _no_db(monkeypatch)
        _patch_ckpt_dir(tmp_path, monkeypatch)
        opt = _opt()
        seeded = _probe_result(50.0)
        opt.state.tested_configs = [seeded]
        _flag(opt, tmp_path)  # pause requested before remaining probes
        opt.benchmark.run_speed_probe = AsyncMock(return_value=_probe_result(51.0))
        with pytest.raises(PauseRequested):
            asyncio.run(opt._phase_speed())
        assert opt.state.run.status.value == "paused"
        assert len(opt.state.tested_configs) == 1  # nothing new measured
        assert opt.benchmark.run_speed_probe.await_count == 0

    def test_pause_during_quality(self, tmp_path, monkeypatch):
        import pytest

        from lm_optimizer.services.optimizer import PauseRequested

        _no_db(monkeypatch)
        _patch_ckpt_dir(tmp_path, monkeypatch)
        opt = _opt()
        first = _quality_result(70.0, eval_batch_size=1)
        second = _quality_result(69.0, eval_batch_size=2)
        calls = []

        async def _tc(cfg, ctx, **kw):
            calls.append(1)
            if len(calls) == 1:
                _flag(opt, tmp_path)
            return first if cfg.eval_batch_size == 1 else second

        opt._test_config = AsyncMock(side_effect=_tc)  # type: ignore
        front = {"winner": first, "finalists": [first, second], "contrarian": None}
        with pytest.raises(PauseRequested):
            asyncio.run(opt._phase_quality(front))
        assert opt.state.run.status.value == "paused"
        assert opt._test_config.await_count == 1

    def test_pause_during_recovery(self, tmp_path, monkeypatch):
        import pytest

        from lm_optimizer.services.optimizer import PauseRequested

        _no_db(monkeypatch)
        _patch_ckpt_dir(tmp_path, monkeypatch)
        opt = _opt()
        _flag(opt, tmp_path)
        opt._run_recovery = AsyncMock()  # type: ignore — must never fire
        with pytest.raises(PauseRequested):
            asyncio.run(
                opt._recover_failed(
                    [_probe_result(10.0)], LoadConfiguration(context_length=8192)
                )
            )
        assert opt._run_recovery.await_count == 0
        assert opt.state.run.status.value == "paused"

    def test_pause_during_final_validation(self, tmp_path, monkeypatch):
        import pytest

        from lm_optimizer.services.optimizer import PauseRequested

        _no_db(monkeypatch)
        _patch_ckpt_dir(tmp_path, monkeypatch)
        opt = _opt()
        opt.state.best_config = _quality_result(70.0)
        _flag(opt, tmp_path)
        opt._test_config = AsyncMock()  # type: ignore
        with pytest.raises(PauseRequested):
            asyncio.run(opt._phase_final_validation())
        assert opt._test_config.await_count == 0
        assert opt.state.run.status.value == "paused"


class TestPauseDurability:
    def test_pause_persists_checkpoint(self, tmp_path, monkeypatch):
        import pytest

        from lm_optimizer.services.optimizer import PauseRequested
        from lm_optimizer.storage import run_checkpoint as rc

        _no_db(monkeypatch)
        _patch_ckpt_dir(tmp_path, monkeypatch)
        opt = _real_checkpoint(_opt())
        _flag(opt, tmp_path)
        with pytest.raises(PauseRequested):
            asyncio.run(opt._pause_point())
        ckpt = rc.load_checkpoint(opt.state.run.id)
        assert ckpt is not None
        assert "paused" in str(ckpt.get("reason", ""))
        pab = ckpt.get("phase_ab") or {}
        assert pab.get("pause", {}).get("reason") == "user_requested"
        assert "phase" in pab["pause"]

    def test_pause_does_not_repeat_completed_work(self, tmp_path, monkeypatch):
        _no_db(monkeypatch)
        _patch_ckpt_dir(tmp_path, monkeypatch)
        opt = _opt()
        first = _quality_result(70.0, eval_batch_size=1)
        second = _quality_result(69.0, eval_batch_size=2)
        calls = []

        async def _tc(cfg, ctx, **kw):
            calls.append(1)
            if len(calls) == 1:
                _flag(opt, tmp_path)
            return first if cfg.eval_batch_size == 1 else second

        opt._test_config = AsyncMock(side_effect=_tc)  # type: ignore
        front = {"winner": first, "finalists": [first, second], "contrarian": None}
        import pytest

        from lm_optimizer.services.optimizer import PauseRequested

        with pytest.raises(PauseRequested):
            asyncio.run(opt._phase_quality(front))
        assert opt._test_config.await_count == 1
        # Resume after clearing the flag: first finalist reused, only second runs.
        # (The pausing run already consumed the flag file; guard for safety.)
        (tmp_path / f"pause_{opt.state.run.id}.flag").unlink(missing_ok=True)
        opt._test_config.reset_mock()
        safe, _ = asyncio.run(opt._phase_quality(front))
        assert opt._test_config.await_count == 1
        assert len(safe) == 2

    def test_current_in_progress_not_falsely_complete(self, tmp_path, monkeypatch):
        import pytest

        from lm_optimizer.services.optimizer import PauseRequested

        _no_db(monkeypatch)
        _patch_ckpt_dir(tmp_path, monkeypatch)
        opt = _opt()
        _flag(opt, tmp_path)
        with pytest.raises(PauseRequested):
            asyncio.run(opt._phase_speed())
        assert opt.state.tested_configs == []
        assert opt._quality_result_exists(("x",), 8192) is None


class TestPauseResumeWiring:
    def _seed_run(self, monkeypatch, phase_ab=None):
        from lm_optimizer.database import repositories as _repo
        from lm_optimizer.domain.models import HardwareInfo

        hw = HardwareInfo(
            os="windows", cpu_name="c", cpu_cores_physical=8, cpu_cores_logical=16,
            total_ram_gb=16.0, gpu_count=1,
        )
        _repo.hardware_repo.save(hw)
        _repo.model_repo.save(_model())
        run = OptimizationRun(model=_model(), hardware=hw)
        if phase_ab is not None:
            run.benchmark_params = {"phase_ab": phase_ab}
        _repo.run_repo.save(run)
        return run

    def _write_checkpoint(self, run):
        from lm_optimizer.storage import run_checkpoint as rc

        seeder = _opt()
        seeder.state.run = run
        assert rc.save_checkpoint(seeder.state, reason="paused:user_requested") is not None

    def test_resume_from_paused_run(self, tmp_path, monkeypatch):
        from lm_optimizer.services.optimizer import AdaptiveOptimizer

        run = self._seed_run(monkeypatch, phase_ab={"phase": "speed"})
        _patch_ckpt_dir(tmp_path, monkeypatch)
        self._write_checkpoint(run)
        opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
        opt.client = MagicMock()
        opt.client.ensure_unloaded = AsyncMock(return_value=True)
        opt.client.capabilities = MagicMock()
        opt.benchmark = MagicMock()
        opt.benchmark.style = "balanced"
        opt._checkpoint = lambda reason="x": None  # type: ignore
        opt._broadcast = AsyncMock()  # type: ignore
        opt._resume_phase_pipeline = AsyncMock()  # type: ignore
        out = asyncio.run(opt.resume_from_checkpoint(run.id))
        assert str(out.id) == str(run.id)

    def test_pause_and_resume_same_run_id(self, tmp_path, monkeypatch):
        from lm_optimizer.services.optimizer import AdaptiveOptimizer

        run = self._seed_run(monkeypatch, phase_ab={"phase": "speed"})
        _patch_ckpt_dir(tmp_path, monkeypatch)
        self._write_checkpoint(run)
        (tmp_path / f"pause_{run.id}.flag").write_text("{}", encoding="utf-8")
        opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
        opt.client = MagicMock()
        opt.client.ensure_unloaded = AsyncMock(return_value=True)
        opt.client.capabilities = MagicMock()
        opt.benchmark = MagicMock()
        opt.benchmark.style = "balanced"
        opt._checkpoint = lambda reason="x": None  # type: ignore
        opt._broadcast = AsyncMock()  # type: ignore
        opt._resume_phase_pipeline = AsyncMock()  # type: ignore
        out = asyncio.run(opt.resume_from_checkpoint(run.id))
        assert str(out.id) == str(run.id)

    def test_stale_checkpoint_cannot_destroy_paused_progress(self, monkeypatch):
        opt = _opt()
        db_pab = {
            "recovery_log": [{"event": "reconfirm"}],
            "recovery_cursor": {"step": 1},
            "pause": {"reason": "user_requested", "phase": "quality"},
        }
        ckpt_pab = {"recovery_log": [], "recovery_cursor": {"step": 0}}
        merged = opt._reconcile_recovery_state(ckpt_pab, db_pab)
        assert len(merged["recovery_log"]) == 1


class TestInterrupt:
    def test_ctrl_c_requests_pause_first(self, monkeypatch):
        opt = _opt()
        handler = opt._sigint_pause_handler()
        handler(2, None)  # first SIGINT: arm pause, do not raise
        assert opt.state.should_pause is True
        import pytest

        with pytest.raises(KeyboardInterrupt):
            handler(2, None)  # second SIGINT: fall through to abort

    def test_pause_cli_writes_flag(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from lm_optimizer.cli.main import app
        from lm_optimizer.storage import run_checkpoint as rc

        monkeypatch.setattr(rc, "checkpoint_dir", lambda: tmp_path)
        rid = str(uuid4())
        seeder = _opt()
        seeder.state.run.id = rid
        from uuid import UUID

        seeder.state.run.id = UUID(rid)
        assert rc.save_checkpoint(seeder.state, reason="test") is not None
        result = CliRunner().invoke(app, ["pause", rid])
        assert result.exit_code == 0
        assert (tmp_path / f"pause_{rid}.flag").exists()

    def test_run_status_cli_shows_paused(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from lm_optimizer.cli.main import app
        from lm_optimizer.storage import run_checkpoint as rc

        monkeypatch.setattr(rc, "checkpoint_dir", lambda: tmp_path)
        from lm_optimizer.domain.models import RunStatus

        run = _seed_db_run(monkeypatch)
        run.status = RunStatus.PAUSED
        run.benchmark_params = {
            "phase_ab": {"phase": "quality", "pause": {"reason": "user_requested"}}
        }
        from lm_optimizer.database import repositories as _repo

        _repo.run_repo.save(run)
        seeder = _opt()
        seeder.state.run = run
        assert rc.save_checkpoint(seeder.state, reason="test") is not None
        (tmp_path / f"pause_{run.id}.flag").write_text("{}", encoding="utf-8")
        result = CliRunner().invoke(app, ["run-status", str(run.id)])
        assert result.exit_code == 0
        assert "PAUSED" in result.output or "paused" in result.output

    def test_api_exposes_pause_block(self):
        from lm_optimizer.api.routes import _convert_run

        run = OptimizationRun(model=_model())
        run.benchmark_params = {"phase_ab": {"pause": {"reason": "user_requested"}}}
        resp = _convert_run(run)
        assert resp.pause == {"reason": "user_requested"}
