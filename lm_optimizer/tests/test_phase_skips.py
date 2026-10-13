"""Phase skips + full-pipeline completion (mission: complete auto tests).

Covers:
- parse_phase_skips (valid names, case-insensitive, list/str, unknown rejects)
- _run_phase_ab honors skip_phases (skipped phases never execute)
- _resume_phase_pipeline honors skips stored on the run
- pick_quality_rungs (top-N stable contexts, descending)
- CLI flags registered (--skip-phases, --physical-batches, resume-menu --all, ctx --quality)
- API AdvancedSettingsSchema accepts skip_phases
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from lm_optimizer.domain.models import (
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
    QualityScore,
)
from lm_optimizer.services.optimizer import (
    PHASE_SKIP_NAMES,
    AdaptiveOptimizer,
    OptimizationState,
    parse_phase_skips,
)
from lm_optimizer.services.search_space import SearchSpace


def _model():
    return ModelIdentity(id="m", name="m", context_limit=131072)


def _run(**kw):
    return OptimizationRun(model=_model(), profile=OptimizationProfile.BALANCED, **kw)


def _opt():
    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    opt.client = MagicMock()
    from lm_optimizer.domain.models import LMStudioCapabilities

    opt.client.capabilities = LMStudioCapabilities()
    opt.benchmark = MagicMock()
    opt.benchmark.gpu_via_cli = False
    opt.benchmark.run_speed_probe = AsyncMock()
    opt.benchmark.run_cases = AsyncMock()
    opt.quality = MagicMock()
    opt.search_generator = MagicMock()
    opt.state = OptimizationState(run=_run(), search_space=SearchSpace())
    opt.state.speed_context = 2048
    opt.state.quality_context = 8192
    opt.state.phase = "speed"
    opt._progress_cb = None
    opt._checkpoint = lambda reason="x": None  # type: ignore
    opt._broadcast = AsyncMock()  # type: ignore
    return opt


class TestParsePhaseSkips:
    def test_names(self):
        assert set(PHASE_SKIP_NAMES) == {
            "speed", "quality", "recovery", "validation", "context",
        }

    def test_comma_string(self):
        assert parse_phase_skips("speed, quality") == {"speed", "quality"}

    def test_list_and_case(self):
        assert parse_phase_skips(["Speed", "CONTEXT"]) == {"speed", "context"}

    def test_none_and_empty(self):
        assert parse_phase_skips(None) == set()
        assert parse_phase_skips([]) == set()
        assert parse_phase_skips("") == set()

    def test_unknown_rejects(self):
        with pytest.raises(ValueError):
            parse_phase_skips("speed,bogus")


class TestRunPhaseAbSkips:
    def _wire(self, opt, monkeypatch):
        import lm_optimizer.services.optimizer as _optmod

        monkeypatch.setattr(_optmod, "run_repo", MagicMock())
        opt._phase_speed = AsyncMock(return_value={"winner": None, "finalists": [], "contrarian": None})
        opt._phase_quality = AsyncMock(return_value=([], []))
        opt._recover_failed = AsyncMock(return_value=([], None))
        opt._phase_final_validation = AsyncMock()
        opt._phase_context_optional = AsyncMock()
        opt._capture_baseline_config = AsyncMock(return_value=None)
        opt._pause_point = AsyncMock()
        opt._recompute_all_scores = MagicMock()
        opt._finalize = AsyncMock()
        opt._sync_phase_params = MagicMock()
        opt._recovery_reference = MagicMock(return_value=(None, "none"))

    def test_no_skips_runs_everything(self, monkeypatch):
        opt = _opt()
        self._wire(opt, monkeypatch)
        asyncio.run(opt._run_phase_ab({}, None))
        opt._phase_speed.assert_awaited_once()
        opt._phase_quality.assert_awaited_once()
        opt._recover_failed.assert_awaited_once()
        opt._phase_final_validation.assert_awaited_once()

    def test_skip_speed_and_quality(self, monkeypatch):
        opt = _opt()
        self._wire(opt, monkeypatch)
        asyncio.run(opt._run_phase_ab({"skip_phases": ["speed", "quality"]}, None))
        opt._phase_speed.assert_not_awaited()
        opt._phase_quality.assert_not_awaited()
        # Downstream still runs and finalizes (no winner path).
        opt._recover_failed.assert_awaited_once()
        opt._phase_final_validation.assert_awaited_once()
        opt._finalize.assert_awaited_once()

    def test_skip_recovery_and_validation(self, monkeypatch):
        opt = _opt()
        self._wire(opt, monkeypatch)
        asyncio.run(opt._run_phase_ab({"skip_phases": "recovery,validation"}, None))
        opt._phase_speed.assert_awaited_once()
        opt._phase_quality.assert_awaited_once()
        opt._recover_failed.assert_not_awaited()
        opt._phase_final_validation.assert_not_awaited()
        opt._finalize.assert_awaited_once()

    def test_skip_context_forces_phase_b_off(self, monkeypatch):
        opt = _opt()
        self._wire(opt, monkeypatch)
        opt.state.best_config = MagicMock()
        asyncio.run(
            opt._run_phase_ab({"skip_phases": ["context"], "optimize_context": True}, None)
        )
        assert opt.state.phase_b_enabled is False
        opt._phase_context_optional.assert_not_awaited()

    def test_unknown_skip_warns_not_crashes(self, monkeypatch):
        opt = _opt()
        self._wire(opt, monkeypatch)
        asyncio.run(opt._run_phase_ab({"skip_phases": ["nope"]}, None))
        opt._phase_speed.assert_awaited_once()
        opt._finalize.assert_awaited_once()


class TestResumePhasePipelineSkips:
    def test_resume_honors_stored_skips(self):
        opt = _opt()
        opt.state.run.advanced_settings = {"skip_phases": ["recovery", "validation"]}
        opt._phase_speed = AsyncMock(return_value={"winner": None, "finalists": [], "contrarian": None})
        opt._phase_quality = AsyncMock(return_value=([], []))
        opt._resume_recover = AsyncMock()
        opt._phase_final_validation = AsyncMock()
        opt._phase_context_optional = AsyncMock()
        opt._pause_point = AsyncMock()
        opt._recompute_all_scores = MagicMock()
        opt._finalize = AsyncMock()
        asyncio.run(opt._resume_phase_pipeline())
        opt._phase_speed.assert_awaited_once()
        opt._resume_recover.assert_not_awaited()
        opt._phase_final_validation.assert_not_awaited()
        opt._finalize.assert_awaited_once()


class TestPickQualityRungs:
    def _report(self):
        return {
            "geometric": [
                {"ctx": 8192, "ok": True, "tok_s": 50.0},
                {"ctx": 16384, "ok": True, "tok_s": 40.0},
                {"ctx": 32768, "ok": False, "tok_s": 0.0, "error": "OOM"},
            ],
            "refinement": [
                {"ctx": 65536, "ok": True, "tok_s": 20.0},
                {"ctx": 131072, "ok": False, "tok_s": 0.0, "error": "OOM"},
            ],
        }

    def test_top_n_stable_descending(self):
        from lm_optimizer.services.context_sweep import pick_quality_rungs

        assert pick_quality_rungs(self._report(), 2) == [65536, 16384]
        assert pick_quality_rungs(self._report(), 10) == [65536, 16384, 8192]
        assert pick_quality_rungs({"geometric": [], "refinement": []}, 3) == []
        assert pick_quality_rungs(self._report(), 0) == []

    def test_estimated_only_never_counts(self):
        from lm_optimizer.services.context_sweep import pick_quality_rungs

        report = {"geometric": [{"ctx": 32768, "ok": False, "estimated": True}], "refinement": []}
        assert pick_quality_rungs(report, 3) == []


class TestCliFlagsRegistered:
    def _params(self, cmd):
        import typer

        from lm_optimizer.cli.main import app

        info = typer.main.get_command(app)
        return [p.name for p in info.commands[cmd].params]

    def test_optimize_flags(self):
        params = self._params("optimize")
        assert "skip_phases" in params
        assert "physical_batches" in params
        assert "warmup_runs" in params
        assert "report_sort" in params

    def test_auto_flags(self):
        params = self._params("auto")
        assert "skip_phases" in params
        assert "physical_batches" in params
        assert "warmup_runs" in params
        assert "report_sort" in params

    def test_resume_flags(self):
        assert "skip_phases" in self._params("resume")

    def test_resume_menu_all(self):
        assert "all" in self._params("resume-menu")

    def test_ctx_quality(self):
        assert "quality_rungs" in self._params("ctx")
        assert "quality_threshold" in self._params("ctx")
        assert "warmup_runs" in self._params("ctx")


class TestWarmupConfig:
    def test_default_one(self):
        from lm_optimizer.cli.main import _benchmark_config_obj

        assert _benchmark_config_obj(3).warmup_repetitions == 1

    def test_zero_and_clamp(self):
        from lm_optimizer.cli.main import _benchmark_config_obj

        assert _benchmark_config_obj(3, 1.0, 0).warmup_repetitions == 0
        assert _benchmark_config_obj(3, 1.0, 99).warmup_repetitions == 5
        assert _benchmark_config_obj(3, 1.0, "bogus").warmup_repetitions == 1

    def test_api_schema(self):
        from lm_optimizer.api.schemas import AdvancedSettingsSchema

        assert AdvancedSettingsSchema().warmup_runs is None
        assert AdvancedSettingsSchema(warmup_runs=0).warmup_runs == 0


class TestApiSchema:
    def test_skip_phases_accepted(self):
        from lm_optimizer.api.schemas import AdvancedSettingsSchema

        s = AdvancedSettingsSchema(skip_phases=["speed", "context"])
        assert s.skip_phases == ["speed", "context"]
        assert AdvancedSettingsSchema().skip_phases == []


class TestResumeAll:
    def _runs(self):
        from types import SimpleNamespace

        def _r(rid, mid):
            return SimpleNamespace(
                id=rid, model=SimpleNamespace(id=mid, name=mid)
            )

        return [_r("id-1", "m1"), _r("id-2", "m2")]

    def test_empty_needs_no_confirm(self):
        import lm_optimizer.cli.main as _main

        assert _main.resume_all_runs([]) == {"resumed": [], "failed": {}}

    def test_continues_on_failure(self, monkeypatch):
        import lm_optimizer.cli.main as _main

        calls = []

        def _fake(rid, revalidate=False, style="balanced"):
            calls.append(rid)
            if rid == "id-2":
                raise SystemExit(1)

        monkeypatch.setattr(_main, "_run_resume", _fake)
        monkeypatch.setattr(
            _main.typer, "confirm", lambda *a, **k: True
        )
        out = _main.resume_all_runs(self._runs())
        assert out["resumed"] == ["id-1"]
        assert list(out["failed"]) == ["id-2"]
        assert calls == ["id-1", "id-2"]

    def test_cancel_runs_nothing(self, monkeypatch):
        import lm_optimizer.cli.main as _main

        monkeypatch.setattr(
            _main, "_run_resume",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")),
        )
        monkeypatch.setattr(_main.typer, "confirm", lambda *a, **k: False)
        out = _main.resume_all_runs(self._runs())
        assert out == {"resumed": [], "failed": {}}


class TestApplyResumeSkips:
    def _seed(self):
        from lm_optimizer.database import repositories as _repo
        from lm_optimizer.domain.models import HardwareInfo

        hw = HardwareInfo(
            os="windows", cpu_name="c", cpu_cores_physical=8,
            cpu_cores_logical=16, total_ram_gb=16.0, gpu_count=1,
        )
        _repo.hardware_repo.save(hw)
        _repo.model_repo.save(_model())
        run = OptimizationRun(model=_model(), hardware=hw)
        _repo.run_repo.save(run)
        return run

    def test_merges_into_stored_settings(self):
        from lm_optimizer.cli.main import _apply_resume_skips
        from lm_optimizer.database import repositories as _repo

        run = self._seed()
        _apply_resume_skips(run, "speed,context")
        # In-memory copy + persisted copy (advanced_settings is not a DB column).
        assert run.advanced_settings["skip_phases"] == ["context", "speed"]
        again = _repo.run_repo.get(str(run.id))
        assert again.benchmark_params["skip_phases"] == ["context", "speed"]

    def test_run_skip_phases_db_fallback(self):
        from lm_optimizer.services.optimizer import run_skip_phases

        run = self._seed()
        run.advanced_settings = {}
        run.benchmark_params = {"skip_phases": ["recovery"]}
        assert run_skip_phases(run) == {"recovery"}
        run.benchmark_params = {"skip_phases": ["bogus"]}
        assert run_skip_phases(run) == set()

    def test_noop_when_empty(self):
        from lm_optimizer.cli.main import _apply_resume_skips
        from lm_optimizer.database import repositories as _repo

        run = self._seed()
        _apply_resume_skips(run, None)
        again = _repo.run_repo.get(str(run.id))
        assert (again.advanced_settings or {}).get("skip_phases") is None

    def test_invalid_exits_2(self):
        import pytest

        from lm_optimizer.cli.main import _apply_resume_skips

        run = self._seed()
        with pytest.raises(SystemExit) as e:
            _apply_resume_skips(run, "bogus")
        assert e.value.code == 2
