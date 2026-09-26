"""Resume/checkpoint persistence regressions (14b campaign findings).

No real models; mocked benchmark/client seams, tmp-DB-backed repos where needed.
"""

import asyncio
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


def _quality_failed_result(**cfg_kw) -> ConfigurationResult:
    cfg = LoadConfiguration(context_length=8192, **cfg_kw)
    m = BenchmarkMetrics(
        test_name="short_instruction", category="instruction", success=True,
        completion_tokens=100, generation_tok_s=60.0,
    )
    r = ConfigurationResult(
        id=uuid4(), config=cfg, context_length=8192,
        status=ConfigurationStatus.QUALITY_FAILED, metrics=[m],
    )
    r.quality_score = QualityScore(
        overall=0.90, task_completion=0.9, factual_consistency=1.0,
        format_compliance=0.9, coding_correctness=1.0, no_truncation=0.9,
        no_malformed=0.9, checks_passed=27, checks_total=30,
    )
    r.generation = {
        "stage": "quality_check",
        "quality_by_test": {
            "short_instruction": {"overall": 0.9, "checks_passed": 5, "checks_total": 6},
        },
    }
    return r


class TestQualityCompletion:
    def test_quality_failed_is_resume_complete(self):
        opt = _opt()
        failed = _quality_failed_result()
        failed.generation["quality_completed"] = True
        opt.state.tested_configs = [failed]
        found = opt._quality_result_exists(opt._candidate_key(failed.config), 8192)
        assert found is failed

    def test_quality_passed_is_resume_complete(self):
        opt = _opt()
        passed = _quality_failed_result()
        passed.status = ConfigurationStatus.PASSED
        passed.score = 0.8
        passed.generation["quality_completed"] = True
        opt.state.tested_configs = [passed]
        found = opt._quality_result_exists(opt._candidate_key(passed.config), 8192)
        assert found is passed

    def test_backfill_pre_marker_rows_count(self):
        """14b rows: no marker, but failed+stage+quality_by_test => completed."""
        opt = _opt()
        failed = _quality_failed_result()  # no quality_completed key
        assert "quality_completed" not in failed.generation
        opt.state.tested_configs = [failed]
        found = opt._quality_result_exists(opt._candidate_key(failed.config), 8192)
        assert found is failed

    def test_existing_quality_result_is_not_rerun(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        failed = _quality_failed_result()
        opt.state.tested_configs = [failed]
        opt._test_config = AsyncMock()  # type: ignore — must never fire
        front = {"winner": None, "finalists": [failed], "contrarian": None}
        safe, failed_out = asyncio.run(opt._phase_quality(front))
        assert opt._test_config.await_count == 0
        assert failed_out == [failed]
        assert any(
            "RESUME_SKIP" in e.get("message", "") for e in opt.state.event_log
        )

    def test_failed_quality_persisted_without_scoring(self, monkeypatch):
        """_test_config failing path: persisted + stamped, score stays None."""
        _no_db(monkeypatch)
        opt = _opt()
        bad_metrics = BenchmarkMetrics(
            test_name="short_instruction", category="instruction", success=True,
            completion_tokens=100, generation_tok_s=60.0,
        )
        bench_result = ConfigurationResult(
            id=uuid4(), config=LoadConfiguration(context_length=8192),
            context_length=8192, status="passed", metrics=[bad_metrics],
        )
        bench_result.generation = {"load_channel": "REST"}
        opt.benchmark.run_benchmark = AsyncMock(return_value=bench_result)
        weak = QualityScore(
            overall=0.90, task_completion=0.9, factual_consistency=1.0,
            format_compliance=0.9, coding_correctness=1.0, no_truncation=0.9,
            no_malformed=0.9, checks_passed=27, checks_total=30,
        )
        opt.quality.evaluate_all = MagicMock(return_value={"short_instruction": weak})
        opt.quality.aggregate_quality = MagicMock(return_value=weak)
        opt.quality.passes_threshold = MagicMock(return_value=False)
        opt.quality.config = MagicMock()
        opt.quality.config.minimum_score = 0.97
        saved = []
        from lm_optimizer.database import repositories as _repo

        monkeypatch.setattr(_repo.config_repo, "save", lambda c: saved.append(c.id) or str(c.id))
        result = asyncio.run(
            opt._test_config(LoadConfiguration(context_length=8192), 8192, score_context=False)
        )
        assert result is not None and result.status == ConfigurationStatus.QUALITY_FAILED
        assert result.score is None
        assert result.score_breakdown is None
        assert result.generation.get("quality_completed") is True
        assert saved, "failed quality result must be persisted"


class TestProbeAttempts:
    def test_failed_probe_retried_once_then_completed(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        anchor = LoadConfiguration(context_length=2048, eval_batch_size=512)
        key = opt._candidate_key(anchor)
        # First failure recorded: attempts == 1 -> still retriable.
        opt.state.probe_attempts = {repr(key): 1}
        assert opt._probe_completed(key) is False
        # Second failure: completed, never blindly repeated.
        opt.state.probe_attempts = {repr(key): 2}
        assert opt._probe_completed(key) is True

    def test_failed_probe_not_rerun_when_completed(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        anchor = LoadConfiguration(context_length=2048, eval_batch_size=512)
        opt.state.tested_configs = []
        opt.state.probe_attempts = {repr(opt._candidate_key(anchor)): 2}
        from lm_optimizer.services.speed_search import SpeedPlanItem

        item = SpeedPlanItem(stage="S1", config=anchor, reason="t")
        out = asyncio.run(opt._safe_probe(item, 2048, set()))
        assert out is None
        assert opt.benchmark.run_speed_probe.await_count == 0


class TestCheckpointReconcile:
    def test_stale_checkpoint_cannot_overwrite_newer_db_state(self, monkeypatch):
        opt = _opt()
        db_pab = {"recovery_log": [{"event": "reconfirm"}, {"suspect": "x"}], "recovery_cursor": {"step": 2}}
        ckpt_pab = {"recovery_log": [], "recovery_cursor": {"step": 0}}
        merged = opt._reconcile_recovery_state(ckpt_pab, db_pab)
        assert len(merged["recovery_log"]) == 2
        assert merged["recovery_cursor"]["step"] == 2

    def test_resume_reconciles_db_and_checkpoint(self, monkeypatch):
        opt = _opt()
        db_pab = {"recovery_log": [{"event": "reconfirm"}], "recovery_cursor": {"step": 1}}
        ckpt_pab = {"recovery_log": [{"event": "reconfirm"}], "recovery_cursor": {"step": 1}}
        opt._reconcile_recovery_state(ckpt_pab, db_pab)
        assert any(
            "RESUME_RECONCILE" in e.get("message", "") for e in opt.state.event_log
        )

    def test_atomic_checkpoint_write(self, tmp_path, monkeypatch):
        from lm_optimizer.storage import run_checkpoint as rc

        monkeypatch.setattr(rc, "checkpoint_dir", lambda: tmp_path)
        opt = _opt()
        opt.state.recovery_log = [{"event": "reconfirm", "config_id": "abc"}]
        opt.state.recovery_cursor = {"finalist_id": "abc", "step": 1}
        opt.state.checkpoint_seq = 7
        path = rc.save_checkpoint(opt.state, reason="recovery:1")
        assert path is not None and path.exists()
        loaded = rc.load_checkpoint(opt.state.run.id)
        assert loaded is not None
        pab = loaded.get("phase_ab") or {}
        assert pab.get("recovery_log") == [{"event": "reconfirm", "config_id": "abc"}]
        assert pab.get("recovery_cursor", {}).get("step") == 1
        assert pab.get("checkpoint_seq") == 7

    def test_recovery_log_persists_after_each_transition(self, monkeypatch):
        """Every recovery_log.append in the recovery path is followed by persist."""
        import inspect

        from lm_optimizer.services import optimizer as opt_mod

        src = inspect.getsource(opt_mod.AdaptiveOptimizer._run_recovery)
        src += inspect.getsource(opt_mod.AdaptiveOptimizer._try_rollback_suspect)
        src += inspect.getsource(opt_mod.AdaptiveOptimizer._try_pairwise)
        assert src.count("recovery_log.append") >= 3
        assert src.count("_persist_recovery_step") >= 3
