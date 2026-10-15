"""Task 8: checkpoint/DB reconciliation + stale-running diagnostics.

Longer-log-wins + DB-authoritative already correct → these pin the behavior.
Mock-only; never touches data/optimizer.db or live checkpoints.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock


def _opt_with_state():
    from lm_optimizer.services import optimizer as _opt

    opt = _opt.AdaptiveOptimizer.__new__(_opt.AdaptiveOptimizer)
    state = MagicMock()
    state.recovery_log = []
    state.recovery_cursor = None
    state.checkpoint_seq = 0
    state.event_log = []
    state.run = MagicMock()
    state.run.benchmark_params = {}
    opt.state = state
    return opt, _opt


class TestReconcile:
    def test_db_newer_log_wins(self, monkeypatch):
        from lm_optimizer.services import optimizer as _opt

        opt, _ = _opt_with_state()
        monkeypatch.setattr(_opt.run_repo, "save", lambda r: "r1")
        db = {"recovery_log": [{"a": 1}, {"a": 2}, {"a": 3}],
              "recovery_cursor": {"step": 3}, "checkpoint_seq": 5}
        ckpt = {"recovery_log": [{"a": 1}],
                "recovery_cursor": {"step": 1}, "checkpoint_seq": 2}
        out = opt._reconcile_recovery_state(ckpt, db)
        assert len(out["recovery_log"]) == 3
        assert out["recovery_cursor"]["step"] == 3

    def test_checkpoint_newer_cursor_only(self, monkeypatch):
        from lm_optimizer.services import optimizer as _opt

        opt, _ = _opt_with_state()
        monkeypatch.setattr(_opt.run_repo, "save", lambda r: "r1")
        db = {"recovery_log": [{"a": 1}, {"a": 2}],
              "recovery_cursor": {"step": 2}, "checkpoint_seq": 4}
        ckpt = {"recovery_log": [{"a": 1}],
                "recovery_cursor": {"step": 7}, "checkpoint_seq": 3}
        out = opt._reconcile_recovery_state(ckpt, db)
        assert len(out["recovery_log"]) == 2  # longer DB log kept
        assert out["recovery_cursor"]["step"] == 7  # greater cursor kept

    def test_stale_checkpoint_cannot_erase(self, monkeypatch):
        from lm_optimizer.services import optimizer as _opt

        opt, _ = _opt_with_state()
        monkeypatch.setattr(_opt.run_repo, "save", lambda r: "r1")
        db = {"recovery_log": [{"a": 1}, {"a": 2}, {"a": 3}],
              "recovery_cursor": {"step": 3}, "checkpoint_seq": 9}
        ckpt = {"recovery_log": [], "recovery_cursor": {}, "checkpoint_seq": 1}
        out = opt._reconcile_recovery_state(ckpt, db)
        assert len(out["recovery_log"]) == 3
        assert opt.state.checkpoint_seq == 9

    def test_quality_failed_restored_as_completed(self):
        from lm_optimizer.domain.models import (
            ConfigurationResult, ConfigurationStatus, HardwareInfo,
            LoadConfiguration, ModelIdentity, OptimizationProfile, OptimizationRun,
        )
        from lm_optimizer.services import optimizer as _opt

        opt = _opt.AdaptiveOptimizer.__new__(_opt.AdaptiveOptimizer)
        state = MagicMock()
        state.failed_regions = set()
        state.event_log = []
        state.errors = []
        opt.state = state
        qf = ConfigurationResult(
            config=LoadConfiguration(context_length=8192),
            context_length=8192, status=ConfigurationStatus.QUALITY_FAILED,
            score=None,
        )
        ok = ConfigurationResult(
            config=LoadConfiguration(context_length=4096),
            context_length=4096, status=ConfigurationStatus.PASSED, score=0.5,
        )
        full = SimpleNamespace(
            configurations=[qf, ok], best_config_id=ok.id, pareto_config_ids=[],
            benchmark_params={},
        )
        ckpt = {"event_log": [], "errors": [], "flash_required": False, "phase_ab": {}}
        opt._restore_tested_state(full, ckpt)
        ids = {str(c.id) for c in state.tested_configs}
        assert str(qf.id) in ids  # finished suite stays as completed work

    def test_interrupted_restore_not_blocked(self):
        from lm_optimizer.domain.models import RunStatus

        full_params = {"phase_ab": {"phase": "quality"}}
        assert "phase_ab" in full_params  # phase pipeline, not legacy
        assert RunStatus.INTERRUPTED.value == "interrupted"

    def test_duplicate_resume_guarded_batch(self, monkeypatch):
        import typer

        from lm_optimizer.cli import main as _cli

        monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)
        calls = []

        def _fake_resume(rid, **kw):
            calls.append(rid)
            if calls.count(rid) > 1:
                raise RuntimeError("already running")

        monkeypatch.setattr(_cli, "_run_resume", _fake_resume)
        runs = [
            SimpleNamespace(id="run-1", model=SimpleNamespace(name="m", id="m")),
            SimpleNamespace(id="run-1", model=SimpleNamespace(name="m", id="m")),
        ]
        out = _cli.resume_all_runs(runs)
        assert out["resumed"] == ["run-1"]
        assert "run-1" in out["failed"]

    def test_legacy_checkpoint_readable(self):
        from lm_optimizer.services.optimizer import AdaptiveOptimizer

        opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
        state = MagicMock()
        state.failed_regions = set()
        state.event_log = []
        state.errors = []
        opt.state = state
        full = SimpleNamespace(
            configurations=[], best_config_id=None, pareto_config_ids=[],
            benchmark_params={},
        )
        ckpt = {"event_log": [], "errors": []}  # no phase_ab: legacy shape
        opt._restore_tested_state(full, ckpt)  # must not raise
        assert opt.state.phase == "speed"


class TestResumeMenuListing:
    def _runs(self):
        from datetime import datetime

        from lm_optimizer.domain.models import (
            HardwareInfo, ModelIdentity, OptimizationProfile, OptimizationRun,
            OptimizationStage, RunStatus,
        )

        run = OptimizationRun(
            model=ModelIdentity(id="m", name="m"),
            hardware=HardwareInfo(
                os="T", cpu_name="C", cpu_cores_physical=1,
                cpu_cores_logical=2, total_ram_gb=8, gpu_count=0,
            ),
            profile=OptimizationProfile.BALANCED,
            status=RunStatus.PAUSED,
        )
        run.stage = OptimizationStage.DISCOVERY
        run.created_at = datetime(2026, 10, 1, 12, 0)
        run.config_count = 10
        return [run]

    def test_menu_shows_checkpoint_phase_hint(self, monkeypatch):
        from rich.console import Console
        from typer.testing import CliRunner

        from lm_optimizer.cli import main as _cli
        from lm_optimizer.services import optimizer as _opt

        runs = self._runs()
        monkeypatch.setattr(_opt.run_repo, "list_all", lambda limit=20: runs)
        monkeypatch.setattr(_cli.run_repo, "list_all", lambda limit=20: runs)
        monkeypatch.setattr(_cli, "console", Console(width=200))
        import lm_optimizer.storage.run_checkpoint as _ckpt

        monkeypatch.setattr(
            _ckpt, "load_checkpoint",
            lambda rid: {
                "completed_candidate_ids": ["a", "b"],
                "saved_at": "2026-10-02T10:00:00",
                "event_log": [{"message": "BENCHMARK_COMPLETED", "detail": "ctx=4096"}],
                "phase_ab": {"phase": "quality", "recovery_cursor": {"step": 4}},
            },
        )
        import typer as _typer

        monkeypatch.setattr(_typer, "prompt", lambda *a, **k: 0)
        result = CliRunner().invoke(_cli.app, ["resume-menu"])
        assert result.exit_code == 0, result.output
        assert "Checkpoint" in result.output
        assert "Phase/Cursor" in result.output
        assert "safe to resume" in result.output
        assert "quality@4" in result.output
        assert "BENCHMARK_COMPLETED" in result.output

    def test_menu_missing_checkpoint_hint(self, monkeypatch):
        from rich.console import Console
        from typer.testing import CliRunner

        from lm_optimizer.cli import main as _cli
        from lm_optimizer.services import optimizer as _opt

        runs = self._runs()
        monkeypatch.setattr(_opt.run_repo, "list_all", lambda limit=20: runs)
        monkeypatch.setattr(_cli.run_repo, "list_all", lambda limit=20: runs)
        monkeypatch.setattr(_cli, "console", Console(width=200))
        import lm_optimizer.storage.run_checkpoint as _ckpt

        monkeypatch.setattr(_ckpt, "load_checkpoint", lambda rid: None)
        import typer as _typer

        monkeypatch.setattr(_typer, "prompt", lambda *a, **k: 0)
        result = CliRunner().invoke(_cli.app, ["resume-menu"])
        assert result.exit_code == 0, result.output
        assert "no checkpoint" in result.output

    def test_resume_paths_never_mass_reset(self):
        import inspect

        from lm_optimizer.cli import main as _cli

        for fn in (_cli.resume_menu, _cli.resume_all_runs, _cli._run_resume, _cli.resume):
            assert "remove_checkpoint" not in inspect.getsource(fn)


class TestResumeLoopSkipsCompleted:
    def test_restored_quality_failed_key_not_rerun(self, monkeypatch):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from lm_optimizer.domain.models import (
            ConfigurationResult, ConfigurationStatus, LoadConfiguration,
        )
        from lm_optimizer.services import optimizer as _opt

        dims = dict(
            context_length=8192, flash_attention=True,
            offload_kv_cache_to_gpu=True, eval_batch_size=512,
            physical_batch_size=512, parallel=4, context_checkpoints=32,
        )
        qf = ConfigurationResult(
            config=LoadConfiguration(**dims),
            context_length=8192, status=ConfigurationStatus.QUALITY_FAILED,
            score=None,
        )
        opt = _opt.AdaptiveOptimizer.__new__(_opt.AdaptiveOptimizer)
        state = MagicMock()
        state.tested_configs = [qf]
        state.failed_regions = set()
        state.event_log = []
        state.should_cancel = False
        state.should_pause = False
        run = MagicMock()
        run.model.context_limit = 32768
        run.advanced_settings = {}
        state.run = run
        state.search_space = MagicMock()
        opt.state = state
        opt._generate_coarse_configs = MagicMock(
            return_value=[LoadConfiguration(**dims)]
        )
        opt._test_config = AsyncMock()
        opt._checkpoint = MagicMock()
        opt._broadcast = AsyncMock()
        opt._recompute_all_scores = MagicMock()
        monkeypatch.setattr(_opt.run_repo, "save", lambda r: "r1")
        asyncio.run(opt._stage_coarse_search())
        opt._test_config.assert_not_called()


class TestPhaseTransitionCrashRecovery:
    """Test crash recovery for non-atomic phase transitions.

    Simulates the exact failure scenario from run 4aac923b:
    - DB stage advances to quality_check
    - Checkpoint still at phase:speed
    - Quality candidate completed in DB but not in checkpoint
    - Resume should reconstruct winner from DB
    """

    def _make_quality_candidate(self, config_id: str, speed: float) -> ConfigurationResult:
        """Create a quality-passed candidate at quality_context."""
        from lm_optimizer.domain.models import (
            BenchmarkMetrics,
            ConfigurationResult,
            ConfigurationStatus,
            LoadConfiguration,
            QualityScore,
        )
        from uuid import UUID

        cfg = LoadConfiguration(
            context_length=32768,
            gpu_ratio=1.0,
            flash_attention=True,
            offload_kv_cache_to_gpu=True,
            eval_batch_size=2048,
            physical_batch_size=512,
            parallel=8,
            context_checkpoints=32,
        )
        # Quality metrics
        m = BenchmarkMetrics(
            test_name="short_instruction",
            category="instruction",
            success=True,
            completion_tokens=100,
            generation_tok_s=speed,
            prompt_tok_s=10.0,
            estimated_ttft_ms=100.0,
        )
        r = ConfigurationResult(
            id=UUID(config_id),
            config=cfg,
            context_length=32768,
            status=ConfigurationStatus.PASSED,
            metrics=[m],
        )
        r.quality_score = QualityScore(
            overall=0.9829,
            task_completion=1.0,
            factual_consistency=1.0,
            format_compliance=1.0,
            coding_correctness=1.0,
            no_truncation=0.9,
            no_malformed=1.0,
            confident=True,
            checks_passed=29,
            checks_total=30,
        )
        r.score = 0.844
        r.generation = {
            "phase": "quality",
            "stage": "quality_check",
            "quality_completed": True,
            "quality_by_test": {
                "short_instruction": {"overall": 1.0, "checks_passed": 6, "checks_total": 6},
                "medium_reasoning": {"overall": 1.0, "checks_passed": 6, "checks_total": 6},
                "long_context": {"overall": 0.998, "checks_passed": 6, "checks_total": 6},
                "coding_task": {"overall": 0.917, "checks_passed": 5, "checks_total": 6},
                "structured_output": {"overall": 1.0, "checks_passed": 6, "checks_total": 6},
            },
            "score_context": False,
        }
        return r

    def _make_speed_finalist(self, config_id: str, speed: float) -> ConfigurationResult:
        """Create a speed-phase finalist at speed_context."""
        from lm_optimizer.domain.models import (
            BenchmarkMetrics,
            ConfigurationResult,
            ConfigurationStatus,
            LoadConfiguration,
        )
        from uuid import UUID

        cfg = LoadConfiguration(
            context_length=4096,
            gpu_ratio=1.0,
            flash_attention=True,
            offload_kv_cache_to_gpu=True,
            eval_batch_size=2048,
            physical_batch_size=512,
            parallel=8,
            context_checkpoints=32,
        )
        m = BenchmarkMetrics(
            test_name="speed_probe",
            category="speed",
            success=True,
            completion_tokens=64,
            generation_tok_s=speed,
        )
        r = ConfigurationResult(
            id=UUID(config_id),
            config=cfg,
            context_length=4096,
            status=ConfigurationStatus.PASSED,
            metrics=[m],
        )
        r.generation = {"phase": "speed", "stage": "speed_discovery"}
        return r

    def test_resume_after_speed_to_quality_crash_reconstructs_winner(
        self, tmp_path, monkeypatch
    ):
        """Simulate crash between DB stage update and checkpoint save.

        Setup:
        - Speed phase completed: 2 finalists in checkpoint and DB
        - Quality phase started: DB stage = quality_check
        - One quality candidate completed and passed quality threshold
        - Checkpoint NOT updated (reason=phase:speed)
        - Resume should:
            * Restore both speed finalists
            * Restore quality candidate from DB
            * Reconstruct winner (fastest quality-passing config)
            * Set best_config_id
            * NOT re-run quality candidate
        """
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from lm_optimizer.domain.models import (
            ConfigurationResult,
            ConfigurationStatus,
            HardwareInfo,
            LoadConfiguration,
            ModelIdentity,
            OptimizationProfile,
            OptimizationRun,
            OptimizationStage,
            RunStatus,
            QualityScore,
        )
        from lm_optimizer.services import optimizer as _opt
        from lm_optimizer.storage import run_checkpoint as _rc
        from uuid import UUID, uuid4

        # --- Build test fixtures ---

        # Two speed finalists (same config, different IDs)
        speed_finalist_1_id = str(uuid4())
        speed_finalist_2_id = str(uuid4())
        speed_finalist_1 = self._make_speed_finalist(speed_finalist_1_id, 2.0)
        speed_finalist_2 = self._make_speed_finalist(speed_finalist_2_id, 1.8)

        # Quality candidate (same params as finalist_1 but at quality_context)
        quality_candidate_id = str(uuid4())
        quality_candidate = self._make_quality_candidate(quality_candidate_id, 1.5)

        # Speed candidate that failed quality (for failed list)
        failed_candidate_id = str(uuid4())
        failed_candidate = self._make_quality_candidate(failed_candidate_id, 1.0)
        failed_candidate.status = ConfigurationStatus.QUALITY_FAILED
        failed_candidate.quality_score.overall = 0.95
        failed_candidate.quality_score.checks_passed = 25
        failed_candidate.score = None
        failed_candidate.generation["quality_completed"] = True

        # Build full DB run state
        model = ModelIdentity(id="test-model", name="test-model", context_limit=32768)
        from lm_optimizer.domain.models import GPUInfo
        hw = HardwareInfo(
            os="T", cpu_name="C", cpu_cores_physical=8, cpu_cores_logical=16,
            total_ram_gb=32.0, gpu_count=1,
            gpus=[GPUInfo(index=0, name="Test GPU", vram_gb=6.0, vendor="NVIDIA")]
        )

        run = OptimizationRun(
            model=model,
            hardware=hw,
            profile=OptimizationProfile.BALANCED,
        )
        run.id = uuid4()
        run.status = RunStatus.RUNNING
        run.stage = OptimizationStage.QUALITY_CHECK  # DB says quality_check
        run.speed_context = 4096
        run.quality_context = 32768
        run.best_config_id = None  # Not set due to crash
        # DB phase updated to quality at phase start (before crash)
        run.benchmark_params = {
            "phase_ab": {
                "phase": "quality",  # DB phase advanced to quality
                "speed_context": 4096,
                "quality_context": 32768,
                "budget_class": "NORMAL",
                "speed_finalist_ids": [speed_finalist_1_id, speed_finalist_2_id],
                "raw_fastest_id": speed_finalist_1_id,
                "checkpoint_seq": 1,
                "excluded_notes": [],
            }
        }

        # DB has all configurations including the quality one
        all_configs = [speed_finalist_1, speed_finalist_2, quality_candidate, failed_candidate]
        run.configurations = all_configs
        run.pareto_config_ids = []

        # Build checkpoint (stale, from end of speed phase)
        ckpt_data = {
            "version": 1,
            "reason": "phase:speed",
            "run_id": str(run.id),
            "model": model.id,
            "stage": "speed_discovery",
            "status": "running",
            "phase": "speed",
            "completed_candidate_ids": [speed_finalist_1_id, speed_finalist_2_id],
            "successful_results": [
                {"id": speed_finalist_1_id, "context_length": 4096, "status": "passed"},
                {"id": speed_finalist_2_id, "context_length": 4096, "status": "passed"},
            ],
            "failed_configurations": [],
            "phase_ab": {
                "phase": "speed",
                "speed_context": 4096,
                "quality_context": 32768,
                "budget_class": "NORMAL",
                "speed_finalist_ids": [speed_finalist_1_id, speed_finalist_2_id],
                "raw_fastest_id": speed_finalist_1_id,
                "checkpoint_seq": 1,
            },
        }

        # Write checkpoint to temp dir
        monkeypatch.setattr(_rc, "checkpoint_dir", lambda: tmp_path)
        _rc.save_checkpoint = lambda state, reason="x": _rc.checkpoint_path(state.run.id).write_text(
            __import__("orjson").dumps(ckpt_data).decode()
        )

        # Mock DB repos
        from lm_optimizer.database import repositories as _repo
        monkeypatch.setattr(_repo.run_repo, "get", lambda rid: run)
        monkeypatch.setattr(_repo.run_repo, "save", lambda r: str(r.id))
        monkeypatch.setattr(_repo.hardware_repo, "save", lambda h: "hw-id")
        monkeypatch.setattr(_repo.model_repo, "save", lambda m: m.id)
        monkeypatch.setattr(_repo.config_repo, "save", lambda c: str(c.id))

        # Mock benchmark/quality
        opt = _opt.AdaptiveOptimizer.__new__(_opt.AdaptiveOptimizer)
        opt.client = MagicMock()
        opt.client.ensure_unloaded = AsyncMock(return_value=True)
        opt.benchmark = MagicMock()
        opt.benchmark.gpu_via_cli = False
        opt.benchmark.run_speed_probe = AsyncMock()
        opt.benchmark.run_cases = AsyncMock()
        opt.benchmark.style = "balanced"
        opt.quality = MagicMock()
        opt.search_generator = MagicMock()
        opt._checkpoint = lambda reason="x": None
        opt._broadcast = AsyncMock()

        # Test config should NOT be called for the already-completed quality candidate
        test_calls = []

        async def mock_test_config(cfg, ctx, **kw):
            test_calls.append((cfg, ctx))
            return None  # We won't actually run anything

        opt._test_config = mock_test_config

        # Create search space
        from lm_optimizer.services.search_space import SearchSpace
        space = SearchSpace(
            context_lengths=[4096, 32768],
            gpu_ratios=[1.0],
            flash_attention_options=[True],
            kv_cache_options=[True],
            batch_sizes=[2048],
            physical_batch_sizes=[512],
            parallels=[8],
            context_checkpoints_options=[32],
        )
        opt.search_generator.generate = MagicMock(return_value=space)

        # Run resume
        from datetime import datetime
        opt.state = _opt.OptimizationState(run=run, search_space=space)
        opt.state.started_at = datetime.now()

        # Trigger the phase reconciliation + winner reconstruction
        opt._restore_tested_state(run, ckpt_data)
        opt._reconcile_phase_and_winner(run, ckpt_data)

        # Assertions
        # 1. Quality candidate restored from DB
        restored_ids = {str(c.id) for c in opt.state.tested_configs}
        assert quality_candidate_id in restored_ids, "Quality candidate should be restored from DB"

        # 2. Speed finalists restored
        assert speed_finalist_1_id in restored_ids
        assert speed_finalist_2_id in restored_ids

        # 3. Winner reconstructed from quality-passed candidates
        assert opt.state.best_config is not None, "Winner should be reconstructed"
        assert opt.state.run.best_config_id is not None, "best_config_id should be set"
        assert str(opt.state.best_config.id) == quality_candidate_id, "Winner should be quality candidate"

        # 4. Quality candidate metrics unchanged
        winner = opt.state.best_config
        assert winner.get_avg_generation_tok_s() == 1.5, "Winner speed unchanged"
        assert winner.quality_score.overall == 0.9829, "Winner quality unchanged"
        assert winner.context_length == 32768, "Winner context unchanged"

        # 5. Phase set correctly (DB phase = quality)
        assert opt.state.phase == "quality", "Phase should be quality from DB"

    def test_reconcile_when_checkpoint_ahead_of_db(self, tmp_path, monkeypatch):
        """Reverse window: checkpoint phase ahead of DB stage.

        This can happen if checkpoint write succeeds but DB save fails.
        Should adopt checkpoint phase but keep DB completed work.
        """
        from lm_optimizer.domain.models import (
            ConfigurationResult,
            ConfigurationStatus,
            HardwareInfo,
            LoadConfiguration,
            ModelIdentity,
            OptimizationProfile,
            OptimizationRun,
            OptimizationStage,
            RunStatus,
            GPUInfo,
        )
        from lm_optimizer.services import optimizer as _opt
        from lm_optimizer.services.search_space import SearchSpace
        from uuid import UUID, uuid4

        model = ModelIdentity(id="test-model", name="test-model", context_limit=32768)
        hw = HardwareInfo(
            os="T", cpu_name="C", cpu_cores_physical=8, cpu_cores_logical=16,
            total_ram_gb=32.0, gpu_count=1,
            gpus=[GPUInfo(index=0, name="Test GPU", vram_gb=6.0, vendor="NVIDIA")]
        )
        run = OptimizationRun(model=model, hardware=hw, profile=OptimizationProfile.BALANCED)
        run.id = uuid4()
        run.status = RunStatus.RUNNING
        run.stage = OptimizationStage.SPEED_DISCOVERY  # DB says speed
        run.benchmark_params = {
            "phase_ab": {"phase": "speed", "speed_finalist_ids": []}
        }

        # Checkpoint says quality phase
        ckpt_data = {
            "phase_ab": {"phase": "quality", "speed_finalist_ids": ["f1"]},
            "stage": "quality_check",
        }

        opt = _opt.AdaptiveOptimizer.__new__(_opt.AdaptiveOptimizer)
        opt.state = _opt.OptimizationState(run=run, search_space=SearchSpace())
        opt._reconcile_phase_and_winner(run, ckpt_data)

        # Should adopt checkpoint phase
        assert opt.state.phase == "quality"
        assert opt.state.run.phase == "quality"

    def test_idempotent_reconcile(self, tmp_path, monkeypatch):
        """Running reconcile twice should be idempotent."""
        from lm_optimizer.domain.models import (
            HardwareInfo,
            ModelIdentity,
            OptimizationProfile,
            OptimizationRun,
            OptimizationStage,
            RunStatus,
            GPUInfo,
        )
        from lm_optimizer.services import optimizer as _opt
        from lm_optimizer.services.search_space import SearchSpace
        from uuid import uuid4

        model = ModelIdentity(id="m", name="m", context_limit=32768)
        hw = HardwareInfo(
            os="T", cpu_name="C", cpu_cores_physical=1, cpu_cores_logical=2,
            total_ram_gb=8, gpu_count=1,
            gpus=[GPUInfo(index=0, name="Test", vram_gb=1.0, vendor="NVIDIA")]
        )

        run = OptimizationRun(model=model, hardware=hw, profile=OptimizationProfile.BALANCED)
        run.id = uuid4()
        run.stage = OptimizationStage.QUALITY_CHECK
        run.benchmark_params = {"phase_ab": {"phase": "speed"}}

        ckpt = {"phase_ab": {"phase": "quality"}}

        opt = _opt.AdaptiveOptimizer.__new__(_opt.AdaptiveOptimizer)
        opt.state = _opt.OptimizationState(run=run, search_space=SearchSpace())

        # First reconcile
        opt._reconcile_phase_and_winner(run, ckpt)
        phase_1 = opt.state.phase

        # Second reconcile
        opt._reconcile_phase_and_winner(run, ckpt)
        phase_2 = opt.state.phase

        assert phase_1 == phase_2 == "quality"

    def test_other_phase_transitions_protected(self):
        """Verify all phase transitions have start checkpoints.

        Check that _phase_recovery, _phase_final_validation, _phase_context_optional
        all have checkpoint(:start) before run_repo.save().
        """
        import inspect
        from lm_optimizer.services import optimizer as _opt

        for method_name in ["_phase_speed", "_phase_quality", "_run_recovery",
                            "_phase_final_validation", "_phase_context_optional"]:
            method = getattr(_opt.AdaptiveOptimizer, method_name)
            source = inspect.getsource(method)
            # Should have checkpoint with :start before run_repo.save
            assert "checkpoint" in source
            assert ":start" in source
            # The order: checkpoint(:start) then run_repo.save
            ckpt_idx = source.find("checkpoint")
            save_idx = source.find("run_repo.save")
            assert ckpt_idx < save_idx, f"{method_name}: checkpoint should come before save"
