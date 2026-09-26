"""Bugfix-pass regressions (E2E findings 1-6). No live models; mocked seams."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from lm_optimizer.domain.models import (
    BenchmarkCase,
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    HardwareInfo,
    LoadConfiguration,
    ModelIdentity,
    OptimizationRun,
)
from lm_optimizer.services.optimizer import AdaptiveOptimizer, OptimizationState
from lm_optimizer.services.search_space import SearchSpace


def _hw():
    return HardwareInfo(
        os="windows", cpu_name="c", cpu_cores_physical=8, cpu_cores_logical=16,
        total_ram_gb=16.0, gpu_count=1,
    )


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
    opt.benchmark.measure_throughput = AsyncMock(
        return_value={"aggregate_tok_s": 10.0, "parallelism": 2}
    )
    opt.benchmark.create_cases_for_context = MagicMock(
        return_value=[
            BenchmarkCase(name="coding_task", category="coding", prompt="p", max_tokens=64),
            BenchmarkCase(name="structured_output", category="format", prompt="q", max_tokens=64),
        ]
    )
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
    from lm_optimizer.domain.models import QualityScore

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
    r.generation = {"phase": "quality", "stage": "quality_check", "score_context": False}
    return r


class TestBug1NullChecks:
    def test_invalid_json_is_explicit_failure(self):
        from lm_optimizer.services.quality import QualityEvaluator

        q = QualityEvaluator().evaluate("m", "structured_output", "{bad json,,,")
        assert q.overall == 0.0
        assert q.checks_passed == 0
        assert q.checks_total == 6
        assert q.details.get("json_valid") is False

    def test_failed_test_ranking_never_compares_none(self):
        from lm_optimizer.services.recovery import rank_failed_tests

        qbt = {
            "short_instruction": {"overall": 1.0, "checks_passed": 6, "checks_total": 6},
            "structured_output": {"overall": 0.0, "checks_passed": None, "checks_total": None},
            "coding_task": {"overall": 0.7, "checks_passed": 4, "checks_total": 6},
            "mystery": {"overall": None, "checks_passed": None, "checks_total": None},
        }
        ranked = rank_failed_tests(qbt)
        assert ranked[0] in ("structured_output", "mystery")  # invalid first
        assert "short_instruction" not in ranked  # passing test excluded

    def test_recovery_all_structured_fail_bounded(self, monkeypatch):
        """4B scenario: every finalist fails the same structured test, null checks."""
        _no_db(monkeypatch)
        opt = _opt()
        failed = _probe_result(70.0)
        failed.config = LoadConfiguration(context_length=8192, eval_batch_size=1024)
        failed.context_length = 8192
        failed.status = ConfigurationStatus.QUALITY_FAILED
        failed.generation = {
            "quality_by_test": {
                "structured_output": {"overall": 0.0, "checks_passed": None, "checks_total": None},
                "short_instruction": {"overall": 1.0, "checks_passed": 6, "checks_total": 6},
            }
        }
        opt.benchmark.run_cases = AsyncMock(return_value=_probe_result(69.0))
        from lm_optimizer.domain.models import QualityScore

        bad = QualityScore(
            overall=0.78, task_completion=0.8, factual_consistency=1.0,
            format_compliance=0.8, coding_correctness=1.0, no_truncation=1.0,
            no_malformed=0.8, checks_passed=24, checks_total=24,
        )
        opt.quality.evaluate_all = MagicMock(return_value={"structured_output": bad})
        opt.quality.aggregate_quality = MagicMock(return_value=bad)
        opt.quality.passes_threshold = MagicMock(return_value=False)
        last_good = LoadConfiguration(context_length=8192, eval_batch_size=512)

        recovered, culprit = asyncio.run(opt._run_recovery(failed, last_good))
        assert recovered is None  # bounded: nothing fixes a weak model
        assert opt.state.recovery_log  # attempts recorded, no crash
        subset_names = opt.benchmark.run_cases.call_args_list[0].args[3]
        assert [c.name for c in subset_names] == ["structured_output"]


class TestResumeRetriesFailed:
    def test_failed_s0_retried_on_resume(self, monkeypatch):
        """Stuck-run safety: a FAILED probe is retried, not skipped forever."""
        _no_db(monkeypatch)
        opt = _opt()
        anchor = opt._default_anchor()
        anchor.context_length = 2048
        failed_s0 = _probe_result(0.0)
        failed_s0.config = anchor
        failed_s0.status = ConfigurationStatus.LOAD_FAILED
        failed_s0.score = None
        failed_s0.generation = {"phase": "speed"}
        opt.state.tested_configs = [failed_s0]
        probed = []
        opt.benchmark.run_speed_probe = AsyncMock(
            side_effect=lambda m, cfg, ctx, **kw: probed.append(cfg.to_dict())
            or _probe_result(50.0)
        )
        asyncio.run(opt._phase_speed())
        anchor_key = opt._candidate_key(anchor)

        def _key(d):
            cfg = LoadConfiguration(
                **{k: v for k, v in d.items() if hasattr(LoadConfiguration, k)}
            )
            return opt._candidate_key(cfg)

        assert any(_key(d) == anchor_key for d in probed)


class TestBug7FinalistLimit:
    def _opt_with_params(self, params):
        opt = _opt()
        opt.state.run.benchmark_params = params
        opt.state.budget_class = "FAST"
        return opt

    def test_key_missing_uses_budget_default(self):
        assert self._opt_with_params({})._finalist_limit() == 5  # FAST default

    def test_none_means_default(self):
        opt = self._opt_with_params({"quality_finalist_limit": None})
        assert opt._finalist_limit() == 5  # not the broken fallback 4

    def test_explicit_int_respected(self):
        assert self._opt_with_params({"quality_finalist_limit": 2})._finalist_limit() == 2

    def test_six_finalists_all_tested_once(self, monkeypatch):
        """5 band + contrarian: all 6 reach QUALITY, exactly once each."""
        _no_db(monkeypatch)
        opt = self._opt_with_params({"quality_finalist_limit": None})
        band = [_quality_result(70.0 - i, eval_batch_size=512) for i in range(5)]
        for i, r in enumerate(band):
            r.config = LoadConfiguration(context_length=2048, eval_batch_size=100 + i)
        contra = _quality_result(60.0, eval_batch_size=999)
        seen = []

        def _key(cfg):
            return {k: v for k, v in cfg.to_dict().items() if k != "context_length"}

        async def _tc(cfg, ctx, **kw):
            seen.append(_key(cfg))
            match = next(r for r in band + [contra] if _key(r.config) == _key(cfg))
            return match

        opt._test_config = AsyncMock(side_effect=_tc)  # type: ignore
        front = {"winner": band[0], "finalists": band, "contrarian": contra}
        safe, failed = asyncio.run(opt._phase_quality(front))
        assert len(seen) == 6  # all six sent to QUALITY...
        assert len({tuple(sorted(d.items())) for d in seen}) == 6  # ...exactly once each
        assert _key(contra.config) in seen  # contrarian actually tested
        assert len(safe) == 6


class TestBug2WinnerSelection:
    def test_recompute_leaves_probes_scoreless(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        opt.state.score_context = False
        probe = _probe_result(85.2)
        valid = _quality_result(75.5)
        opt.state.tested_configs = [probe, valid]
        opt.state.best_config = valid
        opt._recompute_all_scores()
        assert probe.score is None  # never receives synthetic quality/score
        assert opt.state.best_config is valid

    def test_cli_and_report_agree_on_validated(self, capsys, tmp_path):
        from lm_optimizer.cli.main import _display_optimization_result
        from lm_optimizer.services.reporting import save_best_report

        probe = _probe_result(85.2)
        probe.score = 0.978  # legacy-scored probe (pre-fix data shape)
        valid = _quality_result(75.5)
        run = OptimizationRun(model=ModelIdentity(id="m", name="m"), hardware=_hw())
        run.configurations = [probe, valid]
        run.best_config_id = valid.id
        _display_optimization_result(run)
        out = capsys.readouterr().out
        # FINAL RECOMMENDATION Correctness row carries the validated winner.
        assert "0.990 (30/30 checks)" in out
        text = save_best_report(run, out_dir=tmp_path).read_text(encoding="utf-8")
        assert "75.5" in text  # ...and report agrees

    def test_quality_failed_cannot_win(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        bad = _quality_result(99.0)
        bad.status = ConfigurationStatus.QUALITY_FAILED
        bad.score = None
        opt._update_best(bad)
        assert opt.state.best_config is None

    def test_recovered_can_win(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        rec = _quality_result(68.0)
        opt._update_best(rec)
        assert opt.state.best_config is rec

    def test_validation_median_preferred(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        best = _quality_result(76.4)
        best.stability_score = 0.90
        opt.state.best_config = best
        vals = [_quality_result(76.0), _quality_result(75.5), _quality_result(75.0)]
        for v in vals:
            v.score = 0.79
        calls = []

        async def _tc(cfg, ctx, **kw):
            calls.append(1)
            return vals[len(calls) - 1]

        opt._test_config = AsyncMock(side_effect=_tc)  # type: ignore
        asyncio.run(opt._phase_final_validation())
        assert opt.state.best_config.get_avg_generation_tok_s() == 75.5


class TestBug3PhaseSeparation:
    def test_validation_stability_excludes_probes(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        # Same candidate key, SPEED probe at 84.9 + QUALITY result at 76.4.
        probe_cfg = LoadConfiguration(
            context_length=8192, flash_attention=True, offload_kv_cache_to_gpu=True,
            eval_batch_size=2048, physical_batch_size=512, parallel=4,
            context_checkpoints=32,
        )
        probe = _probe_result(84.9)
        probe.config = probe_cfg
        probe.context_length = 8192
        qual = _quality_result(76.4)
        qual.stability_score = 0.99
        opt.state.best_config = qual
        opt.state.tested_configs = [probe, qual]
        calls = []

        async def _tc(cfg, ctx, **kw):
            calls.append(1)
            return _quality_result(76.0)

        opt._test_config = AsyncMock(side_effect=_tc)  # type: ignore
        asyncio.run(opt._phase_final_validation())
        # Quality-only evidence (stability 0.99) -> minimum 2 reps, not 3+.
        assert len(calls) == 2

    def test_phase_tags_present(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        probe = _probe_result(50.0)
        assert probe.generation["phase"] == "speed"
        qual = _quality_result(60.0)

        async def _tc(cfg, ctx, **kw):
            return qual

        opt._test_config = AsyncMock(side_effect=_tc)  # type: ignore
        front = {"winner": probe, "finalists": [probe], "contrarian": None}
        asyncio.run(opt._phase_quality(front))
        assert qual.generation.get("phase") == "quality"


class TestBug4BreakdownWeights:
    def test_score_weights_stored_context_free(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        opt.state.score_context = False
        opt.state.run.profile_weights = {
            "generation_speed": 0.27, "prompt_speed": 0.15, "ttft": 0.12,
            "quality": 0.12, "stability": 0.12, "context": 0.11,
            "memory_efficiency": 0.11,
        }
        r = _quality_result(60.0)
        opt.state.tested_configs = []
        opt._score_eligible(r, False)
        assert r.generation["score_weights"]["context"] == 0.0
        assert abs(sum(r.generation["score_weights"].values()) - 1.0) < 1e-6

    def test_cli_breakdown_shows_actual_weights(self, capsys):
        from lm_optimizer.cli.main import _display_optimization_result

        valid = _quality_result(75.5)
        valid.score = 0.79
        valid.score_breakdown = {"generation_speed": 0.87, "context": 0.125}
        valid.generation["score_weights"] = {"generation_speed": 0.30, "context": 0.0}
        run = OptimizationRun(model=ModelIdentity(id="m", name="m"), hardware=_hw())
        run.configurations = [valid]
        run.best_config_id = valid.id
        _display_optimization_result(run)
        out = capsys.readouterr().out
        assert "context" in out
        # Actual (zero) context weight shown, not the 0.11 default.
        assert "0.00" in out

    def test_api_exposes_score_weights(self):
        from lm_optimizer.api.routes import _convert_config_result

        valid = _quality_result(75.5)
        valid.run_id = uuid4()
        valid.generation["score_weights"] = {"generation_speed": 0.30, "context": 0.0}
        resp = _convert_config_result(valid)
        assert resp.score_weights == {"generation_speed": 0.30, "context": 0.0}


class TestBug5S5Visibility:
    def test_s5_dedup_logged(self, monkeypatch):
        _no_db(monkeypatch)
        opt = _opt()
        c1 = _probe_result(85.0)
        c2 = _probe_result(84.0)
        # Identical keys -> S5 combos all duplicates.
        c2.config = LoadConfiguration(**c1.config.to_dict())
        opt.benchmark.run_speed_probe = AsyncMock(
            side_effect=lambda m, cfg, ctx, **kw: _probe_result(50.0)
        )
        opt.state.tested_configs = [c1, c2]
        for c in (c1, c2):
            c.generation = {"phase": "speed"}
        asyncio.run(opt._phase_speed())
        msgs = [e.get("message", "") for e in opt.state.event_log]
        assert any("S5" in m for m in msgs)


class TestBug6Traceback:
    def test_diagnostic_preserves_traceback(self):
        from lm_optimizer.services.optimizer import format_diagnostic

        try:
            raise ValueError("boom-test-marker")
        except ValueError as e:
            diag = format_diagnostic(e)
        assert diag["type"] == "ValueError"
        assert diag["message"] == "boom-test-marker"
        assert "Traceback" in diag["traceback"]
        assert "boom-test-marker" in diag["traceback"]
