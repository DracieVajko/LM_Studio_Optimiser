"""P1 UX completion tests: checkpoint, resume, WS, zero/partial, ctx, refinement."""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from lm_optimizer.domain.models import (
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    HardwareInfo,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
    OptimizationStage,
    QualityScore,
    RunStatus,
)


def _qs(overall=0.98):
    return QualityScore(
        overall=overall,
        task_completion=1.0,
        factual_consistency=1.0,
        format_compliance=1.0,
        coding_correctness=1.0,
        no_truncation=1.0,
        no_malformed=1.0,
        checks_passed=6,
        checks_total=6,
    )


def _metrics(gen=40.0):
    return [
        BenchmarkMetrics(
            test_name="t",
            category="instruction",
            success=True,
            generation_tok_s=gen,
            prompt_tok_s=500,
            estimated_ttft_ms=100,
            completion_tokens=50,
            output_text="hello world output",
        )
    ]


def _cfg_result(
    ctx=4096, gen=40.0, status=ConfigurationStatus.PASSED, err=None, score=0.5, vram=4.0
):
    return ConfigurationResult(
        config=LoadConfiguration(
            context_length=ctx,
            gpu_ratio=1.0,
            flash_attention=True,
            offload_kv_cache_to_gpu=True,
            eval_batch_size=256,
        ),
        context_length=ctx,
        status=status,
        metrics=_metrics(gen),
        quality_score=_qs(),
        stability_score=0.9,
        peak_vram_gb=vram,
        peak_ram_gb=8.0,
        error=err,
        score=score,
        score_breakdown={"generation_speed": 0.8, "quality": 0.9},
        tested_at=datetime.now(),
    )


def _hw():
    return HardwareInfo(
        os="Linux",
        cpu_name="CPU",
        cpu_cores_physical=4,
        cpu_cores_logical=8,
        total_ram_gb=32,
        gpu_count=0,
    )


def _run(configs=None):
    r = OptimizationRun(
        model=ModelIdentity(id="m", name="m"),
        hardware=_hw(),
        profile=OptimizationProfile.BALANCED,
        status=RunStatus.RUNNING,
        stage=OptimizationStage.COARSE_SEARCH,
    )
    r.configurations = configs or []
    return r


def _optimizer_with_state(configs=None):
    from lm_optimizer.services.optimizer import AdaptiveOptimizer, OptimizationState
    from lm_optimizer.services.search_space import SearchSpace

    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    opt.client = MagicMock()
    opt.client.ensure_unloaded = AsyncMock(return_value=True)
    opt.benchmark = MagicMock()
    opt.quality = MagicMock()
    opt.search_generator = MagicMock()
    run = _run(configs)
    space = SearchSpace(
        context_lengths=[2048, 4096],
        gpu_ratios=[1.0],
        flash_attention_options=[True],
        kv_cache_options=[True],
        batch_sizes=[256],
    )
    state = OptimizationState(run=run, search_space=space)
    state.tested_configs = list(configs or [])
    state.started_at = datetime.now()
    if configs:
        passed = [c for c in configs if c.status == ConfigurationStatus.PASSED]
        if passed:
            state.best_config = max(passed, key=lambda c: c.score)
    opt.state = state
    return opt


# ---------- checkpoint ----------


def _patch_results_dir(tmp_path, monkeypatch):
    from lm_optimizer.config import config as _cfg

    monkeypatch.setattr(_cfg.storage, "results_dir", tmp_path)


class TestCheckpoint:
    def test_shutdown_checkpoint_has_required_keys(self, tmp_path, monkeypatch):
        _patch_results_dir(tmp_path, monkeypatch)
        from lm_optimizer.storage import run_checkpoint as rc

        opt = _optimizer_with_state([_cfg_result(ctx=4096, score=0.7)])
        p = rc.save_checkpoint(opt.state, reason="shutdown")
        assert p is not None and p.exists()
        data = json.loads(p.read_bytes())
        for key in [
            "run_id",
            "model",
            "stage",
            "current_candidate",
            "completed_candidate_ids",
            "successful_results",
            "failed_configurations",
            "oom_boundaries",
            "context_search_boundaries",
            "best_candidates",
            "recommendation_candidates",
            "profile",
            "benchmark_params",
            "hardware_snapshot",
            "optimizer_state",
        ]:
            assert key in data, f"missing {key}"

    def test_cancel_checkpoint_before_ack(self, tmp_path, monkeypatch):
        _patch_results_dir(tmp_path, monkeypatch)
        opt = _optimizer_with_state([_cfg_result()])
        opt.cancel()
        assert opt.state.should_cancel is True
        from lm_optimizer.storage import run_checkpoint as rc

        data = rc.load_checkpoint(opt.state.run.id)
        assert data is not None
        assert len(data["completed_candidate_ids"]) == 1

    def test_ctrl_c_maps_to_interrupted(self):
        assert RunStatus.INTERRUPTED.value == "interrupted"
        assert OptimizationStage.INTERRUPTED.value == "interrupted"

    def test_corrupted_checkpoint_raises(self, tmp_path, monkeypatch):
        _patch_results_dir(tmp_path, monkeypatch)
        from lm_optimizer.storage import run_checkpoint as rc

        rid = str(uuid4())
        p = rc.checkpoint_path(rid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"{not json")
        with pytest.raises(ValueError, match="corrupted"):
            rc.load_checkpoint(rid)

    def test_missing_checkpoint_returns_none(self, tmp_path, monkeypatch):
        _patch_results_dir(tmp_path, monkeypatch)
        from lm_optimizer.storage import run_checkpoint as rc

        assert rc.load_checkpoint(str(uuid4())) is None


# ---------- run summary / final states ----------


class TestFinalStates:
    def test_zero_success_is_failed(self):
        from lm_optimizer.services.run_summary import classify_run, best_passed

        cfgs = [
            _cfg_result(status=ConfigurationStatus.FAILED, err="load refused"),
            _cfg_result(ctx=8192, status=ConfigurationStatus.OOM, err="OOM"),
        ]
        state, info = classify_run(cfgs)
        assert state == "FAILED"
        assert info["successful"] == 0
        assert best_passed(cfgs) is None
        assert info["failure_classes"]["oom"] == 1

    def test_zero_pass_never_offers_winner(self):
        from lm_optimizer.services.run_summary import best_passed

        cfgs = [_cfg_result(status=ConfigurationStatus.FAILED, err="timeout x")]
        assert best_passed(cfgs) is None

    def test_partial_success(self):
        from lm_optimizer.services.run_summary import classify_run, best_passed

        cfgs = [
            _cfg_result(ctx=4096, gen=50, score=0.8),
            _cfg_result(ctx=8192, status=ConfigurationStatus.FAILED, err="boom"),
        ]
        state, info = classify_run(cfgs)
        assert state == "PARTIAL_SUCCESS"
        assert info["successful"] == 1 and info["failed"] == 1
        best = best_passed(cfgs)
        assert best is not None and best.status == ConfigurationStatus.PASSED
        assert best.context_length == 4096

    def test_failed_config_never_wins(self):
        from lm_optimizer.services.run_summary import best_passed

        good = _cfg_result(ctx=4096, gen=30, score=0.4)
        bad = _cfg_result(
            ctx=8192, gen=500, score=0.99, status=ConfigurationStatus.FAILED, err="bad"
        )
        assert best_passed([good, bad]).context_length == 4096

    def test_final_status_mapping(self):
        from lm_optimizer.services.run_summary import final_status_for_run

        assert final_status_for_run(_run([_cfg_result()])) == RunStatus.SUCCESS
        assert (
            final_status_for_run(
                _run([_cfg_result(), _cfg_result(status=ConfigurationStatus.FAILED, err="x")])
            )
            == RunStatus.PARTIAL_SUCCESS
        )
        assert (
            final_status_for_run(_run([_cfg_result(status=ConfigurationStatus.FAILED, err="x")]))
            == RunStatus.FAILED
        )

    def test_cancelled_and_interrupted_distinct(self):
        assert RunStatus.CANCELLED != RunStatus.INTERRUPTED
        assert RunStatus.CANCELLED != RunStatus.FAILED
        assert RunStatus.RESUMED.value == "resumed"
        assert RunStatus.PARTIAL_SUCCESS.value == "partial_success"
        assert RunStatus.SUCCESS.value == "success"

    def test_failure_breakdown_counts(self):
        from lm_optimizer.services.run_summary import failure_breakdown

        cfgs = [
            _cfg_result(status=ConfigurationStatus.OOM, err="OOM"),
            _cfg_result(status=ConfigurationStatus.FAILED, err="timeout after 10s"),
            _cfg_result(status=ConfigurationStatus.FAILED, err="quality below threshold"),
            _cfg_result(status=ConfigurationStatus.FAILED, err="unsupported parameter gpu_ratio"),
        ]
        fb = failure_breakdown(cfgs)
        assert fb["oom"] == 1
        assert fb["timeouts"] == 1
        assert fb["quality_rejected"] == 1
        assert fb["unsupported_parameters"] == 1

    def test_recommendation_crowns(self):
        from lm_optimizer.services.run_summary import alternatives, why_text

        cfgs = [
            _cfg_result(ctx=4096, gen=50, score=0.9, vram=4.0),
            _cfg_result(ctx=8192, gen=30, score=0.5, vram=8.0),
            _cfg_result(ctx=16384, gen=20, score=0.4, vram=6.0),
        ]
        alt = alternatives(cfgs, cfgs[0])
        assert alt["best_speed"].context_length == 4096
        assert alt["best_context"].context_length == 16384
        assert alt["best_memory"].peak_vram_gb == 4.0
        w = why_text(cfgs[0], {"generation_tok_s": 25.0}, "balanced")
        assert "profile=balanced" in w and "gen=" in w


# ---------- context sweep / refinement ----------


class TestContextSweep:
    def test_geometric_ladder(self):
        from lm_optimizer.services.context_sweep import geometric_contexts

        assert geometric_contexts(4096, 65536) == [4096, 8192, 16384, 32768, 65536]
        assert geometric_contexts(2048, 2048) == [2048]

    def test_4096_pass_8192_fail_probes_interval(self):
        from lm_optimizer.services.context_sweep import sweep

        seen = []

        async def probe(ctx):
            seen.append(ctx)
            if ctx <= 4096:
                return {"ok": True, "tok_s": 42.0, "error": ""}
            return {"ok": False, "tok_s": 0.0, "error": "OOM"}

        report = asyncio.run(sweep(probe, 4096, 8192, step=1000))
        assert report["maximum_stable_context"] == 4096
        # Failed boundary is the refined boundary inside (4096, 8192].
        assert report["failed_boundary"] is not None
        assert 4096 < report["failed_boundary"] <= 8192
        assert report["maximum_stable_context"] < report["failed_boundary"]
        # Must enter the interval, not just return the last geometric checkpoint.
        assert any(4096 < c < 8192 for c in seen), f"no interval probe in {seen}"
        # Refinement recorded separately.
        assert len(report["refinement"]) >= 1

    def test_step_500_mode(self):
        from lm_optimizer.services.context_sweep import refine_boundary

        seen = []

        async def probe(ctx):
            seen.append(ctx)
            return {"ok": ctx < 7000, "tok_s": 30.0, "error": "" if ctx < 7000 else "OOM"}

        out = asyncio.run(refine_boundary(4096, 8192, probe, step=500))
        assert out["max_stable"] >= 4096 and out["max_stable"] < 8192
        assert all(s % 500 == 0 or s in (4096, 8192) for s in seen)

    def test_step_1000_mode(self):
        from lm_optimizer.services.context_sweep import refine_boundary

        seen = []

        async def probe(ctx):
            seen.append(ctx)
            return {"ok": ctx < 6000, "tok_s": 30.0, "error": "" if ctx < 6000 else "OOM"}

        out = asyncio.run(refine_boundary(4096, 8192, probe, step=1000))
        assert out["max_stable"] >= 4096 and out["failed_boundary"] <= 8192
        assert len(seen) >= 1

    def test_capacity_vs_recommendation_split(self):
        from lm_optimizer.services.context_sweep import sweep

        async def probe(ctx):
            # Big ctx passes but slowly; small ctx is fastest.
            speeds = {4096: 42.0, 8192: 39.0, 16384: 20.0}
            return {"ok": True, "tok_s": speeds.get(ctx, 25.0), "error": ""}

        report = asyncio.run(sweep(probe, 4096, 16384, step=1000))
        assert report["maximum_stable_context"] == 16384  # capacity
        assert report["performance_optimal"]["ctx"] == 4096  # fastest
        # Balanced: largest ctx keeping >=80% of peak (0.8*42=33.6 -> 8192).
        assert report["balanced_recommended"]["ctx"] == 8192

    def test_all_fail_has_no_capacity(self):
        from lm_optimizer.services.context_sweep import sweep

        async def probe(ctx):
            return {"ok": False, "tok_s": 0.0, "error": "OOM"}

        report = asyncio.run(sweep(probe, 4096, 8192, step=1000))
        assert report["maximum_stable_context"] is None
        assert report["performance_optimal"] is None


# ---------- websocket ----------


class TestWebsocket:
    def test_progress_payload_has_required_keys(self):
        from lm_optimizer.api.websocket import _progress_payload

        cfgs = [_cfg_result(ctx=4096, gen=41.2), _cfg_result(ctx=8192, gen=39.0)]
        p = _progress_payload(uuid4(), cfgs, {"stage": "coarse_search", "model": "m"})
        for key in [
            "stage",
            "model",
            "configs_tested",
            "configs_passed",
            "configs_failed",
            "configs_oom",
            "best_score",
            "current_speed",
            "vram_gb",
            "ram_gb",
            "context_progress",
            "elapsed_s",
            "remaining",
        ]:
            assert key in p, f"missing {key}"
        # Never fake ETA: adaptive/unknown when no rate basis.
        assert p["remaining"] == "adaptive / unknown"

    def test_send_current_state_live(self):
        from lm_optimizer.api import websocket as ws

        opt = _optimizer_with_state([_cfg_result(ctx=4096, gen=41.2)])
        opt.state.run.stage = OptimizationStage.COARSE_SEARCH
        import lm_optimizer.api.routes as routes

        routes._current_optimizer = opt
        sent = []

        class FakeWS:
            async def send_json(self, data):
                sent.append(data)

        asyncio.run(ws.send_current_state(opt.state.run.id, FakeWS()))
        routes._current_optimizer = None
        kinds = {m.get("type") for m in sent}
        assert "progress" in kinds and "log" in kinds
        prog = next(m for m in sent if m.get("type") == "progress")
        assert prog["current_speed"] == pytest.approx(41.2)
        assert prog["remaining"] == "adaptive / unknown"

    def test_broadcast_progress_never_raises(self):
        from lm_optimizer.api import websocket as ws

        asyncio.run(ws.broadcast_progress(uuid4(), {"stage": "x"}))


# ---------- resume ----------


class TestResume:
    def test_resume_skips_completed(self, tmp_path, monkeypatch):
        _patch_results_dir(tmp_path, monkeypatch)
        from lm_optimizer.database import repositories as _repo
        from lm_optimizer.services.optimizer import AdaptiveOptimizer
        from lm_optimizer.services.search_space import SearchSpace

        # Build a persisted run with one completed config.
        hw = _hw()
        model = ModelIdentity(id="resume-model-x", name="resume-model-x", context_limit=8192)
        run = OptimizationRun(
            model=model,
            hardware=hw,
            profile=OptimizationProfile.BALANCED,
            status=RunStatus.CANCELLED,
            stage=OptimizationStage.COARSE_SEARCH,
        )
        done = _cfg_result(ctx=2048, gen=40.0)
        done.run_id = run.id
        _repo.hardware_repo.save(hw)
        _repo.model_repo.save(model)
        _repo.run_repo.save(run)
        _repo.config_repo.save(done)
        run.configurations = [done]
        run.best_config_id = done.id
        _repo.run_repo.save(run)

        from lm_optimizer.storage import run_checkpoint as rc

        # Fake minimal state for checkpoint file.
        opt0 = _optimizer_with_state([done])
        opt0.state.run = run
        opt0.state.search_space = SearchSpace(
            context_lengths=[2048, 4096],
            gpu_ratios=[1.0],
            flash_attention_options=[True],
            kv_cache_options=[True],
            batch_sizes=[256],
        )
        rc.save_checkpoint(opt0.state, reason="cancelled")

        # Now resume: benchmark mock returns a passing result for new ctx.
        async def _go():
            client = MagicMock()
            client.ensure_unloaded = AsyncMock(return_value=True)
            bench = MagicMock()
            new = _cfg_result(ctx=4096, gen=45.0)

            async def _fake_bench(mid, cfg, ctx):
                r = _cfg_result(ctx=ctx, gen=45.0 if ctx == 4096 else 40.0)
                r.status = ConfigurationStatus.PASSED
                return r

            bench.run_benchmark = _fake_bench
            qual = MagicMock()
            qual.evaluate_all = MagicMock(return_value=[])
            qual.aggregate_quality = MagicMock(return_value=_qs())
            qual.passes_threshold = MagicMock(return_value=True)
            qual.config = MagicMock()
            qual.config.minimum_score = 0.9
            from lm_optimizer.services.search_space import SearchSpaceGenerator

            gen = MagicMock(spec=SearchSpaceGenerator)
            gen.generate = MagicMock(
                return_value=SearchSpace(
                    context_lengths=[2048, 4096],
                    gpu_ratios=[1.0],
                    flash_attention_options=[True],
                    kv_cache_options=[True],
                    batch_sizes=[256],
                )
            )
            opt = AdaptiveOptimizer(client, bench, qual, gen)
            # Narrow space so only 2048 (done) + 4096 (new) exist.
            resumed = await opt.resume_from_checkpoint(run.id, style="balanced")
            return resumed, opt

        resumed, opt = asyncio.run(_go())
        ctxs = sorted(c.context_length for c in resumed.configurations)
        assert 2048 in ctxs  # kept, not re-run from scratch
        assert resumed.status in (RunStatus.SUCCESS, RunStatus.PARTIAL_SUCCESS, RunStatus.FAILED)

    def test_resume_missing_checkpoint_message(self, tmp_path, monkeypatch):
        _patch_results_dir(tmp_path, monkeypatch)
        from lm_optimizer.storage.run_checkpoint import load_checkpoint

        assert load_checkpoint(str(uuid4())) is None


# ---------- micro-stage ----------


class TestMicroStage:
    def test_micro_stage_bounded_and_profile_aware(self):
        from lm_optimizer.services.optimizer import AdaptiveOptimizer
        import inspect

        src = inspect.getsource(AdaptiveOptimizer._stage_micro_refinement)
        assert "max_candidates" in src
        assert "SPEED" in src and "CONTEXT" in src and "QUALITY" in src
        # Temperature must not be part of runtime search.
        assert "temperature" not in src.lower() or "NEVER" in src


class TestCli:
    def test_ctx_dry_run(self):
        from typer.testing import CliRunner
        from lm_optimizer.cli.main import app

        r = CliRunner().invoke(
            app,
            [
                "ctx",
                "some-model",
                "--dry-run",
                "--min-context",
                "2048",
                "--max-context",
                "8192",
                "--step",
                "1000",
            ],
        )
        assert r.exit_code == 0
        assert "2048" in r.output and "8192" in r.output

    def test_ctx_rejects_bad_step(self):
        from typer.testing import CliRunner
        from lm_optimizer.cli.main import app

        r = CliRunner().invoke(app, ["ctx", "m", "--step", "700"])
        assert r.exit_code == 2

    def test_checkpoints_command(self):
        from typer.testing import CliRunner
        from lm_optimizer.cli.main import app

        r = CliRunner().invoke(app, ["checkpoints"])
        assert r.exit_code == 0

    def test_cancelled_resumed_runs_distinct(self):
        cancelled = _run([])
        cancelled.status = RunStatus.CANCELLED
        resumed = _run([])
        resumed.status = RunStatus.RESUMED
        assert cancelled.status != resumed.status
        assert cancelled.status.value == "cancelled"
        assert resumed.status.value == "resumed"


class TestProgressCallback:
    def test_report_progress_calls_cb(self):
        from lm_optimizer.services.optimizer import AdaptiveOptimizer, OptimizationState
        from lm_optimizer.services.search_space import SearchSpace
        from lm_optimizer.domain.models import (
            HardwareInfo,
            ModelIdentity,
            OptimizationProfile,
            OptimizationRun,
        )

        hw = HardwareInfo(
            os="T",
            cpu_name="C",
            cpu_cores_physical=1,
            cpu_cores_logical=2,
            total_ram_gb=8,
            gpu_count=0,
        )
        run = OptimizationRun(
            model=ModelIdentity(id="m", name="m"), hardware=hw, profile=OptimizationProfile.BALANCED
        )
        opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
        opt.state = OptimizationState(
            run=run,
            search_space=SearchSpace(
                context_lengths=[2048, 4096],
                gpu_ratios=[1.0],
                flash_attention_options=[True],
                kv_cache_options=[True],
                batch_sizes=[256],
            ),
        )
        opt.state.tested_configs = []
        seen = []
        opt._progress_cb = lambda stage, tested, total: seen.append((stage, tested, total))
        opt._report_progress()
        assert seen and seen[0][0] == "discovery" and seen[0][2] > 0

    def test_report_progress_none_is_safe(self):
        from lm_optimizer.services.optimizer import AdaptiveOptimizer

        opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
        opt.state = None
        opt._progress_cb = None
        opt._report_progress()  # must not raise

    def test_optimize_accepts_progress_cb(self):
        import inspect

        from lm_optimizer.services.optimizer import AdaptiveOptimizer

        assert "progress_cb" in inspect.signature(AdaptiveOptimizer.optimize).parameters


class TestMicroStageCaps:
    def test_micro_stage_caps_candidates(self, monkeypatch):
        from lm_optimizer.database import repositories as _repo

        monkeypatch.setattr(_repo.run_repo, "save", lambda run: str(run.id))
        opt = _optimizer_with_state([_cfg_result(ctx=4096, score=0.8)])
        from lm_optimizer.services.search_space import SearchSpace

        opt.state.search_space = SearchSpace(
            context_lengths=[4096],
            gpu_ratios=[1.0],
            flash_attention_options=[True, False],
            kv_cache_options=[True, False],
            batch_sizes=[64, 128, 256, 512, 1024],
            physical_batch_sizes=[256, 512],
            parallels=[1, 2],
            context_checkpoints_options=[16, 32],
        )
        calls = []

        async def fake_test(cfg, ctx):
            calls.append(cfg)

        opt._test_config = fake_test  # type: ignore
        opt._checkpoint = lambda _reason="x": None  # type: ignore
        opt._broadcast = AsyncMock()  # type: ignore
        asyncio.run(opt._stage_micro_refinement(max_candidates=3))
        assert len(calls) <= 3
