"""Report transparency: storage detection, per-test rows, typicals."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

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


def _hw():
    return HardwareInfo(
        os="T", cpu_name="C", cpu_cores_physical=1, cpu_cores_logical=2, total_ram_gb=8, gpu_count=0
    )


def _metric(name="t", category="instruction", gen=30.0, ok=True):
    return BenchmarkMetrics(
        test_name=name,
        category=category,
        success=ok,
        generation_tok_s=gen,
        prompt_tok_s=500,
        estimated_ttft_ms=100,
        prompt_tokens=10,
        completion_tokens=50,
        total_tokens=60,
        output_text="measured",
    )


def _qs(overall=1.0):
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


def _result(ctx=16384, gen=30.0, score=0.7, stage="validation"):
    return ConfigurationResult(
        config=LoadConfiguration(
            context_length=ctx,
            flash_attention=False,
            offload_kv_cache_to_gpu=True,
            eval_batch_size=512,
        ),
        context_length=ctx,
        status=ConfigurationStatus.PASSED,
        metrics=[_metric(gen=gen)],
        quality_score=_qs(),
        score=score,
        score_breakdown={"generation_speed": 0.8},
        tested_at=datetime.now(),
        generation={
            "stage": stage,
            "style": "balanced",
            "temperature_overrides": {"instruction": 0.7},
            "quality_by_test": {"t": {"overall": 1.0, "checks_passed": 6, "checks_total": 6}},
        },
    )


def _run(configs):
    run = OptimizationRun(
        model=ModelIdentity(id="m-9", name="M9"),
        hardware=_hw(),
        profile=OptimizationProfile.BALANCED,
    )
    run.configurations = configs
    for c in configs:
        c.run_id = run.id
    best = max(configs, key=lambda c: c.score or 0)
    run.best_config_id = best.id
    return run, best


class TestSpeedRange:
    def test_fastest_slowest(self):
        from lm_optimizer.services.run_summary import speed_range

        cfgs = [
            _result(ctx=8192, gen=31.2, score=0.7),
            _result(ctx=16384, gen=24.0, score=0.4),
            _result(ctx=32768, gen=30.4, score=0.6),
        ]
        fast, slow = speed_range(cfgs)
        assert fast.context_length == 8192
        assert slow.context_length == 16384

    def test_empty_or_unscored(self):
        from lm_optimizer.services.run_summary import speed_range

        assert speed_range([]) is None
        bad = _result()
        bad.status = __import__(
            "lm_optimizer.domain.models", fromlist=["ConfigurationStatus"]
        ).ConfigurationStatus.FAILED
        bad.score = None
        assert speed_range([bad]) is None

    def test_report_has_range_and_defaults(self, tmp_path):
        from lm_optimizer.services.reporting import save_best_report

        run, _ = _run([_result(ctx=8192, gen=31.0), _result(ctx=16384, gen=24.0)])
        text = save_best_report(run, out_dir=tmp_path).read_text(encoding="utf-8")
        assert "Speed range across run" in text
        assert "server default (512)" in text


class TestMaxTokensScale:
    def test_default_unchanged(self):
        from lm_optimizer.services.benchmark import BenchmarkService

        cases = BenchmarkService(MagicMock()).create_cases_for_context(4096)
        by_name = {c.name: c.max_tokens for c in cases}
        assert by_name["long_context"] == 1024
        assert by_name["structured_output"] == 256

    def test_scale_halves_with_floor_and_records(self):
        import asyncio

        from lm_optimizer.domain.models import LoadConfiguration
        from lm_optimizer.services.benchmark import BenchmarkConfig, BenchmarkService

        svc = BenchmarkService(MagicMock(), benchmark_config=BenchmarkConfig(max_tokens_scale=0.5))
        cases = svc.create_cases_for_context(4096)
        by_name = {c.name: c.max_tokens for c in cases}
        assert by_name["long_context"] == 512
        assert by_name["structured_output"] == 128

        async def _go():
            client = MagicMock()
            load = MagicMock()
            load.success = True
            load.loaded_config = None
            client.load_model = AsyncMock(return_value=load)
            client.ensure_unloaded = AsyncMock(return_value=True)
            client.chat_completion = AsyncMock(
                return_value={
                    "choices": [{"message": {"content": "measured answer here"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
                    "_stats": {"tokens_per_second": 20.0, "time_to_first_token_seconds": 0.05},
                }
            )
            svc2 = BenchmarkService(
                client,
                benchmark_config=BenchmarkConfig(
                    repetitions=1, warmup_repetitions=0, max_tokens_scale=0.5
                ),
            )
            result = await svc2.run_benchmark("m", LoadConfiguration(context_length=2048), 2048)
            return result

        result = asyncio.run(_go())
        assert result.generation["max_tokens_scale"] == 0.5

    def test_cli_clamps(self):
        from lm_optimizer.cli.main import _benchmark_config_obj

        assert _benchmark_config_obj(3, 2.0).max_tokens_scale == 1.0
        assert _benchmark_config_obj(3, 0.1).max_tokens_scale == 0.25
        assert _benchmark_config_obj(3).max_tokens_scale == 1.0


class TestMemSampling:
    def test_peaks_computed(self, monkeypatch):
        from lm_optimizer.services import benchmark as _bench

        monkeypatch.setattr(
            _bench,
            "_sample_mem",
            lambda: {
                "vram_free_mb": 1000.0,
                "vram_total_mb": 6144.0,
                "ram_free_gb": 8.0,
                "ram_total_gb": 15.4,
            },
        )
        before = _bench._sample_mem()
        peak_vram, peak_ram = _bench._peak_usage(before, before)
        assert peak_vram == round((6144.0 - 1000.0) / 1024, 2)
        assert peak_ram == round(15.4 - 8.0, 2)

    def test_unavailable_gives_none(self):
        from lm_optimizer.services.benchmark import _peak_usage

        assert _peak_usage({}, {}) == (None, None)
        assert _peak_usage(None, None) == (None, None)

    def test_uses_min_free(self):
        from lm_optimizer.services.benchmark import _peak_usage

        before = {"vram_free_mb": 2000.0, "vram_total_mb": 6144.0}
        after = {"vram_free_mb": 500.0, "vram_total_mb": 6144.0}
        peak_vram, _ = _peak_usage(before, after)
        assert peak_vram == round((6144.0 - 500.0) / 1024, 2)


class TestCliDedup:
    def test_single_recommendation_table(self, monkeypatch):
        from rich.console import Console

        from lm_optimizer.cli import main as _cli

        run, _ = _run([_result(ctx=8192, gen=31.0), _result(ctx=16384, gen=24.0)])
        rec = Console(record=True, width=200)
        monkeypatch.setattr(_cli, "console", rec)
        monkeypatch.setattr(_cli, "_generation_for", lambda *a: {})
        monkeypatch.setattr(_cli, "_save_preset_and_report", lambda *a, **k: True)
        _cli._display_optimization_result(run)
        out = rec.export_text()
        assert out.count("FINAL RECOMMENDATION") == 1
        assert "Recommended Configuration" not in out
        assert "Run speed range" in out


class TestStorageDetection:
    def test_deviceid_casing_and_friendly_fallback(self, monkeypatch):
        import json as _json

        from lm_optimizer.services import hostguard as _hg

        payload = _json.dumps(
            [
                {
                    "DeviceID": "0",
                    "FriendlyName": "NVMe X",
                    "MediaType": "SSD",
                    "Size": 512110190592,
                },
                {"FriendlyName": "WDC Y", "MediaType": "HDD", "Size": 500107862016},
            ]
        )

        class _Out:
            returncode = 0
            stdout = payload

        monkeypatch.setattr(_hg.subprocess, "run", lambda *a, **k: _Out())
        monkeypatch.setattr(_hg.platform, "system", lambda: "Windows")
        info = _hg.storage_info()
        assert info["drives"][0]["id"] == "0"
        assert info["drives"][1]["id"] == "WDC Y"
        assert info["drives"][0]["size_gb"] == 476.9
        assert "swap_free_gb" in info and "swap_total_gb" in info

    def test_pagefile_fallback_usage_class(self, monkeypatch):
        import json as _json

        from lm_optimizer.services import hostguard as _hg

        calls = []

        class _Out:
            def __init__(self, rc, out):
                self.returncode = rc
                self.stdout = out

        def fake_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", "")
            text = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
            calls.append(text)
            if "PhysicalDisk" in text:
                return _Out(0, _json.dumps([]))
            if "PageFileSetting" in text:
                return _Out(0, "")
            if "PageFileUsage" in text:
                return _Out(0, _json.dumps({"Name": "D:\\pagefile.sys"}))
            return _Out(1, "")

        monkeypatch.setattr(_hg.subprocess, "run", fake_run)
        monkeypatch.setattr(_hg.platform, "system", lambda: "Windows")
        info = _hg.storage_info()
        assert info["pagefile"] == "D:\\pagefile.sys"
        assert any("PageFileUsage" in c for c in calls)


class TestPerTestSection:
    def test_every_test_listed_with_temp_and_quality(self, tmp_path):
        from lm_optimizer.services.reporting import save_best_report

        run, _ = _run([_result()])
        path = save_best_report(run, out_dir=tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "## Winner: every test explicitly" in text
        assert "| t | instruction | 0.7 |" in text
        assert "10/50" in text
        assert "(6/6)" in text
        assert "Tested in stage: validation" in text

    def test_generation_records_stage_and_quality(self, monkeypatch):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from lm_optimizer.services.optimizer import AdaptiveOptimizer, OptimizationState
        from lm_optimizer.services.search_space import SearchSpace

        bench_result = _result()
        bench_result.status = "passed"
        opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
        opt.client = MagicMock()
        opt.client.ensure_unloaded = AsyncMock(return_value=True)
        opt.benchmark = MagicMock()
        opt.benchmark.run_benchmark = AsyncMock(return_value=bench_result)
        opt.quality = MagicMock()
        opt.quality.evaluate_all = MagicMock(return_value={"t": _qs()})
        opt.quality.aggregate_quality = MagicMock(return_value=_qs())
        opt.quality.passes_threshold = MagicMock(return_value=True)
        opt.quality.config = MagicMock()
        opt.quality.config.minimum_score = 0.9
        run = OptimizationRun(
            model=ModelIdentity(id="m", name="m"),
            hardware=_hw(),
            profile=OptimizationProfile.BALANCED,
        )
        run.profile_weights = {"generation_speed": 1.0}
        opt.state = OptimizationState(
            run=run,
            search_space=SearchSpace(
                context_lengths=[16384],
                gpu_ratios=[1.0],
                flash_attention_options=[False],
                kv_cache_options=[True],
                batch_sizes=[512],
            ),
        )
        opt.state.tested_configs = []
        from lm_optimizer.database import repositories as _repo

        monkeypatch.setattr(_repo.config_repo, "save", lambda c: str(c.id))
        result = asyncio.run(opt._test_config(LoadConfiguration(context_length=16384), 16384))
        assert result.generation["stage"] == "discovery"
        assert result.generation["quality_by_test"]["t"]["checks_total"] == 6


class TestTypical:
    def test_typical_flags_slow_window(self, tmp_path):
        from lm_optimizer.services.reporting import save_best_report

        slow = _result(gen=24.0, score=0.444)
        fast = [_result(gen=30.0, score=0.7), _result(gen=31.0, score=0.75)]
        run, _ = _run([slow, *fast])
        run.best_config_id = slow.id  # stored repeat ran in a slow window
        path = save_best_report(run, out_dir=tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "## Winner typical across identical runs" in text
        assert "Identical measurements of this exact config: 3" in text
        assert "slow window" in text

    def test_typical_representative(self, tmp_path):
        from lm_optimizer.services.reporting import save_best_report

        reps = [_result(gen=30.0, score=0.7), _result(gen=31.0, score=0.75)]
        run, _ = _run(reps)
        path = save_best_report(run, out_dir=tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "representative" in text
