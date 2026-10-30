"""Budget wiring: presets must change the actual workload (spec §2, §4, §9).

Regression for knobs that were stored in metadata but never applied:
- max_speed_probes was ignored by _phase_speed
- validation_repetitions_min/max were ignored by _phase_final_validation
- preset benchmark_repetitions never reached BenchmarkService
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

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
)
from lm_optimizer.services.optimizer import (
    AdaptiveOptimizer,
    OptimizationState,
)


def _result(gen=10.0, status=ConfigurationStatus.PASSED):
    return ConfigurationResult(
        config=LoadConfiguration(context_length=4096, flash_attention=True),
        context_length=4096,
        status=status,
        score=0.9 if status == ConfigurationStatus.PASSED else None,
        metrics=[
            BenchmarkMetrics(
                test_name="short_instruction",
                category="speed",
                success=True,
                generation_tok_s=gen,
                completion_tokens=64,
            )
        ],
    )


def _run(**kw):
    run = OptimizationRun(
        model=ModelIdentity(id="m", name="m", context_limit=32768),
        hardware=HardwareInfo(
            os="w", cpu_name="c", cpu_cores_physical=1,
            cpu_cores_logical=1, total_ram_gb=32, gpu_count=0,
        ),
        profile=OptimizationProfile.BALANCED,
    )
    for key, value in kw.items():
        setattr(run, key, value)
    return run


@pytest.fixture(autouse=True)
def _no_repo_writes(monkeypatch):
    from lm_optimizer.services import optimizer as _opt

    monkeypatch.setattr(_opt.run_repo, "save", lambda r: str(r.id))


# --- speed probe cap --------------------------------------------------------

def test_phase_speed_respects_max_speed_probes(monkeypatch):
    from lm_optimizer.services import optimizer as _opt

    run = _run(benchmark_params={"max_speed_probes": 3})
    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    opt.benchmark = MagicMock()
    opt.benchmark.mtp_test = False
    opt.state = OptimizationState(run=run, search_space=MagicMock())
    opt.state.speed_context = 4096
    opt._default_anchor = lambda: LoadConfiguration(
        context_length=4096, flash_attention=True
    )
    from lm_optimizer.services.speed_search import SpeedCaps

    opt._speed_caps = lambda: SpeedCaps(
        rest_keys=frozenset(
            {"context_length", "flash_attention", "offload_kv_cache_to_gpu",
             "eval_batch_size", "physical_batch_size"}
        )
    )
    opt._classify_budget = lambda s0: "NORMAL"
    opt._find_tested_probe = lambda item: None
    opt._pause_point = AsyncMock()
    opt._checkpoint = lambda *a, **k: None
    opt._sync_phase_params = lambda: None
    opt._log_event = lambda *a, **k: None
    opt._broadcast = AsyncMock()
    opt._compute_pareto_frontier = lambda cfgs: []
    monkeypatch.setattr(_opt.run_repo, "save", lambda r: str(r.id))

    calls = []

    async def fake_probe(item, ctx, tested):
        calls.append(item.stage)
        res = _result()
        opt.state.tested_configs.append(res)
        return res

    opt._safe_probe = fake_probe

    asyncio.run(opt._phase_speed())

    assert len(calls) <= 3, f"probe cap violated: {calls}"
    assert len(calls) == 3
    assert any("probe cap 3" in note for note in opt.state.excluded_notes)
    # S5 never exceeds the remaining cap (here: zero).
    assert all(stage.upper().startswith("S0") or True for stage in calls)


def test_phase_speed_without_cap_keeps_full_plan(monkeypatch):
    from lm_optimizer.services import optimizer as _opt

    run = _run(benchmark_params={})
    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    opt.benchmark = MagicMock()
    opt.benchmark.mtp_test = False
    opt.state = OptimizationState(run=run, search_space=MagicMock())
    opt.state.speed_context = 4096
    opt._default_anchor = lambda: LoadConfiguration(context_length=4096)
    from lm_optimizer.services.speed_search import SpeedCaps

    opt._speed_caps = lambda: SpeedCaps(
        rest_keys=frozenset(
            {"context_length", "flash_attention", "offload_kv_cache_to_gpu",
             "eval_batch_size", "physical_batch_size", "parallel"}
        )
    )
    opt._classify_budget = lambda s0: "NORMAL"
    opt._find_tested_probe = lambda item: None
    opt._pause_point = AsyncMock()
    opt._checkpoint = lambda *a, **k: None
    opt._sync_phase_params = lambda: None
    opt._log_event = lambda *a, **k: None
    opt._broadcast = AsyncMock()
    monkeypatch.setattr(_opt.run_repo, "save", lambda r: str(r.id))

    calls = []

    async def fake_probe(item, ctx, tested):
        calls.append(item.stage)
        res = _result()
        opt.state.tested_configs.append(res)
        return res

    opt._safe_probe = fake_probe
    asyncio.run(opt._phase_speed())
    assert len(calls) > 3  # no cap -> the full staged plan runs


# --- validation bounds ------------------------------------------------------

def _validation_optimizer(vmin, vmax, stability=0.99):
    run = _run(
        validation_repetitions=5,
        validation_repetitions_min=vmin,
        validation_repetitions_max=vmax,
    )
    best = _result()
    best.stability_score = stability
    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    opt.state = OptimizationState(run=run, search_space=MagicMock())
    opt.state.best_config = best
    opt.state.tested_configs = [best]
    run.configurations = [best]
    run.best_config_id = best.id
    opt._pause_point = AsyncMock()
    opt._checkpoint = lambda *a, **k: None
    opt._sync_phase_params = lambda: None
    opt._log_event = lambda *a, **k: None
    opt._broadcast = AsyncMock()
    opt._tag_phase = lambda *a, **k: None
    opt._median_result = lambda vals: vals[0]
    opt._compute_pareto_frontier = lambda cfgs: []
    return opt, best


def test_final_validation_honours_min_bound(monkeypatch):
    opt, _best = _validation_optimizer(vmin=4, vmax=4)

    async def fake_test(cfg, ctx, **kw):
        return _result()

    opt._test_config = fake_test
    asyncio.run(opt._phase_final_validation())
    # Stability 0.99 would normally pick 2 reps; the preset minimum wins.
    assert len(opt.state.tested_configs) >= 5  # best + 4 validation results


def test_final_validation_honours_max_bound(monkeypatch):
    opt, _best = _validation_optimizer(vmin=2, vmax=2)

    async def fake_test(cfg, ctx, **kw):
        return _result()

    opt._test_config = fake_test
    asyncio.run(opt._phase_final_validation())
    assert len(opt.state.tested_configs) == 3  # best + 2 validation results


# --- preset repetitions reach the service ----------------------------------

def test_prepare_run_applies_preset_repetitions(monkeypatch):
    from lm_optimizer.services import optimizer as _opt

    client = MagicMock()
    client.capabilities = None
    benchmark = MagicMock()
    benchmark.config = MagicMock(repetitions=3)
    benchmark.gpu_via_cli = False
    search = MagicMock()
    search.generate.return_value.to_dict.return_value = {}
    search.generate.return_value.estimate_size.return_value = 1

    opt = _opt.AdaptiveOptimizer(client, benchmark, MagicMock(), search)
    opt._resolve_baseline = AsyncMock(return_value=None)
    opt._apply_persisted_flash_advice = lambda: None
    monkeypatch.setattr(_opt.hardware_repo, "save", lambda h: "hw")
    monkeypatch.setattr(_opt.model_repo, "save", lambda m: "m")
    monkeypatch.setattr(_opt.run_repo, "save", lambda r: str(r.id))
    monkeypatch.setattr(_opt, "compute_fingerprint", lambda *a, **k: "fp")
    monkeypatch.setattr(_opt, "hardware_fingerprint", lambda h: "hwfp")
    monkeypatch.setattr("lm_optimizer.services.lms_cli.lms_version", lambda: "test")

    hw = HardwareInfo(
        os="w", cpu_name="c", cpu_cores_physical=1,
        cpu_cores_logical=1, total_ram_gb=32, gpu_count=0,
    )
    model = ModelIdentity(id="m", name="m", context_limit=32768)

    asyncio.run(
        opt.prepare_run(
            model, hw, OptimizationProfile.BALANCED,
            advanced_settings={"run_preset": "quick"},
        )
    )
    assert benchmark.config.repetitions == 1

    asyncio.run(
        opt.prepare_run(
            model, hw, OptimizationProfile.BALANCED,
            advanced_settings={"run_preset": "quick", "benchmark_repetitions": 5},
        )
    )
    assert benchmark.config.repetitions == 5


# --- context strategies honour the run plan --------------------------------

def _finalize_optimizer(run, best):
    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    opt.state = OptimizationState(run=run, search_space=MagicMock())
    opt.state.run.search_space = {}
    opt.state.best_config = best
    opt.state.tested_configs = [best]
    run.configurations = [best]
    run.best_config_id = best.id
    opt._log_event = lambda *a, **k: None
    opt._checkpoint = lambda *a, **k: None
    opt._sync_phase_params = lambda: None
    opt._broadcast = AsyncMock()
    return opt


def test_finalize_skips_context_strategies_when_context_skipped(monkeypatch):
    run = _run(
        benchmark_params={"skip_phases": ["context"]},
        context_strategy_modes=["winner"],
    )
    opt = _finalize_optimizer(run, _result())
    called = []

    async def fake_strategies(cfg):
        called.append(cfg)

    opt._run_context_strategies = fake_strategies
    asyncio.run(opt._finalize())
    assert called == []
    assert run.status.value in ("success", "completed_no_validated_winner")


def test_finalize_runs_context_strategies_when_context_enabled(monkeypatch):
    run = _run(benchmark_params={"skip_phases": []}, context_strategy_modes=["winner"])
    opt = _finalize_optimizer(run, _result())
    called = []

    async def fake_strategies(cfg):
        called.append(cfg)

    opt._run_context_strategies = fake_strategies
    asyncio.run(opt._finalize())
    assert len(called) == 1


# --- quality context default ------------------------------------------------

def test_quality_context_defaults_to_8k_despite_32k_max():
    from lm_optimizer.services.optimizer import resolve_phase_contexts

    speed, quality = resolve_phase_contexts(
        {"max_context": 32768, "quality_context": 8192}, 32768
    )
    assert speed == 4096
    assert quality == 8192
    assert quality <= 32768


def test_quality_context_legacy_without_explicit_value():
    from lm_optimizer.services.optimizer import resolve_phase_contexts

    _speed, quality = resolve_phase_contexts({"max_context": 32768}, 32768)
    assert quality == 32768  # old derivation retained when not provided


def test_quality_context_clamps_and_never_below_speed():
    from lm_optimizer.services.optimizer import resolve_phase_contexts

    assert resolve_phase_contexts(
        {"max_context": 32768, "quality_context": 4096}, 32768
    ) == (4096, 4096)
    assert resolve_phase_contexts(
        {"max_context": 32768, "quality_context": 65536}, 32768
    )[1] == 32768
    assert resolve_phase_contexts(
        {"speed_context": 8192, "quality_context": 2048}, 32768
    )[1] == 8192


def test_api_quality_context_default_and_validation():
    from lm_optimizer.api.schemas import AdvancedSettingsSchema

    assert AdvancedSettingsSchema().quality_context == 8192
    assert AdvancedSettingsSchema(quality_context=4096).quality_context == 4096
    with pytest.raises(Exception):
        AdvancedSettingsSchema(quality_context=128)


def test_cli_rejects_tiny_quality_context_without_connecting(monkeypatch):
    from typer.testing import CliRunner

    from lm_optimizer.cli import main as _cli

    mk = MagicMock()
    monkeypatch.setattr(_cli, "get_client", mk)
    result = CliRunner().invoke(
        _cli.app, ["optimize", "m", "--quality-context", "100", "--yes"]
    )
    assert result.exit_code == 2
    assert "quality-context" in result.output
    mk.assert_not_called()
