"""Warm validation: reuse the loaded winner between validation repetitions."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from lm_optimizer.domain.models import (
    BenchmarkCase,
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
)
from lm_optimizer.services.benchmark import BenchmarkService


# --- _already_loaded_matches -------------------------------------------------

def _svc(instances):
    client = MagicMock()
    client.get_loaded_instances = AsyncMock(return_value=instances)
    client.load_model = AsyncMock()
    client.ensure_unloaded = AsyncMock(return_value=True)
    client.chat_completion = AsyncMock(
        return_value={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }
    )
    return BenchmarkService(client), client


def test_matches_exact_loaded_config():
    cfg = LoadConfiguration(context_length=8192, flash_attention=True)
    svc, _ = _svc([{"model": "m", "instance_id": "i1", "config": cfg.to_api_params()}])
    assert asyncio.run(svc._already_loaded_matches("m", cfg)) is True


def test_rejects_different_context():
    cfg = LoadConfiguration(context_length=8192, flash_attention=True)
    other = LoadConfiguration(context_length=4096, flash_attention=True)
    svc, _ = _svc([{"model": "m", "instance_id": "i1", "config": other.to_api_params()}])
    assert asyncio.run(svc._already_loaded_matches("m", cfg)) is False


def test_state_error_is_not_a_match():
    client = MagicMock()
    client.get_loaded_instances = AsyncMock(side_effect=RuntimeError("down"))
    svc = BenchmarkService(client)
    assert asyncio.run(svc._already_loaded_matches("m", LoadConfiguration())) is False


# --- run_cases warm path -----------------------------------------------------

def _case():
    return BenchmarkCase(
        name="short_instruction", category="instruction", prompt="hi", max_tokens=8
    )


def test_warm_run_skips_load_and_keeps_model_loaded(monkeypatch):
    cfg = LoadConfiguration(context_length=8192, flash_attention=True)
    svc, client = _svc(
        [{"model": "m", "instance_id": "i1", "config": cfg.to_api_params()}]
    )
    svc._channel_load = AsyncMock(return_value=(True, "i1", cfg, "REST", {}, None))

    result = asyncio.run(
        svc.run_cases(
            "m", cfg, 8192, cases=[_case()],
            repetitions=1, warmup_repetitions=0,
            keep_loaded=True, assume_loaded=True,
        )
    )
    assert result.status == "passed"
    svc._channel_load.assert_not_awaited()
    client.ensure_unloaded.assert_not_awaited()


def test_legacy_run_still_unloads(monkeypatch):
    cfg = LoadConfiguration(context_length=8192, flash_attention=True)
    svc, client = _svc([])
    svc._channel_load = AsyncMock(return_value=(True, "i1", cfg, "REST", {}, None))

    result = asyncio.run(
        svc.run_cases(
            "m", cfg, 8192, cases=[_case()],
            repetitions=1, warmup_repetitions=0,
        )
    )
    assert result.status == "passed"
    client.ensure_unloaded.assert_awaited()


# --- final validation wiring ------------------------------------------------

def _run():
    return OptimizationRun(
        model=ModelIdentity(id="m", name="m", context_limit=32768),
        profile=OptimizationProfile.BALANCED,
        validation_repetitions=3,
        validation_repetitions_min=3,
        validation_repetitions_max=3,
    )


def _result():
    return ConfigurationResult(
        config=LoadConfiguration(context_length=8192, flash_attention=True),
        context_length=8192,
        status=ConfigurationStatus.PASSED,
        score=0.9,
        metrics=[
            BenchmarkMetrics(
                test_name="t", category="speed", success=True,
                generation_tok_s=10.0, completion_tokens=64,
            )
        ],
    )


@pytest.fixture(autouse=True)
def _no_repo_writes(monkeypatch):
    from lm_optimizer.services import optimizer as _opt

    monkeypatch.setattr(_opt.run_repo, "save", lambda r: str(r.id))


def test_final_validation_keeps_loaded_between_reps():
    from lm_optimizer.services.optimizer import AdaptiveOptimizer, OptimizationState

    run = _run()
    best = _result()
    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    opt.state = OptimizationState(run=run, search_space=MagicMock())
    opt.state.run.search_space = {}
    opt.state.best_config = best
    opt.state.tested_configs = [best]
    run.configurations = [best]
    run.best_config_id = best.id
    opt.client = MagicMock()
    opt.client.ensure_unloaded = AsyncMock(return_value=True)
    opt._log_event = lambda *a, **k: None
    opt._checkpoint = lambda *a, **k: None
    opt._sync_phase_params = lambda: None
    opt._broadcast = AsyncMock()
    opt._pause_point = AsyncMock()
    opt._tag_phase = lambda *a, **k: None
    opt._median_result = lambda vals: vals[0]
    opt._compute_pareto_frontier = lambda cfgs: []

    calls = []

    async def fake_test(cfg, ctx, **kw):
        calls.append(kw)
        return _result()

    opt._test_config = fake_test
    asyncio.run(opt._phase_final_validation())

    assert len(calls) == 3
    assert all(c["keep_loaded"] is True for c in calls)
    assert [c["assume_loaded"] for c in calls] == [False, True, True]
    # One verified unload after the loop, not one per repetition.
    assert opt.client.ensure_unloaded.await_count == 1
