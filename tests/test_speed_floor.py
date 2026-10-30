"""Speed-floor behaviour (spec §3, §10).

Pins the user-visible contract:
- default floor is 2.0 generated tok/s and is enforced;
- the floor can be overridden or explicitly disabled;
- prompt tok/s and parallel throughput never satisfy it;
- a below-floor candidate cannot become the recommended winner when the
  floor is enforced, but stays visible as a diagnostic; the run ends in an
  explicit no-winner state, never a fabricated passing winner;
- failed / empty / zero-token samples are not 0 tok/s contenders;
- a low anchor (S0) does not abort the search.
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
    RunStatus,
)


def _metrics(gen: float, prompt: float = 100.0, success: bool = True):
    return [
        BenchmarkMetrics(
            test_name="short_instruction",
            category="speed",
            success=success,
            prompt_tok_s=prompt,
            generation_tok_s=gen,
            completion_tokens=64 if success else 0,
        )
    ]


def _best(gen: float, *, prompt: float = 100.0, status=ConfigurationStatus.PASSED):
    return ConfigurationResult(
        config=LoadConfiguration(context_length=4096, flash_attention=True),
        context_length=4096,
        status=status,
        score=0.9 if status == ConfigurationStatus.PASSED else None,
        metrics=_metrics(gen, prompt=prompt, success=status == ConfigurationStatus.PASSED),
    )


def _optimizer(best, run):
    from lm_optimizer.services.optimizer import AdaptiveOptimizer, OptimizationState

    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    opt.state = OptimizationState(run=run, search_space=MagicMock())
    opt.state.best_config = best
    opt.state.tested_configs = [best]
    run.configurations = [best]
    opt._log_event = lambda *a, **k: None
    opt._checkpoint = lambda *a, **k: None
    opt._sync_phase_params = lambda: None
    opt._broadcast = AsyncMock()
    return opt


def _run(**kw):
    run = OptimizationRun(
        model=ModelIdentity(id="m", name="m", context_limit=32768),
        hardware=HardwareInfo(os="w", cpu_name="c", cpu_cores_physical=1,
                              cpu_cores_logical=1, total_ram_gb=32, gpu_count=0),
        profile=OptimizationProfile.BALANCED,
    )
    for key, value in kw.items():
        setattr(run, key, value)
    return run


@pytest.fixture(autouse=True)
def _no_repo_writes(monkeypatch):
    from lm_optimizer.services import optimizer as _opt

    monkeypatch.setattr(_opt.run_repo, "save", lambda r: str(r.id))


# --- defaults and overrides -------------------------------------------------

def test_default_floor_is_two_tok_s_and_enforced():
    run = _run()
    assert run.min_generation_tok_s == 2.0
    assert run.enforce_min_generation_speed is True


def test_api_schema_defaults_and_override():
    from lm_optimizer.api.schemas import AdvancedSettingsSchema, OptimizationRequest

    d = AdvancedSettingsSchema()
    assert d.min_generation_tok_s == 2.0
    assert d.enforce_min_generation_speed is True

    over = AdvancedSettingsSchema(
        min_generation_tok_s=4.0, enforce_min_generation_speed=False
    )
    assert over.min_generation_tok_s == 4.0
    assert over.enforce_min_generation_speed is False

    req = OptimizationRequest(model_id="m", min_generation_tok_s=3.0)
    assert req.min_generation_tok_s == 3.0

    with pytest.raises(Exception):
        AdvancedSettingsSchema(min_generation_tok_s=-1.0)


def test_cli_preflight_default_and_override():
    from lm_optimizer.cli import main as _cli

    def _pf(**kw):
        base = dict(
            model="m",
            profile="balanced",
            style="balanced",
            quality_threshold=0.97,
            repetitions=3,
            min_context=2048,
            max_context=32768,
            warmup_runs=1,
            report_sort="run",
            skip_phases=None,
            physical_batches=None,
            enable_rope=None,
            enable_cpu_moe=None,
            enable_speculative=None,
            yes=True,
            non_interactive=True,
            dry_run=False,
        )
        base.update(kw)
        return _cli._preflight_optimize(**base)

    assert _pf()["min_generation_tok_s"] == 2.0
    assert _pf()["enforce_min_generation_speed"] is True
    eff = _pf(min_generation_tok_s=5.0, enforce_min_generation_speed=False)
    assert eff["min_generation_tok_s"] == 5.0
    assert eff["enforce_min_generation_speed"] is False


def test_cli_rejects_negative_floor_without_connecting(monkeypatch):
    from typer.testing import CliRunner

    from lm_optimizer.cli import main as _cli

    mk = MagicMock()
    monkeypatch.setattr(_cli, "get_client", mk)
    result = CliRunner().invoke(
        _cli.app,
        ["optimize", "m", "--min-generation-tok-s=-1", "--yes"],
    )
    assert result.exit_code == 2
    assert "min-generation-tok-s" in result.output
    mk.assert_not_called()


def test_zero_floor_is_not_the_same_as_disabled():
    # 0.0 is a real floor: everything passes. Disabled keeps the score-only
    # behaviour but records the floor as informational.
    run = _run(min_generation_tok_s=0.0, enforce_min_generation_speed=True)
    best = _best(0.1)
    opt = _optimizer(best, run)
    asyncio.run(opt._finalize())
    assert run.search_space["speed_floor"]["met"] is True
    assert run.best_config_id == best.id
    assert run.no_winner_reason is None


# --- enforcement ------------------------------------------------------------

def test_below_floor_cannot_win_when_enforced_but_stays_diagnostic():
    run = _run(min_generation_tok_s=2.0, enforce_min_generation_speed=True)
    best = _best(1.0)
    opt = _optimizer(best, run)
    asyncio.run(opt._finalize())

    assert run.best_config_id is None
    assert opt.state.best_config is None
    assert run.status == RunStatus.COMPLETED_NO_WINNER
    assert run.no_winner_reason == "NO_CONFIG_MEETS_SPEED_FLOOR"

    floor = run.search_space["speed_floor"]
    assert floor["met"] is False
    assert floor["enforced"] is True
    assert floor["value"] == 2.0
    assert floor["fastest_candidate_id"] == str(best.id)
    assert floor["fastest_tok_s"] == pytest.approx(1.0)


def test_below_floor_wins_when_floor_not_enforced():
    run = _run(min_generation_tok_s=2.0, enforce_min_generation_speed=False)
    best = _best(1.0)
    opt = _optimizer(best, run)
    asyncio.run(opt._finalize())

    assert run.best_config_id == best.id
    assert opt.state.best_config is best
    assert run.status == RunStatus.SUCCESS
    assert run.no_winner_reason is None
    assert run.search_space["speed_floor"]["met"] is False
    assert run.search_space["speed_floor"]["enforced"] is False


def test_at_floor_wins_when_enforced():
    run = _run(min_generation_tok_s=2.0, enforce_min_generation_speed=True)
    best = _best(2.0)
    opt = _optimizer(best, run)
    asyncio.run(opt._finalize())
    assert run.best_config_id == best.id
    assert run.search_space["speed_floor"]["met"] is True


def test_prompt_speed_never_satisfies_the_generation_floor():
    run = _run(min_generation_tok_s=2.0, enforce_min_generation_speed=True)
    # Huge prompt-processing throughput, slow generation.
    best = _best(0.5, prompt=5000.0)
    opt = _optimizer(best, run)
    asyncio.run(opt._finalize())
    assert run.best_config_id is None
    assert run.no_winner_reason == "NO_CONFIG_MEETS_SPEED_FLOOR"


# --- invalid samples --------------------------------------------------------

def test_failed_and_empty_samples_are_not_zero_speed_contenders():
    failed = _best(0.0, status=ConfigurationStatus.FAILED)
    assert failed.get_avg_generation_tok_s() == 0.0
    # A failed config is cleared before the floor check and can never win.
    run = _run(min_generation_tok_s=0.0, enforce_min_generation_speed=True)
    opt = _optimizer(failed, run)
    asyncio.run(opt._finalize())
    assert run.best_config_id is None


def test_zero_token_success_sample_is_zero_speed():
    result = ConfigurationResult(
        config=LoadConfiguration(context_length=4096),
        context_length=4096,
        status=ConfigurationStatus.PASSED,
        metrics=[
            BenchmarkMetrics(
                test_name="t", category="speed", success=True,
                generation_tok_s=0.0, completion_tokens=0,
            )
        ],
    )
    assert result.get_avg_generation_tok_s() == 0.0


# --- low anchor does not abort ---------------------------------------------

def test_low_s0_does_not_cancel_the_search():
    from lm_optimizer.services.optimizer import AdaptiveOptimizer

    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    run = _run()
    from lm_optimizer.services.optimizer import OptimizationState

    opt.state = OptimizationState(run=run, search_space=MagicMock())
    opt.state.should_cancel = False

    slow = _best(0.2)
    budget = opt._classify_budget(slow)

    assert opt.state.should_cancel is False
    assert budget  # a budget class is still produced
    assert opt.state.baseline_result is slow


def test_speed_plan_keeps_high_impact_probes_after_slow_anchor():
    from lm_optimizer.services.speed_search import SpeedCaps, build_speed_plan

    anchor = LoadConfiguration(context_length=2048, flash_attention=True)
    caps = SpeedCaps(
        rest_keys=frozenset(
            {"context_length", "flash_attention", "offload_kv_cache_to_gpu"}
        )
    )
    plan, _notes = build_speed_plan(anchor, 2048, caps)
    assert len(plan) > 1  # S0 plus improvement probes
