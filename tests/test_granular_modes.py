"""Granular test-selection modes: CLI wiring + composition (2026-10-20 audit).

Guards wiring that once existed only as dead code: the optimize CLI accepted
--speed-only/--quality-only/--mtp-only/--reasoning-only but never passed them
to the preflight or the runner, so every "granular" run silently executed a
full campaign. These tests pin:
- the preflight validations (exit 2, zero backend calls),
- the composition of granular modes into skip_phases / speed_context /
  include_categories,
- the pure frontier helper for quality-only / reasoning-only runs,
- the reasoning-only suite semantics (only reasoning tests, 5 levels).
"""
from unittest.mock import AsyncMock, MagicMock

import pytest


def _fake_client():
    fake = MagicMock()
    fake.connect = AsyncMock()
    fake.close = AsyncMock()
    fake.list_models = AsyncMock(return_value=[])
    fake.get_model = AsyncMock(return_value=None)
    return fake


# --- CLI wiring: validations must fire from the optimize command itself ---

@pytest.mark.parametrize(
    "args, needle",
    [
        (["optimize", "some-model", "--speed-only", "--yes"], "requires --context"),
        (["optimize", "some-model", "--mtp-only", "--yes"], "requires --context"),
        (
            ["optimize", "some-model", "--quality-only", "--reasoning-only",
             "--yes"],
            "Cannot use --reasoning-only",
        ),
        (
            ["optimize", "some-model", "--context", "8192", "--yes"],
            "--context only valid",
        ),
        (
            ["optimize", "some-model", "--speed-only", "--mtp-only",
             "--context", "4096", "--yes"],
            "Cannot use --mtp-only",
        ),
    ],
)
def test_granular_validation_exits_2_without_connect(monkeypatch, args, needle):
    from typer.testing import CliRunner

    from lm_optimizer.cli import main as _cli

    fake = _fake_client()
    mk = MagicMock(return_value=fake)
    monkeypatch.setattr(_cli, "get_client", mk)
    result = CliRunner().invoke(_cli.app, args)
    assert result.exit_code == 2, result.output
    assert needle in result.output
    mk.assert_not_called()
    fake.connect.assert_not_called()


# --- Composition: granular modes derive the skip set and context pin ---

def _pf(**kw):
    from lm_optimizer.cli import main as _cli

    base = dict(
        model="m",
        profile="balanced",
        style="chat",
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


def test_speed_only_composes_skips_and_context():
    eff = _pf(speed_only=True, context=8192)
    assert {"quality", "recovery", "validation", "context"} <= set(
        eff["skip_phases"]
    )
    assert "speed" not in eff["skip_phases"]
    assert eff["speed_context"] == 8192
    assert eff["speed_only"] is True


def test_quality_only_composes_skips():
    eff = _pf(quality_only=True)
    assert {"speed", "recovery", "validation", "context"} <= set(
        eff["skip_phases"]
    )
    assert "quality" not in eff["skip_phases"]


def test_reasoning_only_composes_skips_and_categories():
    eff = _pf(reasoning_sweep=True)
    assert {"speed", "recovery", "validation", "context"} <= set(
        eff["skip_phases"]
    )
    assert eff["include_categories"] == {"reasoning"}


def test_mtp_only_composes_skips_but_keeps_speed():
    eff = _pf(mtp_test=True, context=16384)
    assert {"quality", "recovery", "validation", "context"} <= set(
        eff["skip_phases"]
    )
    assert "speed" not in eff["skip_phases"]
    assert eff["speed_context"] == 16384


def test_user_skip_phases_merge_with_granular():
    eff = _pf(quality_only=True, skip_phases="recovery")
    assert {"speed", "recovery", "validation", "context"} <= set(
        eff["skip_phases"]
    )


# --- Pure helper: frontier for quality-only / reasoning-only runs ---

def test_granular_quality_frontier_variants():
    from lm_optimizer.services.optimizer import granular_quality_frontier

    sentinel = object()
    adv = {"quality_only": True}
    f = granular_quality_frontier({"speed", "quality"}, adv, sentinel)
    assert f["finalists"] == [sentinel]
    assert f["winner"] is None
    # Speed skipped but not a granular quality mode -> stays empty.
    assert granular_quality_frontier({"speed", "quality"}, {}, sentinel)[
        "finalists"
    ] == []
    # Speed phase active -> the helper reports an empty frontier (the real
    # one comes from _phase_speed).
    assert granular_quality_frontier(set(), adv, sentinel)["finalists"] == []
    # No baseline available -> empty finalists (honest).
    assert granular_quality_frontier({"speed", "quality"}, adv, None)[
        "finalists"
    ] == []
    # Reasoning-only gets the same baseline treatment.
    f2 = granular_quality_frontier(
        {"speed", "quality"}, {"reasoning_sweep": True}, sentinel
    )
    assert f2["finalists"] == [sentinel]


# --- Suite semantics: reasoning-only battery ---

def test_reasoning_only_suite_is_levels_only():
    from lm_optimizer.benchmark.suite import BENCHMARK_SUITE
    from lm_optimizer.benchmark.suite import create_benchmark_cases

    reasoning_names = [
        bp.name for bp in BENCHMARK_SUITE if bp.category == "reasoning"
    ]
    cases = create_benchmark_cases(4096, reasoning_sweep=True)
    assert cases
    assert all(c.category == "reasoning" for c in cases)
    levels = {"off", "low", "medium", "high", "xhigh"}
    assert {c.name.rsplit("_", 1)[-1] for c in cases} == levels
    for base in reasoning_names:
        for lvl in levels:
            assert any(c.name == f"{base}_{lvl}" for c in cases)


def test_reasoning_sweep_overrides_include_categories():
    from lm_optimizer.benchmark.suite import create_benchmark_cases

    cases = create_benchmark_cases(
        4096, include_categories={"instruction"}, reasoning_sweep=True
    )
    assert cases
    assert all(c.category == "reasoning" for c in cases)


def test_normal_suite_keeps_all_categories():
    from lm_optimizer.benchmark.suite import create_benchmark_cases

    cats = {c.category for c in create_benchmark_cases(4096)}
    assert {"instruction", "reasoning", "context", "coding", "format"} <= cats


# --- Baseline fallback: granular quality modes on a cold server ---

@pytest.mark.asyncio
async def test_granular_baseline_fallback_keeps_quality_candidates():
    """Live bug (2026-10-20, gpt-oss reasoning step): a granular run with no
    loaded model captured no baseline -> empty quality frontier (n=0) ->
    Run FAILED after ~1 s with rc=0 masking it. The fallback must keep a
    candidate alive for granular modes, and stay None for plain campaigns."""
    from types import SimpleNamespace

    from lm_optimizer.services.optimizer import AdaptiveOptimizer

    svc = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    svc.client = _fake_client()  # get_model -> None: cold server
    svc.state = SimpleNamespace(speed_context=None, baseline_result=None)
    model = SimpleNamespace(id="m", context_limit=262144)

    granular = {"speed", "recovery", "validation", "context"}
    cfg = await svc._granular_baseline_or_none(model, None, granular)
    assert cfg is not None, "quality-only must always get a baseline candidate"
    assert cfg.flash_attention is True
    assert cfg.offload_kv_cache_to_gpu is True
    assert cfg.context_length == 2048  # S0 default anchor context

    # Plain campaign: speed plan seeds itself -> None stays None.
    assert await svc._granular_baseline_or_none(model, None, set()) is None
    # Explicit baseline always wins.
    sentinel = SimpleNamespace()
    assert (
        await svc._granular_baseline_or_none(model, sentinel, granular) is sentinel
    )
