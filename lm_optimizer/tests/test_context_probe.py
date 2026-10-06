"""Context needle probe service (Task 1: context-sweep plan).

TDD: probe_context loads the best config at ctx N, runs one short
benchmark + one 90%-fill needle recall; ok requires load ok AND
tok_s >= min_speed AND recall >= min_recall. One load per probe,
unload in finally.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from lm_optimizer.domain.models import (
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
    OptimizationRun,
)
from lm_optimizer.services.context_probe import (
    NEEDLE_FACTS,
    NoBestConfigError,
    build_filler,
    probe_context,
    resolve_best_for_sweep,
    run_needle,
)


def cfg() -> LoadConfiguration:
    return LoadConfiguration(context_length=2048, flash_attention=True)


def _load_ok():
    load = MagicMock()
    load.success = True
    load.loaded_config = None
    return load


def _resp(text: str, tok_s: float, prompt_tokens: int = 20,
          completion_tokens: int = 10) -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "_stats": {
            "tokens_per_second": tok_s,
            "time_to_first_token_seconds": 0.05,
        },
    }


def _client_scripted(*responses: dict):
    c = MagicMock()
    c.load_model = AsyncMock(return_value=_load_ok())
    c.ensure_unloaded = AsyncMock(return_value=True)
    c.chat_completion = AsyncMock(side_effect=list(responses))
    return c


def test_probe_stops_below_speed_floor():
    slow = _client_scripted(_resp("short answer here.", tok_s=0.3))
    out = asyncio.run(
        probe_context(slow, "m", cfg(), 8192, min_speed=1.0, min_recall=0.8)
    )
    assert out["ok"] is False and "speed" in out["error"]


def test_probe_passes_when_fast_and_full_recall():
    all_facts = "\n".join(NEEDLE_FACTS)
    c = _client_scripted(
        _resp("hash tables map keys fast.", tok_s=50.0),
        _resp(all_facts, tok_s=40.0, prompt_tokens=7000),
    )
    out = asyncio.run(
        probe_context(c, "m", cfg(), 8192, min_speed=1.0, min_recall=0.8)
    )
    assert out["ok"] is True, out
    assert out["ctx"] == 8192
    assert out["tok_s"] >= 1.0
    assert out["recall"] == 1.0
    assert out["error"] == ""
    assert c.load_model.await_count == 1
    c.ensure_unloaded.assert_called_with("m")


def test_probe_fails_on_low_recall():
    c = _client_scripted(
        _resp("hash tables map keys fast.", tok_s=50.0),
        _resp("I do not know any facts.", tok_s=40.0, prompt_tokens=7000),
    )
    out = asyncio.run(
        probe_context(c, "m", cfg(), 8192, min_speed=1.0, min_recall=0.8)
    )
    assert out["ok"] is False and "recall" in out["error"]


def test_probe_load_failure_is_not_ok():
    c = MagicMock()
    bad = MagicMock()
    bad.success = False
    bad.error = "boom"
    c.load_model = AsyncMock(return_value=bad)
    c.ensure_unloaded = AsyncMock(return_value=True)
    c.chat_completion = AsyncMock()
    out = asyncio.run(
        probe_context(c, "m", cfg(), 8192, min_speed=1.0, min_recall=0.8)
    )
    assert out["ok"] is False and "load" in out["error"]
    c.ensure_unloaded.assert_called_with("m")


def test_probe_loads_best_config_at_ctx():
    all_facts = "\n".join(NEEDLE_FACTS)
    c = _client_scripted(
        _resp("hash tables map keys fast.", tok_s=50.0),
        _resp(all_facts, tok_s=40.0, prompt_tokens=7000),
    )
    base = cfg()
    asyncio.run(probe_context(c, "m", base, 8192, min_speed=1.0, min_recall=0.8))
    sent_cfg = c.load_model.await_args.args[1]
    assert sent_cfg.context_length == 8192
    assert base.context_length == 2048  # caller config untouched


def test_build_filler_plants_all_facts():
    filler = build_filler(4000)
    assert len(filler) >= 4000 - len("\n".join(NEEDLE_FACTS)) - 500
    for fact in NEEDLE_FACTS:
        assert fact in filler


def test_run_needle_recall_hits_over_asked():
    all_facts = "\n".join(NEEDLE_FACTS)
    c = _client_scripted(_resp(all_facts, tok_s=40.0, prompt_tokens=7000))
    out = asyncio.run(run_needle(c, "m", cfg(), 8192, fill_ratio=0.9))
    assert out["asked"] == len(NEEDLE_FACTS) == 5
    assert out["recall"] == 1.0
    assert out["fill_chars"] > 0
    assert out["prompt_tokens"] == 7000


# ---- Task 2: context-sweep command (resolve + registration) ----


class _EmptyRepo:
    def get_by_model(self, model_id, limit):
        return []


def _measured_run(model_id="m", ctx=4096, gen=42.0):
    best_cfg = ConfigurationResult(
        config=LoadConfiguration(context_length=ctx),
        context_length=ctx,
        status=ConfigurationStatus.PASSED,
        metrics=[
            BenchmarkMetrics(
                test_name="t",
                category="c",
                success=True,
                generation_tok_s=gen,
                prompt_tok_s=800.0,
                estimated_ttft_ms=120.0,
            )
        ],
        generation={"style": "balanced"},
    )
    run = OptimizationRun()
    run.configurations = [best_cfg]
    run.best_config_id = best_cfg.id
    return run


class _SeededRepo:
    def __init__(self, run):
        self._run = run

    def get_by_model(self, model_id, limit):
        return [SimpleNamespace(id=str(self._run.id))]

    def get(self, run_id):
        return self._run


async def test_sweep_refuses_without_best_config():
    with pytest.raises(NoBestConfigError, match="ghost-model"):
        resolve_best_for_sweep("ghost-model", repo=_EmptyRepo())


async def test_sweep_resolves_best_from_db():
    out = resolve_best_for_sweep("m", repo=_SeededRepo(_measured_run()))
    assert out["context_length"] == 4096
    assert out["load_config"].context_length == 4096
    assert out["gen_tok_s"] == 42.0


async def test_probe_skips_needle_when_requested():
    c = _client_scripted(_resp("hash tables map keys fast.", tok_s=50.0))
    out = await probe_context(
        c, "m", cfg(), 8192, min_speed=1.0, min_recall=0.8, skip_needle=True
    )
    assert out["ok"] is True
    assert out.get("skipped_needle") is True
    assert c.chat_completion.await_count == 1  # speed chat only, no fill chat


def test_context_sweep_command_options():
    import typer

    from lm_optimizer.cli.main import app

    info = typer.main.get_command(app)
    cmd = info.commands["context-sweep"]
    params = {p.name: p for p in cmd.params}
    for name in (
        "model",
        "max_context",
        "min_speed",
        "min_recall",
        "skip_fill_test",
        "output",
    ):
        assert name in params, name
    assert params["min_speed"].default == 1.0
    assert params["min_recall"].default == 0.8


def test_sweep_bounds_clamp_to_model_limit():
    from lm_optimizer.cli.main import _sweep_bounds

    assert _sweep_bounds(4096, 65536, 131072) == (4096, 65536)
    assert _sweep_bounds(8192, 65536, 16384) == (8192, 16384)
    assert _sweep_bounds(32768, 8192, None) == (8192, 8192)


def test_probe_row_marks_skipped_recall():
    from lm_optimizer.cli.main import _format_probe_row

    row = _format_probe_row(
        {
            "ctx": 8192,
            "ok": True,
            "tok_s": 40.0,
            "prompt_tok_s": 900.0,
            "ttft_ms": 120.0,
            "recall": 1.0,
            "skipped_needle": True,
            "error": "",
        }
    )
    assert "8192" in row and "PASS" in row and "skip" in row
