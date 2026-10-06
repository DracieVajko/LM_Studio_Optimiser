"""Context needle probe service (Task 1: context-sweep plan).

TDD: probe_context loads the best config at ctx N, runs one short
benchmark + one 90%-fill needle recall; ok requires load ok AND
tok_s >= min_speed AND recall >= min_recall. One load per probe,
unload in finally.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from lm_optimizer.domain.models import LoadConfiguration
from lm_optimizer.services.context_probe import (
    NEEDLE_FACTS,
    build_filler,
    probe_context,
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
