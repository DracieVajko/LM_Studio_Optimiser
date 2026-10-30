"""Unload settle/backoff (services client).

LM Studio accepts an unload before its inventory stops reporting the
instance. A stale entry must not be mistaken for a failed unload on the first
check (observed live 2026-10-24: two 200 OK unloads, then a stale instance).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from lm_optimizer.services.lm_studio import LMStudioClient


def _client(instances_seq):
    c = LMStudioClient.__new__(LMStudioClient)
    c._loaded_models = {}
    c._no_reasoning_models = set()
    seq = iter(instances_seq)

    async def _instances():
        try:
            return next(seq)
        except StopIteration:
            return []

    c.get_loaded_instances = _instances
    c.unload_model = AsyncMock()
    return c


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(
        "lm_optimizer.services.lm_studio.asyncio.sleep", AsyncMock()
    )


def test_stale_instance_settles_on_second_check():
    leftover = [{"model": "m", "instance_id": "i1", "config": {}}]
    c = _client([leftover, leftover, []])
    assert asyncio.run(c.ensure_unloaded("m")) is True
    # First unload + at least one forced-by-instance unload.
    assert c.unload_model.await_count >= 2


def test_persistent_leftover_returns_false_after_three_attempts():
    leftover = [{"model": "m", "instance_id": "i1", "config": {}}]
    c = _client([leftover] * 6)
    assert asyncio.run(c.ensure_unloaded("m")) is False
    # 3 loop attempts + forced per-instance unloads, never a crash.
    assert c.unload_model.await_count >= 3


def test_already_empty_is_fast_and_true():
    c = _client([[]])
    assert asyncio.run(c.ensure_unloaded("m")) is True
    assert c.unload_model.await_count >= 1
