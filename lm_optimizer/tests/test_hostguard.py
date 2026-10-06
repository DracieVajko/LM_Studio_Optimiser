"""Exclusive-access guard tests (single-model invariant, Task 1)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from lm_optimizer.services.hostguard import HostBusyTimeout, ensure_exclusive_access


def _inst(model="foreign-model", iid="inst-1"):
    return {"model": model, "instance_id": iid}


@pytest.fixture
def mock_client_foreign_then_gone():
    client = MagicMock()
    client.get_loaded_instances = AsyncMock(side_effect=[[_inst()], []])
    client.ensure_unloaded = AsyncMock(return_value=True)
    client.unload_all = AsyncMock(return_value={})
    return client


@pytest.fixture
def mock_client_foreign_forever():
    client = MagicMock()
    client.get_loaded_instances = AsyncMock(return_value=[_inst()])
    client.ensure_unloaded = AsyncMock(return_value=False)
    client.unload_all = AsyncMock(return_value={})
    return client


async def test_exclusive_access_waits_then_proceeds(mock_client_foreign_then_gone):
    out = await ensure_exclusive_access(mock_client_foreign_then_gone, "test", wait_interval_s=0, max_waits=3)
    assert out["free"] is True and out["attempts"] >= 1


async def test_exclusive_access_raises_after_retries(mock_client_foreign_forever):
    with pytest.raises(HostBusyTimeout):
        await ensure_exclusive_access(mock_client_foreign_forever, "test", wait_interval_s=0, max_waits=2)


async def test_exclusive_access_immediate_when_empty():
    client = MagicMock()
    client.get_loaded_instances = AsyncMock(return_value=[])
    client.ensure_unloaded = AsyncMock(return_value=True)
    out = await ensure_exclusive_access(client, "test", wait_interval_s=0, max_waits=3)
    assert out == {"free": True, "waited_s": out["waited_s"], "attempts": 1}
    assert out["waited_s"] >= 0.0
    client.ensure_unloaded.assert_not_called()
    client.unload_all.assert_not_called()


async def test_exclusive_access_no_sleep_through_on_success(
    mock_client_foreign_then_gone, monkeypatch
):
    calls = []
    real_sleep = asyncio.sleep

    async def _counting_sleep(delay, *a, **k):
        calls.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _counting_sleep)
    out = await ensure_exclusive_access(
        mock_client_foreign_then_gone, "test", wait_interval_s=60, max_waits=3
    )
    assert out["free"] is True
    assert calls == []


async def test_exclusive_access_blocker_on_timeout(mock_client_foreign_forever):
    with pytest.raises(HostBusyTimeout) as exc:
        await ensure_exclusive_access(mock_client_foreign_forever, "test", wait_interval_s=0, max_waits=1)
    assert "foreign-model" in exc.value.blocker


async def test_exclusive_access_opt_out_double_returns_free():
    client = MagicMock()
    client._skip_exclusive_check = True
    out = await ensure_exclusive_access(client, "test", wait_interval_s=0, max_waits=1)
    assert out["free"] is True and out["attempts"] == 0


async def test_exclusive_access_missing_listing_raises():
    client = MagicMock()  # get_loaded_instances auto-attr is non-async
    with pytest.raises(HostBusyTimeout) as exc:
        await ensure_exclusive_access(client, "test", wait_interval_s=0, max_waits=1)
    assert "unverifiable" in exc.value.blocker[0]


async def test_exclusive_access_non_list_state_raises():
    client = MagicMock()
    client.get_loaded_instances = AsyncMock(return_value={"not": "a-list"})
    with pytest.raises(HostBusyTimeout) as exc:
        await ensure_exclusive_access(client, "test", wait_interval_s=0, max_waits=1)
    assert "unverifiable" in exc.value.blocker[0]
