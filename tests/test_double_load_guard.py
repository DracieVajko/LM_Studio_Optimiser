"""Double-load guard tests (live bug: two instances of one model).

Observed 2026-10-22: under heavy host load the unload of the previous
benchmark instance was slow, smoke_test ignored the failed unload and the
next load created a SECOND instance of the same LLM. _channel_load now
refuses to load while any instance of the model is still alive.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from lm_optimizer.domain.models import LoadConfiguration
from lm_optimizer.services.benchmark import BenchmarkService


def _service(unload_ok: bool):
    client = MagicMock()
    client.ensure_unloaded = AsyncMock(return_value=unload_ok)
    client.load_model = AsyncMock()
    svc = BenchmarkService(client)
    return svc, client


@pytest.mark.asyncio
async def test_load_refused_when_model_still_loaded():
    svc, client = _service(unload_ok=False)
    ok, ident, loaded, channel, applied, err = await svc._channel_load(
        "model/x", LoadConfiguration(context_length=4096)
    )
    assert ok is False
    assert "refusing" in err
    assert channel == "precheck"
    # The load must never have been attempted.
    client.load_model.assert_not_called()
    assert svc.last_unverified_unload == "model/x"


@pytest.mark.asyncio
async def test_load_proceeds_when_unload_verified():
    svc, client = _service(unload_ok=True)
    result = MagicMock()
    result.success = True
    result.identifier = "model/x"
    result.loaded_config = None
    client.load_model = AsyncMock(return_value=result)

    ok, ident, loaded, channel, applied, err = await svc._channel_load(
        "model/x", LoadConfiguration(context_length=4096)
    )
    assert ok is True
    assert channel == "REST"
    client.ensure_unloaded.assert_awaited()
    client.load_model.assert_awaited()
    assert svc.last_unverified_unload is None


@pytest.mark.asyncio
async def test_precheck_exception_does_not_block_the_load():
    svc, client = _service(unload_ok=True)
    client.ensure_unloaded = AsyncMock(side_effect=RuntimeError("boom"))
    result = MagicMock()
    result.success = True
    result.identifier = "model/x"
    result.loaded_config = None
    client.load_model = AsyncMock(return_value=result)

    ok, _ident, _loaded, _channel, _applied, err = await svc._channel_load(
        "model/x", LoadConfiguration(context_length=4096)
    )
    assert ok is True, err


@pytest.mark.asyncio
async def test_smoke_test_records_unverified_unload():
    svc, client = _service(unload_ok=False)
    ok, tok, err = await svc.smoke_test("model/x")
    assert ok is False
    assert svc.last_unverified_unload == "model/x"
