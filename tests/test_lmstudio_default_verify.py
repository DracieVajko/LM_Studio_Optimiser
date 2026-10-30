"""Async verification of saved LM Studio defaults via an echo load."""

from __future__ import annotations

import asyncio

import pytest

from lm_optimizer.api.client import LoadConfig, LoadModelResponse, UnloadModelResponse
from lm_optimizer.domain.models import LoadConfiguration
from lm_optimizer.services.lmstudio_default_verify import verify_defaults


class FakeClient:
    def __init__(self, *, loaded_config=None, success=True, identifier="inst-1", unload_ok=True):
        self.loaded_config = loaded_config
        self.success = success
        self.identifier = identifier
        self.unload_ok = unload_ok
        self.unloaded = []
        self.loaded = []

    async def load_model(self, model_id, load_config):
        self.loaded.append((model_id, load_config.to_api_params()))
        return LoadModelResponse(success=self.success, identifier=self.identifier,
                                loaded_config=self.loaded_config)

    async def unload_model(self, *, identifier=None, model_id=None):
        self.unloaded.append(identifier)
        return UnloadModelResponse(success=self.unload_ok)


def test_verify_reports_match():
    cfg = LoadConfiguration(context_length=1024, flash_attention=True)
    echo = LoadConfig(context_length=1024, flash_attention=True)
    client = FakeClient(loaded_config=echo)
    result = asyncio.run(verify_defaults("m", cfg, client=client))
    assert result["verified"] is True
    assert result["mismatches"] == {}
    assert result["unloaded"] is True
    assert client.unloaded == ["inst-1"]


def test_verify_reports_mismatch():
    echo = LoadConfig(context_length=512, flash_attention=True)
    client = FakeClient(loaded_config=echo)
    result = asyncio.run(verify_defaults("m", LoadConfiguration(context_length=1024), client=client))
    assert result["verified"] is False
    assert "context_length" in result["mismatches"]


def test_verify_unloads_on_load_failure():
    client = FakeClient(success=False, identifier=None)
    result = asyncio.run(verify_defaults("m", LoadConfiguration(context_length=1024), client=client))
    assert result["verified"] is False
    assert result["error"]
    assert result["unloaded"] is False


def test_verify_unloads_when_no_load_config():
    client = FakeClient(loaded_config=None)
    result = asyncio.run(verify_defaults("m", LoadConfiguration(context_length=1024), client=client))
    assert result["verified"] is False
    assert result["error"] == "server did not return load_config"
    assert result["unloaded"] is True


def test_verify_reports_unverified_fields():
    cfg = LoadConfiguration(context_length=1024, gpu_ratio=0.5, rope_freq_base=10000.0)
    echo = LoadConfig(context_length=1024)
    client = FakeClient(loaded_config=echo)
    result = asyncio.run(verify_defaults("m", cfg, client=client))
    assert "gpu_ratio" in result["unverified"]
    assert "rope_freq_base" in result["unverified"]
