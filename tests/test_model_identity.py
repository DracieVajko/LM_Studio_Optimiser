"""Model identity mapping: LM Studio load accepts the KEY, not the variant id.

Regression for the 2026-10-24 failure where ``selected_variant`` was used as
``ModelIdentity.id`` and every variant model failed with HTTP 404
"not found in downloaded models" during load.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from lm_optimizer.services.lm_studio import LMStudioClient


API_MODELS = {
    "models": [
        {
            "key": "qwen/qwen3-coder-30b",
            "selected_variant": "qwen/qwen3-coder-30b@q4_k_m",
            "display_name": "Qwen3 Coder 30B",
            "architecture": "qwen3",
            "quantization": {"name": "Q4_K_M"},
            "max_context_length": 262144,
        },
        {
            "key": "swift-qwen3.8-27b@q3_k_m",
            "selected_variant": "swift-qwen3.8-27b@q3_k_m",
            "display_name": "Swift Qwen3.8 27B",
            "quantization": {"name": "Q3_K_M"},
            "max_context_length": 262144,
        },
        {
            "key": "qwen3.8-9b-distill",
            "selected_variant": None,
            "display_name": "Qwen3.8 9B",
            "quantization": {"name": "Q4_K_M"},
            "max_context_length": 32768,
        },
    ]
}


def _client():
    client = LMStudioClient.__new__(LMStudioClient)
    client._models_cache = []
    client._loaded_models = {}
    response = MagicMock()
    response.json.return_value = API_MODELS
    client._request_with_retry = AsyncMock(return_value=response)
    return client


def test_variant_models_use_the_key_as_load_id():
    models = asyncio.run(_client().list_models(force_refresh=True))
    by_key = {m.key: m for m in models}

    coder = by_key["qwen/qwen3-coder-30b"]
    # The load endpoint needs the key; the variant stays separate metadata.
    assert coder.id == "qwen/qwen3-coder-30b"
    assert coder.selected_variant == "qwen/qwen3-coder-30b@q4_k_m"
    assert coder.quantization == "Q4_K_M"

    swift = by_key["swift-qwen3.8-27b@q3_k_m"]
    assert swift.id == "swift-qwen3.8-27b@q3_k_m"
    assert swift.selected_variant == "swift-qwen3.8-27b@q3_k_m"

    plain = by_key["qwen3.8-9b-distill"]
    assert plain.id == "qwen3.8-9b-distill"
    assert plain.selected_variant == "qwen3.8-9b-distill"


@pytest.mark.parametrize(
    "requested, expected_key",
    [
        ("qwen/qwen3-coder-30b", "qwen/qwen3-coder-30b"),
        ("qwen/qwen3-coder-30b@q4_k_m", "qwen/qwen3-coder-30b"),
        ("qwen3.8-9b-distill", "qwen3.8-9b-distill"),
        ("swift-qwen3.8-27b@q3_k_m", "swift-qwen3.8-27b@q3_k_m"),
    ],
)
def test_get_model_resolves_key_and_variant(requested, expected_key):
    model = asyncio.run(_client().get_model(requested))
    assert model is not None
    assert model.key == expected_key
    # Whatever the user typed, the load identifier is the key.
    assert model.id == expected_key


def test_get_model_returns_none_for_unknown():
    assert asyncio.run(_client().get_model("nope/nope")) is None


def test_force_refresh_clears_cache():
    client = _client()
    client._models_cache = [MagicMock()]
    asyncio.run(client.list_models(force_refresh=True))
    assert client._request_with_retry.await_count == 1
    assert len(client._models_cache) == len(API_MODELS["models"])
