"""Pytest configuration."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_client():
    """Mock LM Studio client for testing."""
    from unittest.mock import AsyncMock, MagicMock

    client = MagicMock()
    client.list_models = AsyncMock(return_value=[])
    client.load_model = AsyncMock()
    client.unload_model = AsyncMock()
    client.chat_completion = AsyncMock()
    client.capabilities = MagicMock()
    client.capabilities.supports_flash_attention = True
    client.capabilities.supports_kv_cache_offload = True
    client.capabilities.get_supported_load_params.return_value = [
        "context_length",
        "gpu_ratio",
        "flash_attention",
        "offload_kv_cache_to_gpu",
        "eval_batch_size",
    ]
    return client
