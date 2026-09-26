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


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Route ALL database access to a temporary SQLite DB (§16).

    Repositories resolve `db_manager` from their module namespace at call
    time, so swapping the singleton redirects every save/load. Production
    `data/optimizer.db` is never touched by tests. Checkpoints for fake
    run ids are redirected to tmp as well.
    """
    from lm_optimizer.config import config as _cfg
    from lm_optimizer.database import manager as _manager
    from lm_optimizer.database import repositories as _repos

    test_db = _manager.DatabaseManager(db_path=tmp_path / "test.db")
    monkeypatch.setattr(_manager, "db_manager", test_db)
    monkeypatch.setattr(_repos, "db_manager", test_db)
    monkeypatch.setattr(_cfg.storage, "results_dir", tmp_path / "results")
    yield test_db


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
