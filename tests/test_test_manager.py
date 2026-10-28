"""Tests for integrated test manager (ordering, VRAM, speed floor, cleanup)."""

from __future__ import annotations

from collections import namedtuple

from lm_optimizer.services.test_manager import TestManager

MockModel = namedtuple("MockModel", ["id", "size_bytes", "architecture"])


def test_order_ascending_and_moe_before_dense():
    models = [
        MockModel(id="dense-large", size_bytes=10_000_000_000, architecture="dense"),
        MockModel(id="moe-medium", size_bytes=5_000_000_000, architecture="qwen3moe"),
        MockModel(id="dense-small", size_bytes=1_000_000_000, architecture="dense"),
    ]
    manager = TestManager()
    result = manager.order_models(models)
    assert result == ["dense-small", "moe-medium", "dense-large"]


def test_speed_floor_triggers_short_tests():
    manager = TestManager(speed_floor=5.0)
    assert manager.should_use_short_tests(3.5) is True
    assert manager.should_use_short_tests(10.0) is False
    assert manager.should_use_short_tests(None) is False


def test_vram_estimate_exists():
    manager = TestManager()
    assert manager.estimate_vram_needed("any-model") > 0
