"""Finalist rotation tests (spec 7.3: ABC/BCA/CAB)."""
import pytest

from lm_optimizer.services.optimizer import AdaptiveOptimizer


def test_rotate_no_offset_is_identity():
    items = ["a", "b", "c"]
    assert AdaptiveOptimizer.rotate_order(items, 0) == ["a", "b", "c"]


def test_rotate_cycles_through_all_positions():
    items = ["a", "b", "c"]
    rotations = [AdaptiveOptimizer.rotate_order(items, i) for i in range(3)]
    assert rotations == [
        ["a", "b", "c"],
        ["b", "c", "a"],
        ["c", "a", "b"],
    ]


def test_rotate_preserves_every_item():
    items = ["a", "b", "c", "d"]
    for offset in range(10):
        rotated = AdaptiveOptimizer.rotate_order(items, offset)
        assert sorted(rotated) == sorted(items)
        assert len(rotated) == len(items)


def test_rotate_wraps_offset():
    items = ["a", "b", "c"]
    assert AdaptiveOptimizer.rotate_order(items, 3) == ["a", "b", "c"]
    assert AdaptiveOptimizer.rotate_order(items, 4) == ["b", "c", "a"]


def test_rotate_handles_empty_and_single():
    assert AdaptiveOptimizer.rotate_order([], 5) == []
    assert AdaptiveOptimizer.rotate_order(["only"], 7) == ["only"]


def test_rotate_negative_offset_is_identity():
    items = ["a", "b", "c"]
    assert AdaptiveOptimizer.rotate_order(items, -1) == ["a", "b", "c"]


def test_rotation_fairness_across_campaigns():
    """Different run ids must be able to put different finalists first.

    This is the actual point: the coolest slot must not always belong to
    the same candidate.
    """
    items = ["fastest", "second", "third"]
    firsts = set()
    for run_seed in range(200):
        offset = run_seed % len(items)
        rotated = AdaptiveOptimizer.rotate_order(items, offset)
        firsts.add(rotated[0])
    assert firsts == {"fastest", "second", "third"}
