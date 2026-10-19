"""Tests for benchmark filtering and reasoning sweep."""

import pytest

from lm_optimizer.benchmark.suite import (
    create_benchmark_cases,
    get_benchmark_suite,
    BenchmarkPrompt,
    BenchmarkCase,
)


def test_get_benchmark_suite_returns_all():
    """Test that get_benchmark_suite returns all 5 prompts."""
    suite = get_benchmark_suite()
    assert len(suite) == 5
    names = {p.name for p in suite}
    assert names == {
        "short_instruction",
        "medium_reasoning",
        "long_context",
        "coding_task",
        "structured_output",
    }


def test_create_benchmark_cases_filters_by_category():
    """Test that create_benchmark_cases filters by category."""
    cases = create_benchmark_cases(4096, include_categories={"coding"})
    assert len(cases) == 1
    assert cases[0].name == "coding_task"


def test_create_benchmark_cases_multiple_categories():
    """Test filtering with multiple categories."""
    cases = create_benchmark_cases(4096, include_categories={"coding", "format"})
    assert len(cases) == 2
    names = {c.name for c in cases}
    assert names == {"coding_task", "structured_output"}


def test_create_benchmark_cases_empty_category_returns_all():
    """Test that empty category set returns all tests."""
    cases = create_benchmark_cases(4096, include_categories=set())
    assert len(cases) == 5


def test_create_benchmark_cases_none_category_returns_all():
    """Test that None category returns all tests (backward compat)."""
    cases = create_benchmark_cases(4096, include_categories=None)
    assert len(cases) == 5


def test_create_benchmark_cases_invalid_category_returns_empty():
    """Test that invalid category returns empty list."""
    cases = create_benchmark_cases(4096, include_categories={"nonexistent"})
    assert len(cases) == 0


def test_reasoning_sweep_creates_5_variants_per_base():
    """Test that reasoning_sweep creates 5 variants per base config."""
    # Only include reasoning category for this test
    cases = create_benchmark_cases(
        4096,
        include_categories={"reasoning"},
        reasoning_sweep=True,
    )
    # Should have 5 variants of medium_reasoning
    assert len(cases) == 5
    # All should be reasoning category
    assert all(c.category == "reasoning" for c in cases)
    # Names should include reasoning level
    names = {c.name for c in cases}
    expected = {
        "medium_reasoning_off",
        "medium_reasoning_low",
        "medium_reasoning_medium",
        "medium_reasoning_high",
        "medium_reasoning_xhigh",
    }
    assert names == expected


def test_reasoning_sweep_all_categories():
    """Test reasoning_sweep with all categories - only reasoning tests get variants."""
    cases = create_benchmark_cases(
        4096,
        include_categories=None,  # all categories
        reasoning_sweep=True,
    )
    # 4 normal tests + 5 reasoning variants = 9 total
    assert len(cases) == 9
    names = {c.name for c in cases}
    expected = {
        "short_instruction",
        "long_context",
        "coding_task",
        "structured_output",
        "medium_reasoning_off",
        "medium_reasoning_low",
        "medium_reasoning_medium",
        "medium_reasoning_high",
        "medium_reasoning_xhigh",
    }
    assert names == expected


def test_reasoning_sweep_temperature_per_level():
    """Test that each reasoning level has appropriate temperature."""
    cases = create_benchmark_cases(
        4096,
        include_categories={"reasoning"},
        reasoning_sweep=True,
    )
    temp_map = {c.name: c.temperature for c in cases}
    # off: 0.0, low: 0.1, medium: 0.3, high: 0.5, xhigh: 0.7
    assert temp_map["medium_reasoning_off"] == 0.0
    assert temp_map["medium_reasoning_low"] == 0.1
    assert temp_map["medium_reasoning_medium"] == 0.3
    assert temp_map["medium_reasoning_high"] == 0.5
    assert temp_map["medium_reasoning_xhigh"] == 0.7


def test_context_length_passed_through():
    """Test that context_length is passed to all cases."""
    cases = create_benchmark_cases(8192, include_categories={"coding"})
    assert len(cases) == 1
    assert cases[0].context_length == 8192


if __name__ == "__main__":
    pytest.main([__file__, "-v"])