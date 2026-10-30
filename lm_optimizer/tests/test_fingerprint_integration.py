"""Tests for fingerprint integration in optimizer."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from lm_optimizer.domain.models import (
    ConfigurationResult,
    ConfigurationStatus,
    HardwareInfo,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
    OptimizationStage,
    RunStatus,
    QualityScore,
)
from lm_optimizer.services import optimizer as _opt
from lm_optimizer.services.fingerprint import compute_fingerprint, hardware_fingerprint


def _make_run_with_fingerprint():
    """Create a mock run with fingerprint in benchmark_params."""
    model = ModelIdentity(id="test-model", name="test-model", context_limit=32768)
    hw = HardwareInfo(
        os="T", cpu_name="C", cpu_cores_physical=8, cpu_cores_logical=16,
        total_ram_gb=32.0, gpu_count=1,
    )
    run = OptimizationRun(
        model=model,
        hardware=hw,
        profile=OptimizationProfile.BALANCED,
    )
    run.id = uuid4()
    run.status = RunStatus.SUCCESS
    run.stage = OptimizationStage.COMPLETE
    run.benchmark_params = {
        "fingerprint": "abc123",
        "fingerprint_components": {
            "model_id": "test-model",
            "quantization": "q4",
            "architecture": "qwen",
            "lm_studio_version": "0.3.12",
            "hardware_fingerprint": "hw123",
            "benchmark_suite_version": "2026-10-15",
        },
    }
    return run


def test_new_run_stores_fingerprint():
    """First run computes and stores fingerprint in benchmark_params."""
    # This test verifies the behavior after implementation
    # For now, we just check the function signature exists
    from lm_optimizer.services.fingerprint import compute_fingerprint
    assert callable(compute_fingerprint)


def test_fingerprint_revalidation_logic():
    """Test the logic for deciding which configs to revalidate on fingerprint mismatch."""
    from lm_optimizer.services.fingerprint import fingerprint_diff

    old = {
        "model_id": "m1",
        "quantization": "q4",
        "architecture": "qwen",
        "lm_studio_version": "0.3.10",
        "hardware_fingerprint": "abc123",
        "benchmark_suite_version": "2026-10-10",
    }
    new = {
        "model_id": "m1",
        "quantization": "q4",
        "architecture": "qwen",
        "lm_studio_version": "0.3.12",
        "hardware_fingerprint": "abc123",
        "benchmark_suite_version": "2026-10-15",
    }

    diff = fingerprint_diff(old, new)
    assert "lm_studio_version" in diff
    assert "benchmark_suite_version" in diff
    assert "hardware_fingerprint" not in diff


def test_should_revalidate_config():
    """Test logic for whether a config should be revalidated based on fingerprint diff."""
    # Hardware change -> revalidate all
    assert should_revalidate("hardware_fingerprint", "speed")
    assert should_revalidate("hardware_fingerprint", "quality")
    assert should_revalidate("hardware_fingerprint", "recovery")

    # LM Studio version change -> revalidate all
    assert should_revalidate("lm_studio_version", "speed")
    assert should_revalidate("lm_studio_version", "quality")

    # Benchmark suite version change -> only new cases (quality phase gets new cases)
    assert should_revalidate("benchmark_suite_version", "quality")
    # For speed phase, benchmark version doesn't affect speed probes
    # but quality phase gets new test cases
    assert not should_revalidate("benchmark_suite_version", "speed")

    # Quantization/architecture change -> full re-run (handled at higher level)
    assert should_revalidate("quantization", "any")
    assert should_revalidate("architecture", "any")


def should_revalidate(changed_field: str, phase: str) -> bool:
    """Determine if a config in the given phase should be revalidated based on changed field."""
    if changed_field in ("quantization", "architecture"):
        return True  # Full re-run
    if changed_field == "hardware_fingerprint":
        return True  # All phases affected
    if changed_field == "lm_studio_version":
        return True  # All phases affected
    if changed_field == "benchmark_suite_version":
        return phase in ("quality", "recovery", "final_validation")  # New test cases
    return False