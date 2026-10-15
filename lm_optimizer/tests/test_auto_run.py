"""Tests for auto-run CLI command."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from lm_optimizer.domain.models import (
    GPUInfo,
    HardwareInfo,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
    OptimizationStage,
    RunStatus,
)
from lm_optimizer.services.fingerprint import (
    compute_fingerprint,
    fingerprint_diff,
    hardware_fingerprint,
    should_revalidate_config,
)


def test_decide_action():
    """Test action decision logic."""
    from lm_optimizer.cli.auto_run import _decide_action

    # No run -> FULL
    assert _decide_action(None, "fp1", {}) == ("FULL", None)

    # Failed run -> FULL
    run = OptimizationRun(
        model=ModelIdentity(id="m1", name="m1"),
        hardware=HardwareInfo(os="T", cpu_name="C", cpu_cores_physical=1, cpu_cores_logical=2, total_ram_gb=8, gpu_count=0),
        profile=OptimizationProfile.BALANCED,
    )
    run.status = RunStatus.FAILED
    assert _decide_action(run, "fp1", {}) == ("FULL", None)

    # Complete + fingerprint match -> SKIP
    run = OptimizationRun(
        model=ModelIdentity(id="m1", name="m1"),
        hardware=HardwareInfo(os="T", cpu_name="C", cpu_cores_physical=1, cpu_cores_logical=2, total_ram_gb=8, gpu_count=0),
        profile=OptimizationProfile.BALANCED,
    )
    run.status = RunStatus.SUCCESS
    run.benchmark_params = {"fingerprint": "fp1", "fingerprint_components": {}}
    assert _decide_action(run, "fp1", {}) == ("SKIP", None)

    # Complete + fingerprint mismatch -> RE_RUN
    run = OptimizationRun(
        model=ModelIdentity(id="m1", name="m1"),
        hardware=HardwareInfo(os="T", cpu_name="C", cpu_cores_physical=1, cpu_cores_logical=2, total_ram_gb=8, gpu_count=0),
        profile=OptimizationProfile.BALANCED,
    )
    run.status = RunStatus.SUCCESS
    run.benchmark_params = {"fingerprint": "old_fp", "fingerprint_components": {"lm_studio_version": "0.3.10"}}
    action, diff = _decide_action(run, "new_fp", {"lm_studio_version": "0.3.12"})
    assert action == "RE_RUN"
    assert "lm_studio_version" in diff

    # Incomplete + fingerprint match -> RESUME
    run = OptimizationRun(
        model=ModelIdentity(id="m1", name="m1"),
        hardware=HardwareInfo(os="T", cpu_name="C", cpu_cores_physical=1, cpu_cores_logical=2, total_ram_gb=8, gpu_count=0),
        profile=OptimizationProfile.BALANCED,
    )
    run.status = RunStatus.INTERRUPTED
    run.benchmark_params = {"fingerprint": "fp1", "fingerprint_components": {}}
    assert _decide_action(run, "fp1", {}) == ("RESUME", None)

    # Incomplete + fingerprint mismatch -> RESUME_REVALIDATE
    run = OptimizationRun(
        model=ModelIdentity(id="m1", name="m1"),
        hardware=HardwareInfo(os="T", cpu_name="C", cpu_cores_physical=1, cpu_cores_logical=2, total_ram_gb=8, gpu_count=0),
        profile=OptimizationProfile.BALANCED,
    )
    run.status = RunStatus.INTERRUPTED
    run.benchmark_params = {"fingerprint": "old_fp", "fingerprint_components": {"hardware_fingerprint": "old_hw"}}
    action, diff = _decide_action(run, "new_fp", {"hardware_fingerprint": "new_hw"})
    assert action == "RESUME_REVALIDATE"
    assert "hardware_fingerprint" in diff


def test_compute_current_fingerprint():
    """Test fingerprint computation for a model."""
    from lm_optimizer.cli.auto_run import _compute_current_fingerprint, _get_lms_version
    from unittest.mock import patch

    model = ModelIdentity(
        id="test-model",
        name="test-model",
        quantization="q4_k_m",
        architecture="qwen3",
        context_limit=32768,
    )
    hardware = HardwareInfo(
        os="Windows",
        cpu_name="AMD Ryzen",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        total_ram_gb=32.0,
        gpu_count=1,
        gpus=[GPUInfo(index=0, name="RTX 3060", vram_gb=6.0, vendor="NVIDIA")],
    )

    with patch('lm_optimizer.cli.auto_run._get_lms_version', return_value="test-version"):
        fp, components = _compute_current_fingerprint(model, hardware)
        assert len(fp) == 64
        assert components["model_id"] == "test-model"
        assert components["quantization"] == "q4_k_m"
        assert components["architecture"] == "qwen3"
        assert components["lm_studio_version"] == "test-version"


def test_generate_report():
    """Test markdown report generation."""
    from lm_optimizer.cli.auto_run import _generate_report

    hardware = HardwareInfo(
        os="Windows",
        cpu_name="AMD Ryzen 7",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        total_ram_gb=32.0,
        gpu_count=1,
        gpus=[GPUInfo(index=0, name="RTX 3060", vram_gb=6.0, vendor="NVIDIA")],
    )

    results = [
        {"model": "m1", "action": "SKIP", "run_id": "run1", "best_score": 0.9, "fp_diff": None, "error": None},
        {"model": "m2", "action": "RESUME", "run_id": "run2", "best_score": 0.8, "fp_diff": None, "error": None},
        {"model": "m3", "action": "FULL", "run_id": "run3", "best_score": 0.85, "fp_diff": None, "error": None},
        {"model": "m4", "action": "FAILED", "run_id": None, "best_score": None, "fp_diff": None, "error": "Connection failed"},
    ]

    md = _generate_report(results, Path("results"), hardware, "0.3.12")

    assert "## Summary" in md
    assert "m1" in md and "SKIP" in md
    assert "m2" in md and "RESUME" in md
    assert "m3" in md and "FULL" in md
    assert "m4" in md and "FAILED" in md
    assert "Connection failed" in md
    assert "RTX 3060" in md
    assert "AMD Ryzen 7" in md


def test_confirm_plan():
    """Test plan confirmation logic."""
    from lm_optimizer.cli.auto_run import _confirm_plan

    plan = [
        {"action": "FULL"},
        {"action": "SKIP"},
        {"action": "RESUME"},
    ]

    # dry_run = True -> always confirm
    assert _confirm_plan(plan, dry_run=True, yes=False, non_interactive=False)

    # yes = True -> always confirm
    assert _confirm_plan(plan, dry_run=False, yes=True, non_interactive=False)

    # non_interactive = True -> always confirm
    assert _confirm_plan(plan, dry_run=False, yes=False, non_interactive=True)