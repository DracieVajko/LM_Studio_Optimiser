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


def _seed_run(status, fp=None, comps=None):
    run = OptimizationRun(
        model=ModelIdentity(id="m1", name="m1"),
        hardware=HardwareInfo(
            os="T", cpu_name="C", cpu_cores_physical=1,
            cpu_cores_logical=2, total_ram_gb=8, gpu_count=0,
        ),
        profile=OptimizationProfile.BALANCED,
    )
    run.status = status
    run.benchmark_params = {"fingerprint": fp, "fingerprint_components": comps or {}}
    return run


_OLD = {
    "model_id": "m1",
    "quantization": "q4",
    "architecture": "qwen",
    "lm_studio_version": "semver:0.3.10",
    "hardware_fingerprint": "hw1",
    "benchmark_suite_version": "2026-10-10",
}


def test_decide_action():
    """Test action decision logic."""
    from lm_optimizer.cli.auto_run import _decide_action

    new = {**_OLD, "lm_studio_version": "semver:0.3.12"}

    # No run -> FULL
    assert _decide_action(None, "fp1", {}) == ("FULL", None)

    # Failed/cancelled (not resumable) -> FULL
    assert _decide_action(_seed_run(RunStatus.FAILED, "fp1", _OLD), "fp1", _OLD) == ("FULL", None)
    assert _decide_action(_seed_run(RunStatus.CANCELLED, "fp1", _OLD), "fp1", _OLD) == ("FULL", None)

    # COMPLETE + fingerprint match -> SKIP
    assert _decide_action(_seed_run(RunStatus.SUCCESS, "fp1", _OLD), "fp1", _OLD) == ("SKIP", None)

    # COMPLETE + fingerprint mismatch -> RE_RUN with changed scope
    action, diff = _decide_action(_seed_run(RunStatus.SUCCESS, "old_fp", _OLD), "new_fp", new)
    assert action == "RE_RUN"
    assert diff == ["lm_studio_version"]

    # INCOMPLETE + fingerprint match -> RESUME
    assert _decide_action(_seed_run(RunStatus.INTERRUPTED, "fp1", _OLD), "fp1", _OLD) == ("RESUME", None)
    assert _decide_action(_seed_run(RunStatus.RUNNING, "fp1", _OLD), "fp1", _OLD) == ("RESUME", None)

    # INCOMPLETE + fingerprint mismatch -> RESUME_REVALIDATE
    action, diff = _decide_action(_seed_run(RunStatus.INTERRUPTED, "old_fp", _OLD), "new_fp", new)
    assert action == "RESUME_REVALIDATE"
    assert diff == ["lm_studio_version"]


def test_absent_fingerprint_is_unknown_not_mismatch():
    """A run predating fingerprinting must NOT be treated as changed.

    Absent fingerprint carries no evidence of a change; re-running every
    completed model on the first auto-run would discard measured work for no
    reason, and the fingerprint is backfilled on the next resume.
    """
    from lm_optimizer.cli.auto_run import _decide_action

    current = {**_OLD, "lm_studio_version": "0.3.12", "benchmark_suite_version": "2026-10-15"}
    no_fp_complete = _seed_run(RunStatus.SUCCESS)
    no_fp_complete.benchmark_params = {}
    no_fp_incomplete = _seed_run(RunStatus.INTERRUPTED)
    no_fp_incomplete.benchmark_params = {}

    assert _decide_action(no_fp_complete, "fp_new", current) == ("SKIP", None)
    assert _decide_action(no_fp_incomplete, "fp_new", current) == ("RESUME", None)


def test_identity_fields_excluded_from_revalidation_scope():
    """model_id/quantization/architecture cannot differ for a looked-up run.

    They identify WHICH model the run belongs to. A stale stored value must not
    scope a revalidation when every environment field still matches, because
    the measurements are still valid for this model on this host.
    """
    from lm_optimizer.cli.auto_run import _decide_action

    # Environment fields all match; only the identity fields are stale.
    run = _seed_run(RunStatus.INTERRUPTED, "old_fp", {
        "model_id": "", "quantization": None, "architecture": None,
        "lm_studio_version": "semver:0.3.12",
        "hardware_fingerprint": "hw1",
        "benchmark_suite_version": "2026-10-15",
    })
    current = {
        "model_id": "m1",
        "quantization": "q4",
        "architecture": "qwen",
        "lm_studio_version": "semver:0.3.12",
        "hardware_fingerprint": "hw1",
        "benchmark_suite_version": "2026-10-15",
    }

    action, diff = _decide_action(run, "new_fp", current)
    assert action == "RESUME", "identity-only diff must not force revalidation"
    assert diff is None


def test_environment_change_does_scope_revalidation():
    """A hardware/version change alongside identity noise still revalidates."""
    from lm_optimizer.cli.auto_run import _decide_action

    run = _seed_run(RunStatus.INTERRUPTED, "old_fp", {
        "model_id": "", "quantization": None, "architecture": None,
        "lm_studio_version": "semver:0.3.10",
        "hardware_fingerprint": "hw1",
        "benchmark_suite_version": "2026-10-15",
    })
    current = {
        "model_id": "m1",
        "quantization": "q4",
        "architecture": "qwen",
        "lm_studio_version": "semver:0.3.12",
        "hardware_fingerprint": "hw1",
        "benchmark_suite_version": "2026-10-15",
    }

    action, diff = _decide_action(run, "new_fp", current)
    assert action == "RESUME_REVALIDATE"
    assert diff == ["lm_studio_version"], "identity noise must not widen the scope"


def test_compute_current_fingerprint():
    """Test fingerprint computation for a model."""
    from lm_optimizer.cli.auto_run import _compute_current_fingerprint, _get_lms_version
    from lm_optimizer.services.version_identity import VersionIdentity
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

    with patch(
        'lm_optimizer.cli.auto_run._version_identity',
        return_value=VersionIdentity(
            raw="CLI commit: dcefd5a", semantic=None, build="dcefd5a",
            source="lms-cli", compatibility_key="build:non-loading",
        ),
    ):
        fp, components = _compute_current_fingerprint(model, hardware)
        assert len(fp) == 64
        assert components["model_id"] == "test-model"
        assert components["quantization"] == "q4_k_m"
        assert components["architecture"] == "qwen3"
        # The compatibility key is fingerprinted, never the raw build string.
        assert components["lm_studio_version"] == "build:non-loading"
        # The raw build string travels for diagnostics only.
        assert components["_version_identity"]["raw"] == "CLI commit: dcefd5a"
        assert "_version_identity" not in (
            __import__("lm_optimizer.services.fingerprint", fromlist=["FINGERPRINT_FIELDS"]).FINGERPRINT_FIELDS
        )


def test_generate_report(tmp_path):
    """Report rendering writes ONLY into the given temporary directory."""
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
        {"model": "qwen3-32b-instruct", "action": "SKIP", "run_id": "run-a", "best_score": 0.9, "fp_diff": None, "error": None},
        {"model": "llama3.1-8b", "action": "RESUME", "run_id": "run-b", "best_score": 0.8, "fp_diff": None, "error": None},
        {"model": "mistral-7b-v0.3", "action": "FULL", "run_id": "run-c", "best_score": 0.85, "fp_diff": None, "error": None},
        {"model": "gemma-2-2b", "action": "FAILED", "run_id": None, "best_score": None, "fp_diff": None, "error": "Connection refused"},
    ]

    md = _generate_report(results, tmp_path, hardware, "0.3.12")

    assert "## Summary" in md
    assert "qwen3-32b-instruct" in md and "SKIP" in md
    assert "llama3.1-8b" in md and "RESUME" in md
    assert "mistral-7b-v0.3" in md and "FULL" in md
    assert "gemma-2-2b" in md and "FAILED" in md
    assert "Connection refused" in md
    assert "RTX 3060" in md
    assert "AMD Ryzen 7" in md

    # Regression: a unit test must never write into the repository's
    # persistent results directory.
    assert list(tmp_path.glob("auto-run-*.md")), "report written to tmp_path"
    assert not (Path.cwd() / "results" / "auto-run-2026").exists()


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