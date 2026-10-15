"""Tests for fingerprint service."""

import pytest
from lm_optimizer.domain.models import GPUInfo, HardwareInfo
from lm_optimizer.services.fingerprint import (
    compute_fingerprint,
    hardware_fingerprint,
    fingerprint_diff,
    FINGERPRINT_FIELDS,
)


def test_fingerprint_deterministic():
    """Same inputs always produce same fingerprint."""
    fp1 = compute_fingerprint(
        model_id="m1",
        quantization="q4",
        architecture="arch",
        lm_studio_version="0.3.12",
        hardware_fingerprint="abc123",
        benchmark_suite_version="2026-10-15",
    )
    fp2 = compute_fingerprint(
        model_id="m1",
        quantization="q4",
        architecture="arch",
        lm_studio_version="0.3.12",
        hardware_fingerprint="abc123",
        benchmark_suite_version="2026-10-15",
    )
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA256 hex


def test_fingerprint_changes_on_input_change():
    """Any field change produces different fingerprint."""
    base = dict(
        model_id="m1",
        quantization="q4",
        architecture="arch",
        lm_studio_version="0.3.12",
        hardware_fingerprint="abc123",
        benchmark_suite_version="2026-10-15",
    )
    fp1 = compute_fingerprint(**base)
    fp2 = compute_fingerprint(**{**base, "lm_studio_version": "0.3.13"})
    assert fp1 != fp2


def test_hardware_fingerprint():
    """Hardware fingerprint from GPU names, VRAM, CPU, RAM."""
    hw = HardwareInfo(
        os="T",
        cpu_name="C",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        total_ram_gb=32.0,
        gpu_count=1,
        gpus=[GPUInfo(index=0, name="RTX 3060", vram_gb=6.0, vendor="NVIDIA")],
    )
    fp = hardware_fingerprint(hw)
    assert len(fp) == 16  # prefix
    # Same hardware = same fingerprint
    assert fp == hardware_fingerprint(hw)


def test_hardware_fingerprint_changes_on_gpu_change():
    """Different GPU produces different fingerprint."""
    hw1 = HardwareInfo(
        os="T",
        cpu_name="C",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        total_ram_gb=32.0,
        gpu_count=1,
        gpus=[GPUInfo(index=0, name="RTX 3060", vram_gb=6.0, vendor="NVIDIA")],
    )
    hw2 = HardwareInfo(
        os="T",
        cpu_name="C",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        total_ram_gb=32.0,
        gpu_count=1,
        gpus=[GPUInfo(index=0, name="RTX 4090", vram_gb=24.0, vendor="NVIDIA")],
    )
    assert hardware_fingerprint(hw1) != hardware_fingerprint(hw2)


def test_fingerprint_diff():
    """Diff returns list of changed field names."""
    old = {"lm_studio_version": "0.3.10", "hardware_fingerprint": "abc"}
    new = {"lm_studio_version": "0.3.12", "hardware_fingerprint": "abc"}
    assert fingerprint_diff(old, new) == ["lm_studio_version"]


def test_fingerprint_diff_multiple_changes():
    """Diff returns all changed fields."""
    old = {"lm_studio_version": "0.3.10", "hardware_fingerprint": "abc", "quantization": "q4"}
    new = {"lm_studio_version": "0.3.12", "hardware_fingerprint": "def", "quantization": "q4"}
    diff = fingerprint_diff(old, new)
    assert set(diff) == {"lm_studio_version", "hardware_fingerprint"}


def test_fingerprint_fields_constant():
    """FINGERPRINT_FIELDS contains all expected fields."""
    assert FINGERPRINT_FIELDS == [
        "model_id",
        "quantization",
        "architecture",
        "lm_studio_version",
        "hardware_fingerprint",
        "benchmark_suite_version",
    ]