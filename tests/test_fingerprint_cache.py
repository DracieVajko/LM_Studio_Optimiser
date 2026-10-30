"""Fingerprint and cache-invalidation correctness (spec §10).

Candidate identity must change with every parameter that materially changes a
result; cached evidence must invalidate when model/runtime/hardware/benchmark
changes, and a benchmark-suite change must not invalidate speed probes.
"""

from __future__ import annotations

import pytest

from lm_optimizer.domain.models import (
    ConfigurationResult,
    GPUInfo,
    HardwareInfo,
    LoadConfiguration,
)
from lm_optimizer.services.fingerprint import (
    FINGERPRINT_FIELDS,
    compute_fingerprint,
    filter_configs_for_revalidation,
    fingerprint_diff,
    hardware_fingerprint,
    should_revalidate_config,
)
from lm_optimizer.services.optimizer import AdaptiveOptimizer


def _hw(vram=24.0, ram=64.0, gpus=None):
    return HardwareInfo(
        os="windows",
        cpu_name="cpu",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        total_ram_gb=ram,
        gpu_count=len(gpus if gpus is not None else [1]),
        gpus=gpus
        if gpus is not None
        else [GPUInfo(index=0, name="RTX 4090", vram_gb=vram, vendor="NVIDIA")],
    )


def _fp(**overrides):
    base = dict(
        model_id="m",
        quantization="Q4_K_M",
        architecture="qwen",
        lm_studio_version="0.3.12",
        hardware_fingerprint="abc123",
        benchmark_suite_version="2026-10-01",
    )
    base.update(overrides)
    return compute_fingerprint(**base)


# --- fingerprint differentiation -------------------------------------------

def test_fingerprint_is_sha256_and_deterministic():
    a = _fp()
    assert a == _fp()
    assert len(a) == 64
    int(a, 16)  # valid hex


@pytest.mark.parametrize(
    "field, value",
    [
        ("model_id", "other"),
        ("quantization", "Q3_K_M"),
        ("architecture", "llama"),
        ("lm_studio_version", "0.3.13"),
        ("hardware_fingerprint", "def456"),
        ("benchmark_suite_version", "2026-11-01"),
    ],
)
def test_every_material_component_changes_the_fingerprint(field, value):
    assert _fp(**{field: value}) != _fp()


def test_fingerprint_fields_list_is_stable():
    assert FINGERPRINT_FIELDS == [
        "model_id",
        "quantization",
        "architecture",
        "lm_studio_version",
        "hardware_fingerprint",
        "benchmark_suite_version",
    ]


# --- hardware signature -----------------------------------------------------

def test_hardware_fingerprint_is_stable_and_short():
    assert hardware_fingerprint(_hw()) == hardware_fingerprint(_hw())
    assert len(hardware_fingerprint(_hw())) == 16


def test_hardware_fingerprint_ignores_gpu_order():
    a = _hw(
        gpus=[
            GPUInfo(index=0, name="A", vram_gb=24, vendor="NVIDIA"),
            GPUInfo(index=1, name="B", vram_gb=24, vendor="NVIDIA"),
        ]
    )
    b = _hw(
        gpus=[
            GPUInfo(index=1, name="B", vram_gb=24, vendor="NVIDIA"),
            GPUInfo(index=0, name="A", vram_gb=24, vendor="NVIDIA"),
        ]
    )
    assert hardware_fingerprint(a) == hardware_fingerprint(b)


def test_hardware_fingerprint_changes_with_memory_and_cores():
    assert hardware_fingerprint(_hw(ram=64)) != hardware_fingerprint(_hw(ram=32))
    assert hardware_fingerprint(_hw(vram=24)) != hardware_fingerprint(_hw(vram=12))


# --- diff / revalidation policy --------------------------------------------

def test_fingerprint_diff_reports_only_changed_fields():
    old = {k: "x" for k in FINGERPRINT_FIELDS}
    new = dict(old)
    new["quantization"] = "Q3_K_M"
    new["benchmark_suite_version"] = "2026-11-01"
    assert fingerprint_diff(old, new) == ["quantization", "benchmark_suite_version"]


def test_identical_components_have_no_diff():
    comps = {k: "x" for k in FINGERPRINT_FIELDS}
    assert fingerprint_diff(comps, dict(comps)) == []


@pytest.mark.parametrize(
    "field",
    ["quantization", "architecture", "hardware_fingerprint", "lm_studio_version"],
)
def test_material_change_revalidates_all_phases(field):
    for phase in ("speed", "quality", "recovery", "final_validation"):
        assert should_revalidate_config(field, phase) is True


def test_benchmark_suite_change_keeps_speed_probes():
    # A new task set does not invalidate a measured decode speed.
    assert should_revalidate_config("benchmark_suite_version", "speed") is False
    for phase in ("quality", "recovery", "final_validation"):
        assert should_revalidate_config("benchmark_suite_version", phase) is True


def test_unknown_field_does_not_force_revalidation():
    assert should_revalidate_config("cosmetic", "quality") is False


def test_filter_configs_splits_by_phase():
    def res(phase):
        r = ConfigurationResult(config=LoadConfiguration(context_length=4096))
        r.generation = {"phase": phase}
        return r

    speed = res("speed")
    quality = res("quality")
    keep, reval = filter_configs_for_revalidation(
        [speed, quality], ["benchmark_suite_version"]
    )
    assert keep == [speed]
    assert reval == [quality]


# --- candidate identity / duplicates ---------------------------------------

def test_identical_configs_share_a_candidate_key():
    a = LoadConfiguration(context_length=4096, flash_attention=True, eval_batch_size=512)
    b = LoadConfiguration(context_length=4096, flash_attention=True, eval_batch_size=512)
    assert AdaptiveOptimizer._candidate_key(a) == AdaptiveOptimizer._candidate_key(b)


@pytest.mark.parametrize(
    "field, value",
    [
        ("context_length", 8192),
        ("flash_attention", False),
        ("offload_kv_cache_to_gpu", False),
        ("eval_batch_size", 1024),
        ("physical_batch_size", 1024),
        ("parallel", 2),
        ("context_checkpoints", 64),
        ("num_experts", 4),
        ("speculative_draft_max_tokens", 3),
        ("speculative_draft_model", "draft"),
    ],
)
def test_every_load_parameter_changes_the_candidate_key(field, value):
    base = LoadConfiguration(
        context_length=4096,
        flash_attention=True,
        offload_kv_cache_to_gpu=True,
        eval_batch_size=512,
        physical_batch_size=512,
        parallel=1,
        context_checkpoints=32,
        num_experts=8,
        speculative_draft_max_tokens=2,
        speculative_draft_model="m",
    )
    other = LoadConfiguration(**base.to_dict())
    setattr(other, field, value)
    assert AdaptiveOptimizer._candidate_key(base) != AdaptiveOptimizer._candidate_key(other)


def test_duplicate_probe_is_recognised_by_the_tested_key_set():
    cfg = LoadConfiguration(context_length=4096, flash_attention=True)
    tested = {AdaptiveOptimizer._candidate_key(cfg)}
    duplicate = LoadConfiguration(context_length=4096, flash_attention=True)
    fresh = LoadConfiguration(context_length=8192, flash_attention=True)
    assert AdaptiveOptimizer._candidate_key(duplicate) in tested
    assert AdaptiveOptimizer._candidate_key(fresh) not in tested
