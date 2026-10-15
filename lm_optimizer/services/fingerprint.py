"""Fingerprint service for model/run identification."""

import hashlib
import json
from typing import Any

from lm_optimizer.domain.models import GPUInfo, HardwareInfo


FINGERPRINT_FIELDS = [
    "model_id",
    "quantization",
    "architecture",
    "lm_studio_version",
    "hardware_fingerprint",
    "benchmark_suite_version",
]


def compute_fingerprint(
    model_id: str,
    quantization: str | None,
    architecture: str | None,
    lm_studio_version: str,
    hardware_fingerprint: str,
    benchmark_suite_version: str,
) -> str:
    """Compute SHA256 fingerprint from components.

    Args:
        model_id: Model identifier
        quantization: Model quantization
        architecture: Model architecture
        lm_studio_version: LM Studio version string
        hardware_fingerprint: 16-char hardware fingerprint prefix
        benchmark_suite_version: Benchmark suite version date

    Returns:
        64-char SHA256 hex digest
    """
    components = [
        model_id or "",
        quantization or "",
        architecture or "",
        lm_studio_version or "",
        hardware_fingerprint or "",
        benchmark_suite_version or "",
    ]
    joined = "|".join(components)
    return hashlib.sha256(joined.encode()).hexdigest()


def hardware_fingerprint(hw: HardwareInfo) -> str:
    """Compute 16-char hardware fingerprint from HardwareInfo.

    Args:
        hw: Hardware information snapshot

    Returns:
        16-char SHA256 hex prefix
    """
    gpu_data = []
    for gpu in hw.gpus or []:
        gpu_data.append({"name": gpu.name, "vram_gb": gpu.vram_gb})

    payload = {
        "gpus": sorted(gpu_data, key=lambda x: x["name"]),
        "cpu_cores_physical": hw.cpu_cores_physical,
        "cpu_cores_logical": hw.cpu_cores_logical,
        "total_ram_gb": hw.total_ram_gb,
    }
    json_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def fingerprint_diff(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Return list of field names that differ between two fingerprint component dicts.

    Args:
        old: Previous fingerprint components
        new: Current fingerprint components

    Returns:
        List of field names where values differ
    """
    changed = []
    for key in FINGERPRINT_FIELDS:
        old_val = old.get(key)
        new_val = new.get(key)
        if old_val != new_val:
            changed.append(key)
    return changed