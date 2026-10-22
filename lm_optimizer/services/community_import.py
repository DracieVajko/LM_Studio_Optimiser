"""Community benchmark database import/export (spec 16.4).

The database ships with curated records; this module lets users import
additional records from JSON and export the merged set.

Schema per record (all fields required unless noted):
    hardware: {gpu_name, vram_gb, gpu_bandwidth_gbps, gpu_tdp_w,
               cpu_name, physical_cores, logical_threads, ram_gb,
               ram_channels, disk_type, os_name}
    model_repository, model_filename, quant, context_length,
    gpu_offload, flash_attention, kv_on_gpu, kv_quant,
    eval_batch_size, threads, mmap, mtp, prompt_tps, decode_tps,
    ttft_s (optional), source, confidence (A|B|C|D)

Validation is strict: a malformed record is rejected with a clear reason so
a bad import can never silently poison the recommender.
"""

from __future__ import annotations

import json
from pathlib import Path

from lm_optimizer.services.recommender import (
    CommunityBenchmark,
    Confidence,
    HardwareFingerprint,
)

_REQUIRED_TOP = (
    "hardware",
    "model_repository",
    "model_filename",
    "quant",
    "context_length",
    "decode_tps",
)

_REQUIRED_HW = ("gpu_name", "vram_gb")


class ImportError_(ValueError):
    """Raised when a community benchmark record is malformed."""


def _hw_from_dict(data: dict) -> HardwareFingerprint:
    if not isinstance(data, dict):
        raise ImportError_("hardware must be an object")
    for key in _REQUIRED_HW:
        if key not in data:
            raise ImportError_(f"hardware.{key} is required")
    return HardwareFingerprint(
        gpu_name=str(data.get("gpu_name", "")),
        vram_gb=float(data.get("vram_gb", 0) or 0),
        gpu_bandwidth_gbps=float(data.get("gpu_bandwidth_gbps", 0) or 0),
        gpu_tdp_w=float(data.get("gpu_tdp_w", 0) or 0),
        cpu_name=str(data.get("cpu_name", "")),
        physical_cores=int(data.get("physical_cores", 0) or 0),
        logical_threads=int(data.get("logical_threads", 0) or 0),
        ram_gb=float(data.get("ram_gb", 0) or 0),
        ram_channels=int(data.get("ram_channels", 1) or 1),
        disk_type=str(data.get("disk_type", "")),
        os_name=str(data.get("os_name", "")),
    )


def record_from_dict(data: dict) -> CommunityBenchmark:
    """Validate one record and convert it to a CommunityBenchmark."""
    if not isinstance(data, dict):
        raise ImportError_("record must be an object")
    for key in _REQUIRED_TOP:
        if key not in data:
            raise ImportError_(f"{key} is required")
    try:
        confidence = Confidence(str(data.get("confidence", "D")).upper())
    except ValueError:
        raise ImportError_(
            f"confidence must be one of A/B/C/D, got {data.get('confidence')!r}"
        )
    decode = float(data["decode_tps"])
    if decode <= 0:
        raise ImportError_("decode_tps must be > 0")
    return CommunityBenchmark(
        hardware=_hw_from_dict(data["hardware"]),
        model_repository=str(data["model_repository"]),
        model_filename=str(data["model_filename"]),
        quant=str(data["quant"]),
        context_length=int(data["context_length"]),
        gpu_offload=float(data.get("gpu_offload", 1.0)),
        flash_attention=bool(data.get("flash_attention", True)),
        kv_on_gpu=bool(data.get("kv_on_gpu", True)),
        kv_quant=str(data.get("kv_quant", "f16")),
        eval_batch_size=int(data.get("eval_batch_size", 2048)),
        threads=int(data.get("threads", 0)),
        mmap=bool(data.get("mmap", True)),
        mtp=bool(data.get("mtp", False)),
        prompt_tps=float(data.get("prompt_tps", 0) or 0),
        decode_tps=decode,
        ttft_s=data.get("ttft_s"),
        source=str(data.get("source", "")),
        confidence=confidence,
    )


def record_to_dict(record: CommunityBenchmark) -> dict:
    """Serialize one record back to the import schema."""
    return {
        "hardware": record.hardware.as_dict(),
        "model_repository": record.model_repository,
        "model_filename": record.model_filename,
        "quant": record.quant,
        "context_length": record.context_length,
        "gpu_offload": record.gpu_offload,
        "flash_attention": record.flash_attention,
        "kv_on_gpu": record.kv_on_gpu,
        "kv_quant": record.kv_quant,
        "eval_batch_size": record.eval_batch_size,
        "threads": record.threads,
        "mmap": record.mmap,
        "mtp": record.mtp,
        "prompt_tps": record.prompt_tps,
        "decode_tps": record.decode_tps,
        "ttft_s": record.ttft_s,
        "source": record.source,
        "confidence": record.confidence.value,
    }


def import_records(
    payload: list[dict] | dict,
    existing: list[CommunityBenchmark] | None = None,
) -> tuple[list[CommunityBenchmark], list[str]]:
    """Import records, deduplicate against `existing`, return (records, warnings).

    Deduplication key: (model_filename, quant, hardware fingerprint, mtp).
    A duplicate is skipped with a warning rather than silently overwriting.
    """
    if isinstance(payload, dict):
        if "records" not in payload:
            raise ImportError_("object payload must contain a 'records' list")
        payload = payload.get("records", [])
    if not isinstance(payload, list):
        raise ImportError_("payload must be a list of records or {'records': [...]}")

    base = list(existing or [])
    seen = {
        (
            r.model_filename,
            r.quant,
            r.hardware.as_dict().__repr__(),
            r.mtp,
        )
        for r in base
    }
    out = list(base)
    warnings: list[str] = []
    for index, raw in enumerate(payload):
        try:
            record = record_from_dict(raw)
        except ImportError_ as e:
            warnings.append(f"record {index}: {e}")
            continue
        key = (
            record.model_filename,
            record.quant,
            record.hardware.as_dict().__repr__(),
            record.mtp,
        )
        if key in seen:
            warnings.append(
                f"record {index}: duplicate of {record.model_filename} "
                f"({record.quant}, {record.hardware.gpu_name})"
            )
            continue
        seen.add(key)
        out.append(record)
    return out, warnings


def load_json_file(path: str | Path) -> list[dict]:
    """Read a JSON import file (a list or {"records": [...]})."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("records", []))
    if isinstance(data, list):
        return data
    raise ImportError_("JSON root must be a list or an object with 'records'")


def export_records(records: list[CommunityBenchmark]) -> dict:
    """Export records in the import schema."""
    return {"records": [record_to_dict(r) for r in records]}
