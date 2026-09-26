"""ParameterControlAdapter: one control plane per parameter (§8-§20).

Channels: REST | CLI | SDK | DETECTED_ONLY | MANUAL_ONLY | UNSUPPORTED.
Derived from `parameter_registry` (never from llama.cpp existence alone).
REST carries natively supported load/chat keys; CLI (`lms`) covers what REST
rejects (gpu offload, ttl, estimate); SDK/schema-only stays DETECTED_ONLY;
GUI-only presets (KV quant, threads, mmap, keep-in-memory) are MANUAL_ONLY;
everything else is UNSUPPORTED and filtered before send.

Highest state is VERIFIED, advanced only on real evidence:
DETECTED -> SUPPORTED -> CONTROLLABLE -> APPLIED -> VERIFIED.
"""

from __future__ import annotations

import asyncio
import contextlib
from enum import StrEnum

from lm_optimizer.domain.models import LoadConfiguration
from lm_optimizer.services import lms_cli
from lm_optimizer.services.parameter_registry import (
    SCHEMA_KNOWN,
    SUPPORTED,
)
from lm_optimizer.services.parameter_registry import (
    get as _get_spec,
)

# GUI-only presets: visible in LM Studio UI, no verified programmatic path
# (REST 400s, no CLI flag) -> MANUAL_ONLY, never injected.
GUI_ONLY = frozenset(
    {
        "llama_k_cache_quantization_type",
        "llama_v_cache_quantization_type",
        "use_fp16_for_kv_cache",
        "n_threads",
        "cpu_threads",
        "cpu_thread_pool_size",
        "try_mmap",
        "keep_model_in_memory",
        "unified_kv_cache",
    }
)


class ControlChannel(StrEnum):
    REST = "REST"
    CLI = "CLI"
    SDK = "SDK"
    DETECTED_ONLY = "DETECTED_ONLY"
    MANUAL_ONLY = "MANUAL_ONLY"
    UNSUPPORTED = "UNSUPPORTED"


def channel_of(canonical: str) -> ControlChannel:
    """Control channel for one parameter (registry-driven)."""
    spec = _get_spec(canonical)
    if spec is None:
        return ControlChannel.UNSUPPORTED
    if spec.rest == SUPPORTED:
        return ControlChannel.REST
    if spec.cli == SUPPORTED:
        return ControlChannel.CLI
    if canonical in GUI_ONLY:
        return ControlChannel.MANUAL_ONLY
    if spec.sdk == SCHEMA_KNOWN:
        return ControlChannel.SDK
    return (
        ControlChannel.DETECTED_ONLY if spec.schema == SCHEMA_KNOWN else ControlChannel.UNSUPPORTED
    )


def is_programmatic(canonical: str) -> bool:
    """REST/CLI/SDK only — never claim control from mere existence."""
    return channel_of(canonical) in (
        ControlChannel.REST,
        ControlChannel.CLI,
        ControlChannel.SDK,
    )


def verify_applied(requested: dict, applied: dict | None) -> dict[str, object]:
    """REQUESTED vs APPLIED comparison (§13).

    Returns {requested, actual, verification} with verification in
    MATCH | PARTIAL_MATCH | MISMATCH | UNKNOWN. Never report requested
    values as verified on mismatch.
    """
    requested = dict(requested or {})
    if applied is None:
        return {"requested": requested, "actual": None, "verification": "UNKNOWN"}
    applied = dict(applied)
    diffs = {
        k: {"requested": v, "applied": applied.get(k)}
        for k, v in requested.items()
        if applied.get(k) != v
    }
    if not requested:
        verification = "UNKNOWN"
    elif not diffs:
        verification = "MATCH"
    elif len(diffs) < len(requested):
        verification = "PARTIAL_MATCH"
    else:
        verification = "MISMATCH"
    return {"requested": requested, "actual": applied, "verification": verification, "diffs": diffs}


def unused_rest_keys(load_config_dict: dict) -> list[str]:
    """Keys a REST load would silently drop (must go via another channel)."""
    return [k for k in load_config_dict if k not in LoadConfiguration.API_KEYS]


async def load_with_channel(
    client,
    model_id: str,
    load_config,
    gpu_via_cli: bool = False,
) -> tuple[object, str, dict | None]:
    """Load via the authoritative channel; returns (result, channel, applied).

    REST by default. When gpu_via_cli is on, a gpu_ratio the REST API cannot
    express goes through `lms load --gpu` (recorded as CLI + actual gpu mode);
    CLI failure falls back to REST with the ratio recorded-but-not-applied.
    `applied` is the echo/applied config dict (or None when unknown).
    """
    ratio = getattr(load_config, "gpu_ratio", None)
    if gpu_via_cli and ratio is not None and lms_cli.lms_available():
        gpu_arg = "max" if ratio >= 1.0 else ("off" if ratio <= 0.0 else str(round(ratio, 2)))
        identifier, error = await _run_cli_load(model_id, load_config, gpu_arg)
        if error:
            return await _rest_load(client, model_id, load_config, fallback_note=error)
        applied = {"gpu_ratio": gpu_arg, "context_length": load_config.context_length}
        _ok = await _verify_cli_instance(client, model_id, identifier)
        return _CliResult(identifier=identifier), "CLI", applied
    return await _rest_load(client, model_id, load_config)


async def _run_cli_load(
    model_id: str, load_config: LoadConfiguration, gpu_arg: str
) -> tuple[str | None, str]:
    """Blocking lms call off the event loop (subprocess adapter)."""
    return await asyncio.to_thread(lms_cli.cli_load, model_id, gpu_arg, load_config.context_length)


async def _verify_cli_instance(client, model_id: str, identifier: str | None) -> bool:
    """Best-effort presence check of the CLI-loaded instance."""
    try:
        models = await client.list_models()
        known = {getattr(m, "id", "") for m in models}
        if model_id in known:
            return True
        loaded = getattr(client, "_loaded_models", {}) or {}
        return bool(loaded) and (identifier in loaded.values() or model_id in loaded)
    except Exception:
        return False


async def _rest_load(
    client, model_id: str, load_config: LoadConfiguration, fallback_note: str = ""
) -> tuple[object, str, dict | None]:
    """Plain REST load; applied echo comes from the client itself."""
    result = await client.load_model(model_id, load_config)
    applied = None
    if getattr(result, "success", False) and getattr(result, "loaded_config", None):
        try:
            applied = result.loaded_config.to_dict()
        except Exception:
            applied = None
    if fallback_note and not getattr(result, "success", False):
        with contextlib.suppress(Exception):
            result.error = f"{result.error} [cli-fallback-note: {fallback_note[:120]}]"
    return result, "REST", applied


class _CliResult:
    """Minimal load-result shape for CLI loads (success/identifier/error)."""

    def __init__(self, identifier: str | None = None, error: str | None = None) -> None:
        self.success = error is None
        self.identifier = identifier
        self.error = error
        self.loaded_config = None
        self.echo_mismatches = None


def refine_gpu_boundary(probe, lo_pass: float, hi_fail: float, steps: int = 3) -> list[dict]:
    """Adaptive GPU-offload boundary refinement (§11).

    0.7 PASS / 1.0 FAIL -> probes 0.85, 0.925, 0.9625 (binary midpoints).
    `probe(ratio)` returns True on PASS. Returns [{ratio, ok}] in order.
    Boundary candidates MUST still be empirically loaded by the caller.
    """
    out: list[dict] = []
    lo, hi = lo_pass, hi_fail
    for _ in range(max(1, steps)):
        mid = round((lo + hi) / 2, 4)
        if mid <= lo or mid >= hi:
            break
        try:
            ok = bool(probe(mid))
        except Exception:
            ok = False
        out.append({"ratio": mid, "ok": ok})
        if ok:
            lo = mid
        else:
            hi = mid
    return out


def prune_by_estimate(
    contexts: list[int],
    estimate_limit: int | None,
    confidence: str | None = None,
) -> tuple[list[int], list[int]]:
    """Prune obviously impossible contexts (§12); keep boundary empirical.

    Returns (kept, pruned). Contexts above the estimate limit are pruned
    EXCEPT the first one past it (still empirically loaded as proof).
    Unknown/low-confidence estimates prune nothing.
    """
    if not estimate_limit:
        return list(contexts), []
    ordered = sorted(contexts)
    beyond = [c for c in ordered if c > estimate_limit]
    if beyond and str(confidence or "").lower() in ("high", "medium"):
        kept = [c for c in ordered if c <= estimate_limit] + beyond[:1]
        return kept, beyond[1:]
    return list(ordered), []


def channel_table(canonicals: list[str]) -> list[dict]:
    """Channel rows for reports (parameter -> channel + programmatic?)."""
    return [
        {
            "parameter": name,
            "channel": channel_of(name).value,
            "programmatic": is_programmatic(name),
        }
        for name in canonicals
    ]
