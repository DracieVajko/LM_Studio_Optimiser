"""Host safety guard: unload-all + resource snapshot + verify-empty.

Called at the start of every command/script run: no test may begin with a
stale loaded model, and the run records how much VRAM/RAM/swap is free.
All explanations are predefined (no AI needed at runtime).
"""

import asyncio
import inspect
import json
import platform
import subprocess
import time
from typing import Any

from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)

NVSMI_TIMEOUT_S = 15
MB_PER_GB = 1024**3
NVSMI_FIELDS = 4  # index, name, mem.used, mem.total


def gpu_free_mb() -> list[dict[str, Any]]:
    """Per-GPU free/total MB via nvidia-smi (empty-error dict if unavailable).

    Apple Silicon has unified memory (no separate VRAM): reported honestly
    as a unified entry instead of a fake nvidia-smi error.
    """
    if platform.system() == "Darwin":
        return [
            {
                "id": 0,
                "name": "Apple Silicon (unified memory)",
                "unified_memory": True,
                "note": "No separate VRAM on Apple Silicon; see RAM figures.",
            }
        ]
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=NVSMI_TIMEOUT_S,
            check=False,
        )
    except Exception as e:
        return [{"error": f"GPU monitor unavailable: {e}"}]
    snap = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != NVSMI_FIELDS:
            continue
        idx, name, used, total = parts
        snap.append(
            {
                "id": int(idx),
                "name": name,
                "free_mb": float(total) - float(used),
                "total_mb": float(total),
            }
        )
    return snap or [{"error": "nvidia-smi returned no GPUs"}]


def mem_free_gb() -> dict[str, Any]:
    """Free RAM/swap GB, or {unavailable} when psutil is missing."""
    try:
        import psutil  # noqa: PLC0415 - lazy import keeps guard working without psutil
    except ImportError as e:
        return {"unavailable": f"RAM monitor unavailable: {e}"}
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {
        "ram_free_gb": vm.available / MB_PER_GB,
        "ram_total_gb": vm.total / MB_PER_GB,
        "swap_free_gb": (sw.total - sw.used) / MB_PER_GB,
        "swap_total_gb": sw.total / MB_PER_GB,
    }


def storage_info() -> dict[str, Any]:
    """Storage + pagefile placement + swap (detection only, never modified).

    Windows: physical disk media types (NVMe/SSD/HDD) + pagefile locations.
    Elsewhere: unknown (guarded). Property names match case-insensitively
    (DeviceId/DeviceID, FriendlyName fallback) because CIM/PS versions vary.
    """
    info: dict[str, Any] = {"drives": [], "pagefile": "unknown"}
    if platform.system() != "Windows":
        info["note"] = "storage detection implemented for Windows only"
        return info

    def _get(row: dict, *names: str, default=None):
        lowered = {str(k).lower(): v for k, v in row.items()} if isinstance(row, dict) else {}
        for name in names:
            if name.lower() in lowered:
                return lowered[name.lower()]
        return default

    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-PhysicalDisk | Select-Object DeviceId, FriendlyName, MediaType, Size | ConvertTo-Json",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            data = json.loads(out.stdout.strip())
            rows = data if isinstance(data, list) else [data]
            for row in rows:
                info["drives"].append(
                    {
                        "id": _get(row, "DeviceId", "DeviceID", "FriendlyName", default="?"),
                        "name": _get(row, "FriendlyName", default="?"),
                        "media": _get(row, "MediaType", default="unknown"),
                        "size_gb": round(float(_get(row, "Size", default=0) or 0) / MB_PER_GB, 1),
                    }
                )
    except Exception as e:
        info["drives_error"] = str(e)[:120]
    for _query in (
        "Get-CimInstance Win32_PageFileSetting | Select-Object Name | ConvertTo-Json",
        "Get-CimInstance Win32_PageFileUsage | Select-Object Name | ConvertTo-Json",
    ):
        _found = _query_pagefile(_query)
        if _found:
            info["pagefile"] = _found
            break
    try:
        mem = mem_free_gb()
        for _key in ("ram_free_gb", "ram_total_gb", "swap_free_gb", "swap_total_gb"):
            if _key in mem:
                info[_key] = round(float(mem[_key]), 1)
    except Exception as e:
        info["swap_error"] = str(e)[:120]
    return info


def _query_pagefile(_query: str) -> str:
    """One pagefile query; empty string when unavailable (caller tries next)."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", _query],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            data = json.loads(out.stdout.strip())
            rows = data if isinstance(data, list) else [data]
            names = [str(r.get("Name", "?")) for r in rows if isinstance(r, dict) and r.get("Name")]
            return ", ".join(names)
    except Exception:
        pass
    return ""


class HostBusyTimeout(Exception):
    """Raised when a foreign model never leaves the endpoint.

    Carries .blocker: model ids still loaded when the wait budget ran out
    (empty when the endpoint state itself was unverifiable).
    """

    def __init__(self, message: str = "", blocker=None):
        super().__init__(message)
        if blocker is None:
            self.blocker = []
        elif isinstance(blocker, str):
            self.blocker = [blocker]
        else:
            self.blocker = list(blocker)


async def ensure_exclusive_access(client, purpose: str = "",
                                  wait_interval_s: float = 60.0,
                                  max_waits: int = 30) -> dict[str, Any]:
    """Wait until the endpoint holds no loaded model; evict foreigners first.

    Lists loaded instances; empty means immediate return. Otherwise each
    blocker model is evicted via client.ensure_unloaded (unload_all
    fallback) and the endpoint is re-checked right away -- a successful
    unload never sleeps through wait_interval_s. Only a still-loaded
    endpoint sleeps, at most max_waits times.

    Returns {free: True, waited_s, attempts}. Raises HostBusyTimeout with
    .blocker model ids after exhausting waits, or when the endpoint state
    cannot be verified at all (fail-closed: never measure blind).

    Test doubles with no loadable endpoint opt out explicitly via
    client._skip_exclusive_check = True (checked with `is True` so plain
    MagicMock auto-attributes never opt out by accident).
    """
    started = time.monotonic()
    skip = getattr(client, "_skip_exclusive_check", False) is True
    list_fn = getattr(client, "get_loaded_instances", None)
    if list_fn is None or not inspect.iscoroutinefunction(list_fn):
        if skip:
            logger.debug("Exclusive-access guard skipped (explicit opt-out)",
                         purpose=purpose)
            return {"free": True, "waited_s": 0.0, "attempts": 0}
        raise HostBusyTimeout(
            f"Host busy check failed before {purpose or 'load'}: "
            "unverifiable: no instance listing.",
            blocker="unverifiable: no instance listing",
        )

    async def _loaded() -> list | None:
        try:
            instances = await client.get_loaded_instances()
        except Exception as e:
            raise HostBusyTimeout(
                f"Host busy check failed before {purpose or 'load'}: "
                f"endpoint state unverifiable ({str(e)[:160]}).",
                blocker=[],
            ) from e
        if not isinstance(instances, list):
            if skip:
                logger.debug("Exclusive-access guard skipped (non-list state)",
                             purpose=purpose)
                return None
            raise HostBusyTimeout(
                f"Host busy check failed before {purpose or 'load'}: "
                "unverifiable: non-list instance state.",
                blocker="unverifiable: non-list instance state",
            )
        return [i for i in instances if isinstance(i, dict) and i.get("instance_id")]

    async def _evict(blockers: list) -> None:
        unload_fn = getattr(client, "ensure_unloaded", None)
        if unload_fn is not None and inspect.iscoroutinefunction(unload_fn):
            for inst in blockers:
                try:
                    await client.ensure_unloaded(inst.get("model"))
                except Exception as e:
                    logger.warning("Evict attempt failed", purpose=purpose,
                                   model=inst.get("model"), error=str(e)[:160])
        else:
            try:
                await client.unload_all()
            except Exception as e:
                logger.warning("Evict unload_all failed", purpose=purpose,
                               error=str(e)[:160])

    attempts = 0
    waits_used = 0
    while True:
        blockers = await _loaded()
        attempts += 1
        if not blockers:
            return {"free": True, "waited_s": time.monotonic() - started,
                    "attempts": attempts}
        await _evict(blockers)
        # Re-check right after the unload; never sleep through a success.
        blockers = await _loaded()
        attempts += 1
        if not blockers:
            return {"free": True, "waited_s": time.monotonic() - started,
                    "attempts": attempts}
        if waits_used >= max_waits:
            ids = [i.get("model") for i in blockers]
            raise HostBusyTimeout(
                f"Host busy before {purpose or 'load'}: still loaded after "
                f"{waits_used} waits: {', '.join(str(m) for m in ids)}. "
                "Unload the model in LM Studio and retry.",
                blocker=ids,
            )
        await asyncio.sleep(wait_interval_s)
        waits_used += 1


async def prepare_host(client, purpose: str = "") -> dict[str, Any]:
    """Preventive safety: unload everything, snapshot free resources, verify empty.

    Returns {gpus, mem, unloaded, verified_empty}. Never raises for monitor
    failures (records them); raises only if stale models cannot be evicted.
    """
    tag = f" [{purpose}]" if purpose else ""
    try:
        await client.unload_all()
    except Exception as e:
        logger.warning("Preventive unload_all failed" + tag, error=str(e))
    snap: dict[str, Any] = {
        "ts": time.strftime("%H:%M:%S"),
        "gpus": gpu_free_mb(),
        "mem": mem_free_gb(),
    }
    try:
        leftovers = await client.get_loaded_instances()
    except Exception as e:
        snap["verified_empty"] = False
        snap["state_error"] = str(e)[:160]
        return snap
    snap["verified_empty"] = not leftovers
    if leftovers:
        snap["leftovers"] = [
            {"model": i.get("model"), "instance_id": i.get("instance_id")} for i in leftovers
        ]
        logger.warning(
            "Models still loaded after preventive unload" + tag, leftovers=snap["leftovers"]
        )
    else:
        logger.info("Host prepared, nothing loaded" + tag, snapshot=snap)
    return snap


def format_snapshot(snap: dict[str, Any]) -> list[str]:
    """Predefined ASCII lines describing a snapshot (for console/.md)."""
    lines = []
    for g in snap.get("gpus", []):
        if "error" in g:
            lines.append(f"GPU: {g['error']}")
        elif g.get("unified_memory"):
            lines.append(f"GPU: {g.get('name', 'unified memory')} (no separate VRAM; see RAM)")
        else:
            lines.append(
                f"GPU {g['id']} {g['name']}: free {g['free_mb']:.0f}/{g['total_mb']:.0f} MB"
            )
    m = snap.get("mem", {}) or {}
    if "unavailable" in m:
        lines.append(f"RAM: {m['unavailable']}")
    elif all(k in m for k in ("ram_free_gb", "ram_total_gb", "swap_free_gb", "swap_total_gb")):
        lines.append(
            f"RAM free: {m['ram_free_gb']:.1f}/{m['ram_total_gb']:.1f} GB, "
            f"swap free: {m['swap_free_gb']:.1f}/{m['swap_total_gb']:.1f} GB"
        )
    else:
        lines.append("RAM: measurements unavailable on this host.")
    if snap.get("verified_empty"):
        lines.append("Loaded models before start: none (verified).")
    elif "leftovers" in snap:
        lines.append(f"WARNING: still loaded: {snap['leftovers']}")
    else:
        lines.append("Loaded-models state could not be verified.")
    return lines
