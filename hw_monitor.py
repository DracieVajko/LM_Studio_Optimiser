#!/usr/bin/env python3
"""Host VRAM/RAM/swap monitor with optional tok/s probe.

Without arguments: prints a one-shot hardware + LM Studio snapshot and exits.
Loads NOTHING (no client connect, which would probe-load a model).

With --model: loads the model with the given config, runs a short generation,
reports tok/s + VRAM deltas, and always unloads. ASCII output only.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
import urllib.request
from typing import Any

from lm_optimizer.domain.models import LoadConfiguration
from lm_optimizer.services.lm_studio import LMStudioClient

LM_STUDIO_URL = "http://127.0.0.1:1234"
HTTP_TIMEOUT_S = 15
NVSMI_TIMEOUT_S = 15
NVSMI_FIELDS = 5  # index, name, mem.used, mem.total, util.gpu
MB_PER_GB = 1024**3


def gpu_snapshot() -> list[dict[str, Any]]:
    """Snapshot per-GPU used/total MB (empty-error dict if unavailable).

    Uses nvidia-smi directly (GPUtil is unmaintained and broken on py3.12+).
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=NVSMI_TIMEOUT_S,
            check=True,
        )
    except Exception as e:
        return [{"error": f"GPU monitor unavailable: {e}"}]
    else:
        snap = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != NVSMI_FIELDS:
                continue
            idx, name, used, total, util = parts
            snap.append(
                {
                    "id": int(idx),
                    "name": name,
                    "used_mb": float(used),
                    "total_mb": float(total),
                    "load": float(util) / 100.0,
                }
            )
        return snap or [{"error": "nvidia-smi returned no GPUs"}]


def mem_snapshot() -> dict[str, Any]:
    """Return RAM/swap GB dict, or {unavailable} when psutil is missing."""
    try:
        import psutil  # noqa: PLC0415 - lazy import keeps snapshot working without psutil
    except ImportError as e:
        return {"unavailable": f"RAM monitor unavailable: {e}"}
    else:
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        return {
            "ram_used_gb": vm.used / MB_PER_GB,
            "ram_total_gb": vm.total / MB_PER_GB,
            "swap_used_gb": sw.used / MB_PER_GB,
            "swap_total_gb": sw.total / MB_PER_GB,
        }


def lm_studio_loaded(url: str) -> list[dict[str, Any]]:
    """Loaded instances from the server (stdlib only, no client connect)."""
    try:
        with urllib.request.urlopen(url + "/api/v1/models", timeout=HTTP_TIMEOUT_S) as resp:
            data = json.load(resp)
    except Exception as e:
        return [{"error": str(e)[:120]}]
    else:
        return [
            {"model": m.get("key"), "instance_id": inst.get("id")}
            for m in data.get("models", [])
            for inst in m.get("loaded_instances") or []
        ]


def snapshot(url: str) -> dict[str, Any]:
    """One-shot snapshot dict."""
    return {
        "ts": time.strftime("%H:%M:%S"),
        "gpus": gpu_snapshot(),
        "mem": mem_snapshot(),
        "loaded": lm_studio_loaded(url),
    }


def print_snapshot(snap: dict[str, Any]) -> None:
    """Print a snapshot (ASCII only)."""
    print(f"[{snap['ts']}] hardware snapshot")
    for g in snap["gpus"]:
        if "error" in g:
            print(f"  GPU: {g['error']}")
        else:
            pct = 100 * g["used_mb"] / max(g["total_mb"], 1)
            print(
                f"  GPU {g['id']} {g['name']}: "
                f"{g['used_mb']:.0f}/{g['total_mb']:.0f} MB ({pct:.1f}%) "
                f"load {g['load'] * 100:.0f}%"
            )
    m = snap["mem"]
    if "unavailable" in m:
        print(f"  RAM: {m['unavailable']}")
    else:
        print(
            f"  RAM: {m['ram_used_gb']:.1f}/{m['ram_total_gb']:.1f} GB  "
            f"swap: {m['swap_used_gb']:.1f}/{m['swap_total_gb']:.1f} GB"
        )
    loaded = snap["loaded"]
    if loaded and "error" not in loaded[0]:
        for inst in loaded:
            print(f"  loaded: {inst['model']} [{inst['instance_id']}]")
    elif loaded:
        print(f"  loaded: {loaded[0]['error']}")
    else:
        print("  loaded: none")


async def probe(args: argparse.Namespace) -> int:
    """Load model, run short generation with VRAM snapshots, always unload."""
    client = LMStudioClient(base_url=args.url)
    await client.connect()
    exit_code = 0
    try:
        pre = await client.get_loaded_instances()
        if pre:
            print(f"WARNING: {len(pre)} stale instance(s), unloading first")
            await client.unload_all()

        cfg = LoadConfiguration(
            context_length=args.context,
            flash_attention=not args.no_flash,
            offload_kv_cache_to_gpu=not args.kv_cpu,
            eval_batch_size=args.batch,
        )
        print("before load:")
        print_snapshot(snapshot(args.url))

        t0 = time.perf_counter()
        res = await client.load_model(args.model, cfg)
        load_s = time.perf_counter() - t0
        if not res.success:
            print(f"LOAD FAILED: {res.error}")
            exit_code = 1
        else:
            print(f"loaded in {load_s:.1f}s as {res.identifier}")
            print("after load:")
            print_snapshot(snapshot(args.url))

            t0 = time.perf_counter()
            out = await client.chat_completion(
                model=args.model,
                input_text=args.prompt,
                temperature=0.3,
                max_output_tokens=args.max_tokens,
            )
            wall_s = time.perf_counter() - t0
            stats = out.get("_stats", {}) or {}
            tok_s = stats.get("tokens_per_second", 0) or 0
            ttft = stats.get("time_to_first_token_seconds", 0) or 0
            print(
                f"generation: {tok_s:.1f} tok/s, TTFT {ttft * 1000:.0f} ms, "
                f"wall {wall_s:.1f}s, "
                f"output: {out['choices'][0]['message']['content'][:120]!r}"
            )
            print("after generation:")
            print_snapshot(snapshot(args.url))
    finally:
        await client.unload_all()
        leftovers = await client.get_loaded_instances()
        print("after unload:")
        print_snapshot(snapshot(args.url))
        if leftovers:
            print(f"WARNING: leftovers still loaded: {leftovers}")
            exit_code = 1
        await client.close()
    return exit_code


def main(argv: list[str] | None = None) -> int:
    """CLI entry: snapshot by default, probe with --model."""
    ap = argparse.ArgumentParser(description="Host VRAM/RAM monitor + tok/s probe")
    ap.add_argument("--url", default=LM_STUDIO_URL, help="LM Studio base URL")
    ap.add_argument("--model", default=None, help="Model to probe (default: snapshot only)")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--no-flash", action="store_true")
    ap.add_argument("--kv-cpu", action="store_true")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--prompt", default="Say hi in 5 words.")
    ap.add_argument("--max-tokens", type=int, default=30)
    args = ap.parse_args(argv)

    if args.model is None:
        print_snapshot(snapshot(args.url))
        return 0
    return asyncio.run(probe(args))


if __name__ == "__main__":
    sys.exit(main())
