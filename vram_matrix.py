#!/usr/bin/env python3
"""6GB VRAM fit/speed matrix: context x flash-attention x KV placement.

Ultra-short generations ("Say hi in 5 words.", max 30 tokens): measures
fit/speed, NOT quality. Always unloads; warns if anything is left loaded.
ASCII output only. Core logic lives in lm_optimizer.services.matrix.
"""

import argparse
import asyncio
import sys

from lm_optimizer.services.hostguard import format_snapshot, prepare_host
from lm_optimizer.services.lm_studio import LMStudioClient
from lm_optimizer.services.matrix import (
    DEFAULT_CONTEXTS,
    MAX_TOKENS,
    PROMPT,
    build_matrix,
    fmt_cfg,
    recommendation_line,
    run_one,
)


async def amain(args: argparse.Namespace) -> int:
    client = LMStudioClient(base_url=args.url)
    await client.connect()
    try:
        snap = await prepare_host(client, purpose="vram_matrix")
        for line in format_snapshot(snap):
            print("  " + line)
        if not snap.get("verified_empty") and snap.get("leftovers"):
            print("Stale models loaded, aborting. Unload them first.")
            return 1
        cfgs = build_matrix(args.contexts, args.flash, args.kv)
        results = []
        for i, cfg in enumerate(cfgs, 1):
            print(
                f"[{i}/{len(cfgs)}] ctx={cfg.context_length} "
                f"flash={cfg.flash_attention} kv-gpu={cfg.offload_kv_cache_to_gpu} ...",
                flush=True,
            )
            r = await run_one(client, args.model, cfg)
            results.append(r)
            status = "OK " if r["ok"] else "FAIL"
            print(
                f"         {status} {r['tok_s']:.1f} tok/s, "
                f"TTFT {r['ttft_ms']:.0f} ms, load {r['load_s']:.1f}s {r['error']}"
            )

        print(f"\nVRAM matrix for {args.model} (prompt: {PROMPT!r}, max {MAX_TOKENS})")
        print(
            f"{'ctx':<6} {'flash':<5} {'kv':<4} {'status':<6} "
            f"{'tok/s':>7} {'TTFTms':>7} {'loads':>6}"
        )
        for r in results:
            print(
                f"{r['ctx']:<6} {r['flash']!s:<5} {r['kv']:<4} "
                f"{'OK' if r['ok'] else 'FAIL':<6} {r['tok_s']:>7.1f} "
                f"{r['ttft_ms']:>7.0f} {r['load_s']:>6.1f}"
            )

        print("\n" + recommendation_line(results))
        return 0 if all(r["ok"] for r in results) else 1
    finally:
        await client.close()


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="6GB VRAM fit/speed matrix")
    ap.add_argument("model", nargs="?", default="qwen3.5-4b", help="Model key to test")
    ap.add_argument("--url", default="http://127.0.0.1:1234")
    ap.add_argument("--contexts", type=int, nargs="+", default=list(DEFAULT_CONTEXTS))
    ap.add_argument(
        "--flash",
        choices=["on", "off"],
        nargs="*",
        default=None,
        help="Restrict flash options (default: both)",
    )
    ap.add_argument(
        "--kv",
        choices=["gpu", "cpu"],
        nargs="*",
        default=None,
        help="Restrict KV placement (default: both)",
    )
    ap.add_argument("--dry-run", action="store_true", help="List configs without loading")
    args = ap.parse_args(argv)
    args.flash = None if not args.flash else [v == "on" for v in args.flash]
    args.kv = None if not args.kv else [v == "gpu" for v in args.kv]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    if args.dry_run:
        for cfg in build_matrix(args.contexts, args.flash, args.kv):
            print(fmt_cfg(cfg))
        return 0
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
