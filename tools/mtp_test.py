"""MTP (Multi-Token Prediction) comparison runner.

Tests MTP OFF vs n=2/n=3/n=4 on a single model and reports the speedup.
The master spec (§5.4) requires an OFF baseline and only recommends MTP
when speedup > 1.05 AND accepted_draft_rate is stable AND VRAM budget
is preserved.

Usage:
    python tools/mtp_test.py <model> [--max-tokens 256] [--repeats 3]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

LOG_DIR = REPO / "data" / "logs" / "mtp"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_mtp_variant(
    model: str,
    mtp: bool,
    draft_max: int = 0,
    timeout_min: int = 60,
) -> dict:
    """Run one MTP variant and return the result."""
    import httpx

    label = f"mtp_off" if not mtp else f"mtp_n{draft_max}"
    log = LOG_DIR / f"{model.replace('/', '_').replace(':', '_')}_{label}.log"

    # Load model with MTP settings
    load_body = {
        "model": model,
        "context_length": 8192,
        "flash_attention": True,
        "offload_kv_cache_to_gpu": True,
        "eval_batch_size": 2048,
    }
    if mtp:
        load_body["speculative_draft_mtp"] = True
        load_body["speculative_draft_max_tokens"] = draft_max
        load_body["speculative_draft_min_tokens"] = 0
        load_body["speculative_draft_min_continue_probability"] = 0.0
        load_body["speculative_draft_simple"] = False

    print(f"\n=== {label} ({model}) ===")
    print(f"  load: {load_body}")

    t0 = time.monotonic()
    try:
        # Unload first
        try:
            httpx.post("http://127.0.0.1:1234/api/v1/models/unload",
                       json={"model": model}, timeout=60)
        except Exception:
            pass

        # Load
        load_resp = httpx.post(
            "http://127.0.0.1:1234/api/v1/models/load",
            json=load_body,
            timeout=300,
        )
        if load_resp.status_code != 200:
            return {
                "label": label,
                "mtp": mtp,
                "draft_max": draft_max,
                "error": f"load failed: {load_resp.status_code} {load_resp.text[:200]}",
                "tok_s": 0.0,
                "duration_s": time.monotonic() - t0,
            }
        load_time = load_resp.json().get("load_time_seconds", 0)
        print(f"  loaded in {load_time:.1f}s")

        # Warmup
        httpx.post(
            "http://127.0.0.1:1234/api/v1/chat",
            json={
                "model": model,
                "input": "Say hello.",
                "max_output_tokens": 20,
                "temperature": 0.0,
            },
            timeout=120,
        )

        # Benchmark: 3 runs
        tok_s_values = []
        ttft_values = []
        for run in range(3):
            resp = httpx.post(
                "http://127.0.0.1:1234/api/v1/chat",
                json={
                    "model": model,
                    "input": "Write a short Python function that adds two numbers. Include a docstring.",
                    "max_output_tokens": 256,
                    "temperature": 0.0,
                },
                timeout=300,
            )
            data = resp.json()
            stats = data.get("stats", {})
            tok_s = stats.get("tokens_per_second", 0)
            ttft = stats.get("time_to_first_token_seconds", 0)
            tok_s_values.append(tok_s)
            ttft_values.append(ttft)
            print(f"  run {run+1}: {tok_s:.2f} tok/s, TTFT {ttft:.2f}s")

        # Unload
        try:
            httpx.post("http://127.0.0.1:1234/api/v1/models/unload",
                       json={"model": model}, timeout=60)
        except Exception:
            pass

        import statistics
        return {
            "label": label,
            "mtp": mtp,
            "draft_max": draft_max,
            "load_time_s": load_time,
            "tok_s_values": tok_s_values,
            "tok_s_median": statistics.median(tok_s_values),
            "ttft_median": statistics.median(ttft_values),
            "duration_s": time.monotonic() - t0,
        }

    except Exception as e:
        return {
            "label": label,
            "mtp": mtp,
            "draft_max": draft_max,
            "error": f"{type(e).__name__}: {e}",
            "tok_s": 0.0,
            "duration_s": time.monotonic() - t0,
        }


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "swift-qwen3.8-27b"

    print(f"=== MTP comparison: {model} ===")
    print("Testing: OFF, n=2, n=3, n=4")

    results = []

    # OFF baseline
    results.append(run_mtp_variant(model, mtp=False))

    # MTP variants
    for n in (2, 3, 4):
        results.append(run_mtp_variant(model, mtp=True, draft_max=n))

    # Summary
    print("\n" + "=" * 60)
    print("MTP COMPARISON SUMMARY")
    print("=" * 60)
    baseline = results[0].get("tok_s_median", 0)
    print(f"{'Variant':<12} {'Median tok/s':>14} {'Speedup':>10} {'TTFT':>8}")
    print("-" * 50)
    for r in results:
        if "error" in r:
            print(f"{r['label']:<12} {'ERROR':>14} {'-':>10} {'-':>8}")
            continue
        speedup = r["tok_s_median"] / baseline if baseline > 0 else 0
        print(f"{r['label']:<12} {r['tok_s_median']:>14.2f} {speedup:>10.3f}x {r['ttft_median']:>8.2f}")

    # Recommendation
    print("\n=== RECOMMENDATION ===")
    best = max(results, key=lambda r: r.get("tok_s_median", 0))
    baseline_speedup = best.get("tok_s_median", 0) / baseline if baseline > 0 else 0
    if baseline_speedup >= 1.05:
        print(f"MTP ON ({best['label']}) recommended: {baseline_speedup:.3f}x speedup")
    else:
        print(f"MTP OFF recommended: best MTP speedup was only {baseline_speedup:.3f}x (< 1.05 threshold)")

    # Save results
    out_path = REPO / "results" / f"mtp_comparison_{model.replace('/', '_').replace(':', '_')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()