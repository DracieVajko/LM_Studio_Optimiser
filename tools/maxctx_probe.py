"""Max-context probe: escalate each model's context to its real limit, with quality.

Answers "how much context can this model actually hold before speed or quality
collapses" by reusing the `ctx` command with the WINNING runtime config from
each model's optimize run (so the sweep measures the tuned config, not a
generic one).

`ctx` already:
  - clamps to the model's advertised context_limit,
  - asks `lms estimate` for a memory estimate BEFORE every load and refuses to
    probe a rung the server would reject (this is the TDR/BSOD guard),
  - bisects with step 1000 around the refusal boundary.

We add: quality suites on the top-N stable rungs, plus a KV-on-CPU pass for
the high end (KV on CPU removes the VRAM pressure that triggers TDR).

Read-only against the DB; writes only under data/logs/maxctx/.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DB = REPO / "data" / "optimizer.db"
LOG_DIR = REPO / "data" / "logs" / "maxctx"

# (gpu_ratio, eval_batch_size, flash_attention, offload_kv_cache_to_gpu) come
# from the optimize winner; these are only fallbacks for models with no winner.
FALLBACK_RUNTIME = {
    "gpu_ratio": 0.8,
    "eval_batch_size": 2048,
    "flash_attention": True,
    "offload_kv_cache_to_gpu": True,
}


def winner_runtime(model_id: str) -> tuple[dict, str]:
    """(runtime kwargs, source run id) from the newest run that has a winner."""
    if not DB.exists():
        return dict(FALLBACK_RUNTIME), ""
    try:
        with sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True) as con:
            con.row_factory = sqlite3.Row
            run = con.execute(
                "SELECT id, best_config_id FROM runs "
                "WHERE model_id = ? AND best_config_id IS NOT NULL "
                "AND status IN ('success', 'partial_success') "
                "ORDER BY started_at DESC LIMIT 1",
                (model_id,),
            ).fetchone()
            if not run:
                return dict(FALLBACK_RUNTIME), ""
            cfg_row = con.execute(
                "SELECT config_json FROM configurations WHERE id = ?",
                (run["best_config_id"],),
            ).fetchone()
    except sqlite3.Error:
        return dict(FALLBACK_RUNTIME), ""
    if not cfg_row or not cfg_row["config_json"]:
        return dict(FALLBACK_RUNTIME), run["id"]
    cfg = json.loads(cfg_row["config_json"])
    rt = dict(FALLBACK_RUNTIME)
    for k in rt:
        if cfg.get(k) is not None:
            rt[k] = cfg[k]
    return rt, run["id"]


def model_limit(model_id: str) -> int:
    """Advertised context_limit from the DB models table (262144 for qwen/mistral)."""
    try:
        with sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True) as con:
            row = con.execute(
                "SELECT context_limit FROM models WHERE id = ?", (model_id,)
            ).fetchone()
            if row and row[0]:
                return int(row[0])
    except sqlite3.Error:
        pass
    return 65536


def run_ctx(model_id: str, kv_gpu: bool, quality_rungs: int, reps: int, timeout_min: int):
    rt, src_run = winner_runtime(model_id)
    limit = model_limit(model_id)
    safe = model_id.replace("/", "-").replace(":", "-")
    kv_tag = "kvgpu" if kv_gpu else "kvcpu"
    log = LOG_DIR / f"{safe}__{kv_tag}.log"

    cmd = [
        sys.executable, "-m", "lm_optimizer", "ctx", model_id,
        "--max-context", str(limit),
        "--gpu-ratio", str(rt["gpu_ratio"]),
        "--batch", str(rt["eval_batch_size"]),
        "--quality-rungs", str(quality_rungs),
        "--repetitions", str(reps),
        "--warmup-runs", "1",
        "--quality-threshold", "0.97",
    ]
    cmd.append("--kv-gpu" if kv_gpu else "--kv-cpu")
    cmd.append("--flash" if rt["flash_attention"] else "--no-flash")

    print(f"\n{'=' * 72}\n[{datetime.now():%H:%M:%S}] ctx {model_id} "
          f"limit={limit} kv={'GPU' if kv_gpu else 'CPU'} "
          f"(winner from {(src_run or 'fallback')[:8]})\n{'=' * 72}", flush=True)

    t0 = time.monotonic()
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        fh.write(f"# {datetime.now().isoformat()} {' '.join(cmd)}\n")
        fh.flush()
        try:
            p = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                               timeout=timeout_min * 60, check=False)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = -1
    dt = time.monotonic() - t0
    print(f"[{datetime.now():%H:%M:%S}] done rc={rc} in {dt / 60:.1f} min -> {log}",
          flush=True)
    return {"model_id": model_id, "kv_gpu": kv_gpu, "rc": rc,
            "limit": limit, "minutes": round(dt), "log": str(log)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="restrict to model id")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quality-rungs", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout-min", type=int, default=90)
    ap.add_argument("--kv", choices=["gpu", "cpu", "both"], default="both")
    args = ap.parse_args()

    models = args.model or [
        "prism-ml/bonsai-27b",
        "qwen3.8-9b-distill",
        "mistralai/ministral-3-14b-reasoning",
        "qwen/qwen3-coder-30b",
        "openai/gpt-oss-20b",
        "mirothinker-v1.5-30b-q2ks-mixed-autoround",
    ]

    print(f"[maxctx] {len(models)} model(s), limit from DB context_limit")
    for m in models:
        rt, src = winner_runtime(m)
        print(f"   {m:<48} limit={model_limit(m):<7} "
              f"gpu={rt['gpu_ratio']} batch={rt['eval_batch_size']} "
              f"kv={'gpu' if rt['offload_kv_cache_to_gpu'] else 'cpu'} "
              f"(winner {(src or 'fallback')[:8]})")
    if args.dry_run:
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    kv_modes = [True, False] if args.kv == "both" else [args.kv == "gpu"]
    results = []
    for m in models:
        for kv in kv_modes:
            results.append(
                run_ctx(m, kv, args.quality_rungs, args.reps, args.timeout_min)
            )
            (LOG_DIR / "results.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )
    print(f"\n[maxctx] finished {len(results)} sweep(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())