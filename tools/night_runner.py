"""Night runner: sequential `optimize` over a model queue.

Design goals
- Sequential (LM Studio loads one model at a time; job lock is exclusive).
- Crash-resilient: skips models that already have a `success` run, so after a
  BSOD/reboot you can just relaunch the runner and it continues where it stopped.
- Per-model `--max-context` to bias away from the ctx=32768 TDR/VRAM band that
  BSODs this box.
- Never kills anything, never removes the job lock, never writes to `data/results`.

Usage:  python tools/night_runner.py [--dry-run] [--only MODEL_ID] [--stop-after N]
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

from lm_optimizer.services.run_reconcile import reconcile_stale_runs

REPO = Path(__file__).resolve().parents[1]
DB = REPO / "data" / "optimizer.db"
LOG_DIR = REPO / "data" / "logs" / "night"
STATUS = REPO / "data" / "logs" / "night" / "queue_status.json"

# (model_id, max_context) — ascending by size; small models finish first so the
# night yields results even if a later big model takes the machine down.
QUEUE: list[tuple[str, int]] = [
    ("qwen3.8-9b-distill", 32768),
    ("mistralai/ministral-3-14b-reasoning", 32768),
    ("mirothinker-v1.5-30b-q2ks-mixed-autoround", 16384),
    ("openai-gpt-oss-20b-claude-4.5-opus-heretic-uncensored-i1", 16384),
    ("google/gemma-4-26b-a4b-qat", 16384),
    ("qwen/qwen3.6-35b-a3b", 16384),
]

# Never queued: meta/muse-glimmer (abandoned, <=2 tok/s),
# qwen/qwen3.8-27b (LM Studio guardrail refuses the 17.84 GB load),
# qwen3.8-27b-...-mtp@iq4_xs + @iq3_xs (q4 already probed at 1.14 tok/s vs q3 0.93).


def already_successful(model_id: str) -> bool:
    """True when this model already finished a full successful optimization."""
    if not DB.exists():
        return False
    uri = f"file:{DB.as_posix()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as con:
            row = con.execute(
                "SELECT COUNT(*) FROM runs WHERE model_id = ? AND status = 'success'",
                (model_id,),
            ).fetchone()
        return bool(row and row[0])
    except sqlite3.Error:
        return False


def run_one(model_id: str, max_ctx: int, timeout_s: int) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe = model_id.replace("/", "-").replace(":", "-")
    log = LOG_DIR / f"{safe}__ctx{max_ctx}.log"

    cmd = [
        sys.executable, "-m", "lm_optimizer", "optimize", model_id,
        "--max-context", str(max_ctx),
        "--profile", "balanced",
        "--yes", "--non-interactive",
    ]
    print(f"\n{'=' * 72}\n[night] {datetime.now():%Y-%m-%d %H:%M:%S} START {model_id} "
          f"(max-context={max_ctx})\n{'=' * 72}", flush=True)

    t0 = time.monotonic()
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        fh.write(f"# {datetime.now().isoformat()} {' '.join(cmd)}\n")
        fh.flush()
        try:
            proc = subprocess.run(
                cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                timeout=timeout_s, check=False,
            )
            rc = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            rc = -1
            timed_out = True

    dt = time.monotonic() - t0
    rec = {
        "model_id": model_id, "max_context": max_ctx,
        "returncode": rc, "timed_out": timed_out,
        "seconds": round(dt), "log": str(log),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "success_run": already_successful(model_id),
    }
    rec["reconciled"] = reconcile_dead_run(
        model_id, rc=rc, timed_out=timed_out, success_run=rec["success_run"],
    )

    print(f"[night] DONE {model_id} rc={rc} timed_out={timed_out} "
          f"{dt / 60:.1f} min success={rec['success_run']}", flush=True)
    return rec


def reconcile_dead_run(
    model_id: str, rc: int, timed_out: bool, success_run: bool
) -> dict:
    """Reconcile runs of a model whose optimizer process is known dead.

    The child was hard-killed (timeout) or exited non-zero: it cannot write
    fresh checkpoints any more, so run rows still in ``running``/``resumed``
    belong to a dead process and must not stay ``running`` indefinitely.
    """
    if not timed_out and rc == 0:
        if success_run:
            return {"skipped": "success"}
        return {"skipped": "clean exit without success run"}
    # is_fresh=False is safe: no live process can update the checkpoints.
    return reconcile_stale_runs(
        dry_run=False, model_id=model_id, is_fresh=lambda rid, now: False,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--stop-after", type=int, default=0,
                    help="stop after N attempted models (0 = whole queue)")
    ap.add_argument("--timeout-min", type=int, default=480)
    args = ap.parse_args()

    queue = [(m, c) for m, c in QUEUE if not args.only or m in args.only]
    if not queue:
        print("[night] empty queue")
        return 2

    print(f"[night] {len(queue)} model(s) queued:")
    for m, c in queue:
        done = already_successful(m)
        print(f"   {'SKIP(success)' if done else 'queue':<14} {m}  max-context={c}")
    if args.dry_run:
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({"started": datetime.now().isoformat(),
                                  "queue": [m for m, _ in queue]}, indent=2),
                      encoding="utf-8")

    results: list[dict] = []
    for i, (model_id, max_ctx) in enumerate(queue, 1):
        if already_successful(model_id):
            print(f"[night] ({i}/{len(queue)}) SKIP {model_id} - already success")
            continue
        if args.stop_after and len(results) >= args.stop_after:
            print(f"[night] stop-after reached ({args.stop_after})")
            break
        rec = run_one(model_id, max_ctx, args.timeout_min * 60)
        results.append(rec)
        (STATUS.parent / "queue_results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n[night] queue finished. {len(results)} run(s) attempted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())