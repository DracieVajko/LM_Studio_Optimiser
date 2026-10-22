#!/usr/bin/env python3
"""Sequential granular-test suite runner (job-lock aware, offline-safe).

Runs one optimizer CLI invocation at a time - LM Studio loads one model at a
time and the job lock is exclusive. Built for the 2026-10-20 27B plan, generic
enough to rerun for future models:

    python tools/granular_suite.py            # full suite
    python tools/granular_suite.py --only 0,2 # subset by step index
    python tools/granular_suite.py --dry-run

Each step logs to `data/logs/granular/<seq>_<name>.log`, failures never stop
the queue, and a `suite_status.json` (night_runner style) is refreshed after
every step. Two hang guards per step:
- total timeout (subprocess hard-kill),
- stall watchdog: no new log output for `--stall-min` minutes (default 20)
  also hard-kills the child. A killed process cannot write fresh checkpoints,
  so immediate reconciliation with is_fresh=False is safe (night_runner
  contract).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "data" / "logs" / "granular"
STATUS = LOG_DIR / "suite_status.json"

FABLE = "qwen3.8-27b-turbo-fable-cold-fusion-735-882-heretic-uncensored-neo-coder-max-mtp"
HAUHAUCS = "qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp@iq4_xs"
BASE27B = "qwen/qwen3.8-27b"
SWIFT = "swift-qwen3.8-27b"
GPTOSS = "openai/gpt-oss-20b"
GPTOSS_HERETIC = "openai-gpt-oss-20b-claude-4.5-opus-heretic-uncensored-i1"

# (name, timeout_min, kind, model_id, granular args tail)
STEPS: list[tuple[str, int, str, str, list[str]]] = [
    # 1. KV-quantized speed re-test of swift (exercises the new --speed-only)
    #    12 GB model @16384 loads in ~9 min -> one full sweep needs hours.
    #    d1ff2bdd already holds 13/17 passed probes from the 75-min timeout;
    #    resume it instead of duplicating the work (granular flags are
    #    persisted in run.advanced_settings).
    ("swift_speed_16384", 120, "resume", SWIFT,
     ["d1ff2bdd-1665-4402-80b1-38a75d19785c"]),
    # 2. reasoning-only implementation validation on a small reasoning model
    #    (first attempt failed with an empty frontier - cold-server baseline
    #    fallback fixed in optimizer; re-queued as a fresh granular run)
    ("gptoss20b_reasoning", 120, "opt", GPTOSS,
     ["--reasoning-only", "--max-context", "16384"]),
    # 3. max-context probe per 27B qwen (TDR-guarded, winner runtime config)
    #    inner timeout raised: every rung pays the same ~9 min slow load.
    ("maxctx_swift", 170, "maxctx", SWIFT, []),
    ("maxctx_fable", 170, "maxctx", FABLE, []),
    ("maxctx_hauhaucs", 170, "maxctx", HAUHAUCS, []),
    ("maxctx_base27b", 170, "maxctx", BASE27B, []),
    # 4. MTP/speculative test on swift (exercises the new --mtp-only)
    ("swift_mtp", 40, "opt", SWIFT, ["--mtp-only", "--context", "16384"]),
    # 5. reasoning-only sweep per 27B qwen (exercises the new --reasoning-only)
    ("swift_reasoning_only", 180, "opt", SWIFT,
     ["--reasoning-only", "--max-context", "16384"]),
    ("fable_reasoning_only", 180, "opt", FABLE,
     ["--reasoning-only", "--max-context", "16384"]),
    ("hauhaucs_reasoning_only", 180, "opt", HAUHAUCS,
     ["--reasoning-only", "--max-context", "16384"]),
    ("base27b_reasoning_only", 180, "opt", BASE27B,
     ["--reasoning-only", "--max-context", "16384"]),
    # 6. quality battery on swift (exercises the new --quality-only)
    #    CPU-bound 27B @16384: one load (~10 min) + ~10 slow tests -> needs margin.
    ("swift_quality", 180, "opt", SWIFT,
     ["--quality-only", "--max-context", "16384"]),
    # 7. GAP-FILL steps (2026-10-20 evening): data missing for the
    #    keep/delete comparison (run with --only 12,13,14,15 after the
    #    main queue). hauhaucs has zero history; fable/heretic never
    #    validated quality. Speed at 4096 matches the other 27Bs' probe
    #    protocol (default frozen speed context) for a fair median axis.
    ("hauhaucs_speed_4096", 75, "opt", HAUHAUCS,
     ["--speed-only", "--context", "4096"]),
    ("hauhaucs_quality_16384", 180, "opt", HAUHAUCS,
     ["--quality-only", "--max-context", "16384"]),
    ("fable_quality_16384", 180, "opt", FABLE,
     ["--quality-only", "--max-context", "16384"]),
    ("heretic_quality_16384", 150, "opt", GPTOSS_HERETIC,
     ["--quality-only", "--max-context", "16384"]),
]


def build_cmd(kind: str, model: str, tail: list[str]) -> list[str]:
    py = sys.executable
    if kind == "opt":
        return [py, "-m", "lm_optimizer", "optimize", model, *tail,
                "--yes", "--non-interactive"]
    if kind == "resume":
        # Resume a dead run from its DB state (granular flags are persisted
        # in run.advanced_settings, so it resumes as the same granular mode).
        return [py, "-m", "lm_optimizer", "resume", *tail]
    if kind == "maxctx":
        return [py, str(REPO / "tools" / "maxctx_probe.py"),
                "--model", model, "--timeout-min", "150"]
    raise ValueError(kind)


def reconcile_model(model_id: str) -> dict:
    """After a hard kill: mark the model's dead run rows interrupted."""
    from lm_optimizer.services.run_reconcile import reconcile_stale_runs

    return reconcile_stale_runs(dry_run=False, model_id=model_id,
                                is_fresh=lambda rid, now: False)


def _quiet_seconds(step_log: Path, since_wall: float,
                   inner_dirs: list[Path]) -> float:
    """Seconds since the last write to the step log or any inner log.

    Must compare against time.time(): file mtimes are wall-clock, and the
    first version mixed in time.monotonic(), producing a huge negative value
    that made the stall watchdog inert (found 2026-10-20). maxctx steps are
    silent in their step log by design (rung results go to
    data/logs/maxctx/*.log), so inner dirs count as liveness too.
    """
    newest = since_wall
    try:
        newest = max(newest, step_log.stat().st_mtime)
    except OSError:
        pass
    for d in inner_dirs:
        try:
            if not d.is_dir():
                continue
            for f in d.glob("*.log"):
                newest = max(newest, f.stat().st_mtime)
        except OSError:
            continue
    return max(0.0, time.time() - newest)


def run_step(seq: int, name: str, timeout_min: int, cmd: list[str],
             model: str, stall_min: int,
             inner_dirs: list[Path] | None = None) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{seq:02d}_{name}.log"
    t0 = time.monotonic()
    t0_wall = time.time()
    inner_dirs = inner_dirs or []
    deadline = t0 + timeout_min * 60
    rec: dict = {"step": seq, "name": name, "model": model,
                 "cmd": " ".join(cmd), "log": str(log)}
    print(f"\n[suite] === {seq:02d} {name} (timeout {timeout_min} min, "
          f"stall {stall_min} min) ===", flush=True)
    timed_out = False
    stalled = False
    rc = None
    try:
        with open(log, "w", encoding="utf-8") as fh:
            p = subprocess.Popen(cmd, cwd=REPO, stdout=fh,
                                 stderr=subprocess.STDOUT)
            while True:
                time.sleep(30)
                rc = p.poll()
                if rc is not None:
                    break
                now = time.monotonic()
                quiet_s = _quiet_seconds(log, t0_wall, inner_dirs)
                if now > deadline:
                    timed_out = True
                    p.kill()
                    p.wait()
                    break
                if quiet_s > stall_min * 60:
                    # No log output for `stall_min` minutes: treat the child
                    # as hung (stuck load/chat) and kill it. The total timeout
                    # is only the backstop. Killed processes cannot write
                    # fresh checkpoints, so reconciling right after is safe.
                    stalled = True
                    p.kill()
                    p.wait()
                    break
        rc = p.returncode
    except Exception as e:  # noqa: BLE001 - queue must survive anything
        rec["error"] = f"{type(e).__name__}: {e}"

    dt = time.monotonic() - t0
    if timed_out or stalled or rc not in (0, None):
        try:
            rec["reconciled"] = reconcile_model(model)
        except Exception as e:  # noqa: BLE001
            rec["reconcile_error"] = str(e)
    rec.update({
        "returncode": rc, "timed_out": timed_out, "stalled": stalled,
        "seconds": round(dt),
        "finished": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"[suite] {name}: rc={rc} timed_out={timed_out} stalled={stalled} "
          f"{dt / 60:.1f} min", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="comma-separated step indices (0-based)")
    ap.add_argument("--stall-min", type=int, default=20,
                    help="kill a step whose log shows no output this long")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    wanted = set(int(x) for x in args.only.split(",")) if args.only else None
    results = []
    if STATUS.exists():
        try:
            results = json.loads(STATUS.read_text(encoding="utf-8")).get("steps", [])
        except Exception:
            results = []

    for seq, (name, timeout_min, kind, model, tail) in enumerate(STEPS):
        if wanted is not None and seq not in wanted:
            continue
        cmd = build_cmd(kind, model, tail)
        if args.dry_run:
            print(f"[dry] {seq:02d} {name}: {' '.join(cmd)}")
            continue
        inner = [REPO / "data" / "logs" / "maxctx"] if kind == "maxctx" else []
        rec = run_step(seq, name, timeout_min, cmd, model, args.stall_min,
                       inner_dirs=inner)
        results = [r for r in results if r.get("name") != name]
        results.append(rec)
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(
            {"updated": datetime.now().isoformat(timespec="seconds"),
             "steps": results},
            indent=2, ensure_ascii=False), encoding="utf-8")

    ok = sum(1 for r in results if r.get("returncode") == 0)
    print(f"\n[suite] finished: {ok}/{len(results)} step(s) clean.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
