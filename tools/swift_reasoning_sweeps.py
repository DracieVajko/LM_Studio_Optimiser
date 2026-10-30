"""Reasoning-only sweeps for the three Swift Qwen3.8-27B K-quant variants.

Each run executes the 5-level reasoning battery (off/low/medium/high/xhigh)
on the frozen baseline config at an 8K context, so the comparison isolates
quantization: same model family, same battery, same context - q2_k vs
q3_k_m vs q4_k_s.

Waits for any running optimize job (the speed recheck) to finish first.
"""

import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
DB = REPO / "data" / "optimizer.db"
LOCK = REPO / "data" / "reports" / ".optimizer.lock"

VARIANTS = [
    "swift-qwen3.8-27b@q2_k",
    "swift-qwen3.8-27b@q3_k_m",
    "swift-qwen3.8-27b@q4_k_s",
]

LOG_DIR = REPO / "data" / "logs" / "swift_reasoning"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def other_job_running() -> bool:
    """True while the lock is held by a LIVE process or a run is RUNNING."""
    if LOCK.exists():
        try:
            import json

            holder = json.loads(LOCK.read_text(encoding="utf-8"))
            import psutil

            if psutil.pid_exists(int(holder.get("pid", -1))):
                return True
        except Exception:
            pass
    try:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        n = con.execute(
            "SELECT COUNT(*) FROM runs WHERE status IN ('running','resumed')"
        ).fetchone()[0]
        con.close()
        return bool(n)
    except Exception:
        return True


def run_one(model: str, timeout_min: int = 180) -> int:
    label = model.split("@")[-1]
    log = LOG_DIR / f"{label}.log"
    cmd = [PY, "-m", "lm_optimizer", "optimize", model,
           "--reasoning-only", "--max-context", "8192", "--yes", "--non-interactive"]
    print(f"\n=== reasoning sweep {model} ===")
    print(f"log: {log}")
    t0 = time.monotonic()
    deadline = t0 + timeout_min * 60
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
        while True:
            time.sleep(30)
            rc = p.poll()
            if rc is not None:
                break
            now = time.monotonic()
            if now > deadline:
                print(f"  TIMEOUT after {timeout_min} min, killing")
                p.kill()
                p.wait()
                rc = -1
                break
            try:
                quiet = now - log.stat().st_mtime
            except OSError:
                quiet = 0
            if quiet > 1800:
                print("  STALL (no log output 30 min), killing")
                p.kill()
                p.wait()
                rc = -2
                break
    dt = time.monotonic() - t0
    print(f"  finished in {dt/60:.1f} min, rc={rc}")
    return rc


def main() -> int:
    print("SWIFT_REASONING_QUEUE_START")
    while other_job_running():
        print("waiting for running optimizer job...", flush=True)
        time.sleep(60)
    rcs = []
    for model in VARIANTS:
        rcs.append(run_one(model))
    print(f"SWIFT_REASONING_DONE rcs={rcs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
