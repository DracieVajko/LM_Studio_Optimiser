"""Clean speed test for all Qwen 27B variants + flint at 16K context."""

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

MODELS = [
    ("swift_q2_k",   "swift-qwen3.8-27b@q2_k"),
    ("swift_q3_k_m", "swift-qwen3.8-27b@q3_k_m"),
    ("swift_q4_k_s", "swift-qwen3.8-27b@q4_k_s"),
    ("qwen_9b",      "qwen3.8-9b-distill"),
    ("flint",        "recursivesi/flint@q4_k_m"),
]

LOG_DIR = REPO / "data" / "logs" / "speed16k_clean"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_one(label: str, model: str, timeout_min: int = 180) -> tuple[int, float]:
    log = LOG_DIR / f"{label}.log"
    cmd = [PY, "-m", "lm_optimizer", "optimize", model,
           "--speed-only", "--context", "16384",
           "--min-generation-tok-s", "0.2",
           "--no-enforce-min-generation-speed",
           "--yes", "--non-interactive"]
    print(f"\n=== {label} ({model}) ===")
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
                print(f"  STALL (no log output 30 min), killing")
                p.kill()
                p.wait()
                rc = -2
                break
    dt = time.monotonic() - t0
    print(f"  finished in {dt/60:.1f} min, rc={rc}")
    return rc, dt


def main():
    print("=== Fresh speed test: Qwen 27B variants + Flint @16K ===")
    for label, model in MODELS:
        run_one(label, model)
    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()