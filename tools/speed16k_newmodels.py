"""Speed @16K for the newly downloaded models + resume of the interrupted one.

Ordered fastest-first (by expected class) so partial results stay useful.
Covers the user's three new swift variants (IQ4_XS / IQ3_XS / IQ2_XS) plus
the bartowski Qwen3.8-27B IQ3_XS and the new 40B "flint" model.
"""

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

# (label, model_id) - fastest first.
MODELS = [
    ("swift_iq2xs",   "swift-qwen3.8-27b@iq2_xs"),   # 10.2 GB
    ("swift_iq3xs",   "swift-qwen3.8-27b@iq3_xs"),   # 13.9 GB
    ("qwen38_iq3xs",  "qwen3.8-27b"),                # 13.7 GB (bartowski)
    ("swift_iq4xs",   "swift-qwen3.8-27b@iq4_xs"),   # 16.6 GB
    ("flint_40b",     "recursivesi/flint"),          # 24.2 GB, dense 40B
]

# Interrupted run to continue (has a valid checkpoint).
RESUMES = [
    ("qwen35b_a3b", "d2494c2a-3679-4eb6-accb-9a13a67c5e89"),
]

LOG_DIR = REPO / "data" / "logs" / "speed16k"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _run(cmd, log, timeout_min, label):
    print(f"\n=== {label} ===")
    print(f"cmd: {' '.join(cmd[2:6])} ...")
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
    return rc, dt


def run_speed(label, model, timeout_min=150):
    cmd = [PY, "-m", "lm_optimizer", "optimize", model,
           "--speed-only", "--context", "16384",
           "--yes", "--non-interactive"]
    return _run(cmd, LOG_DIR / f"{label}.log", timeout_min, label)


def run_resume(label, run_id, timeout_min=180):
    cmd = [PY, "-m", "lm_optimizer", "resume", run_id]
    return _run(cmd, LOG_DIR / f"{label}_resume.log", timeout_min, f"{label} (resume)")


def main():
    print("=== Speed @16K: new models (fastest first) ===")
    for label, model in MODELS:
        run_speed(label, model)
    print("\n=== Resuming interrupted runs ===")
    for label, run_id in RESUMES:
        run_resume(label, run_id)
    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
