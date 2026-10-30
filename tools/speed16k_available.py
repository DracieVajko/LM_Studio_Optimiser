"""Speed @16K for all models currently available in LM Studio.

Ordered fastest-first (by expected class) so partial results stay useful
if the session is interrupted. Already-measured models are skipped.
"""

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

# (label, model_id) - ordered by expected speed, fastest first.
MODELS = [
    ("qwen9b",        "qwen3.8-9b-distill"),                       # 9B Q4_K_M, 5.8 GB
    ("mirothinker",   "mirothinker-v1.5-30b-q2ks-mixed-autoround"), # 30B Q2_K_S, 10.7 GB
    ("gemma4_26b",    "google/gemma-4-26b-a4b-qat"),                # MoE 26B-A4B, 15.6 GB
    ("qwen_coder30b", "qwen/qwen3-coder-30b"),                      # MoE 30B-A3B, 18.6 GB
    ("qwen35b_a3b",   "qwen/qwen3.6-35b-a3b"),                     # MoE 35B-A3B, 22.1 GB
]

LOG_DIR = REPO / "data" / "logs" / "speed16k"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_one(label: str, model: str, timeout_min: int = 150) -> tuple[int, float]:
    log = LOG_DIR / f"{label}.log"
    cmd = [PY, "-m", "lm_optimizer", "optimize", model,
           "--speed-only", "--context", "16384",
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
    print("=== Speed @16K for available models (fastest first) ===")
    for label, model in MODELS:
        run_one(label, model)
    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
