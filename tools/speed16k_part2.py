"""Resume fable + run remaining 4 models at 16K."""

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

# fable resume (run 5683a335 was interrupted at 3/17 probes)
FABLE_RESUME = "5683a335-8651-4e41-9bbc-5ca4c6a31cb0"

REMAINING = [
    ("hauhaucs",       "qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp@iq4_xs"),
    ("base27b",        "qwen/qwen3.8-27b"),
    ("gptoss_orig",    "openai/gpt-oss-20b"),
    ("gptoss_heretic", "openai-gpt-oss-20b-claude-4.5-opus-heretic-uncensored-i1"),
]

LOG_DIR = REPO / "data" / "logs" / "speed16k"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_one(label: str, model: str, timeout_min: int = 180) -> tuple[int, float]:
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


def run_resume(run_id: str, label: str, timeout_min: int = 180) -> tuple[int, float]:
    log = LOG_DIR / f"{label}_resume.log"
    cmd = [PY, "-m", "lm_optimizer", "resume", run_id]
    print(f"\n=== RESUME {label} ({run_id[:8]}) ===")
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
    print("=== Resume fable + remaining 4 models @16K ===")
    
    # 1. Resume fable
    run_resume(FABLE_RESUME, "fable")
    
    # 2. Remaining models
    for label, model in REMAINING:
        run_one(label, model)
    
    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()