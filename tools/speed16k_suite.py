"""Minimal 16K speed comparison for keep/delete decision.

Runs --speed-only --context 16384 on:
  - swift-qwen3.8-27b           (already has 16 probes from resume, but fresh run = clean)
  - qwen3.8-27b-turbo-fable-... (fable)
  - qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp@iq4_xs (hauhaucs)
  - qwen/qwen3.8-27b            (base)
  - openai/gpt-oss-20b          (orig)
  - openai-gpt-oss-20b-claude-4.5-opus-heretic-uncensored-i1 (heretic)

Each run: ~17 probes x (9 min load + test) = ~3 hours per model.
But the suite runs sequentially with 120 min timeout + 30 min stall.
For speed-only at 16384 with KV quant on swift, the load is the bottleneck.
"""

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

MODELS = [
    ("swift",          "swift-qwen3.8-27b"),
    ("fable",          "qwen3.8-27b-turbo-fable-cold-fusion-735-882-heretic-uncensored-neo-coder-max-mtp"),
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
    print(f"cmd: {' '.join(cmd)}")
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
            # stall check on step log
            try:
                quiet = now - log.stat().st_mtime
            except OSError:
                quiet = 0
            if quiet > 1800:  # 30 min stall
                print(f"  STALL (no log output 30 min), killing")
                p.kill()
                p.wait()
                rc = -2
                break
    dt = time.monotonic() - t0
    print(f"  finished in {dt/60:.1f} min, rc={rc}")
    return rc, dt


def main():
    print("Minimal 16K speed comparison suite")
    print("=" * 60)
    for label, model in MODELS:
        rc, dt = run_one(label, model)
        if rc != 0:
            print(f"  WARNING: {label} exited with {rc}")
    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()