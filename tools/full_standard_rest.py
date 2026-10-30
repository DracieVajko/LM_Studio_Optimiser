"""Complete optimizer runs (standard preset, no sandbox) for every model
except the Qwen 27B variants and Flint (handled afterwards by priority).

Standard preset = speed -> quality -> recovery -> final validation.
Context sweep is opt-in (off) and the final sandbox is off by default.
"""

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

MODELS = [
    ("qwen_9b",         "qwen3.8-9b-distill"),
    ("qwen_coder",      "qwen/qwen3-coder-30b"),
    ("bonsai",          "prism-ml/bonsai-27b"),
    ("gemma_26b",       "google/gemma-4-26b-a4b-qat"),
    ("mirothinker",     "mirothinker-v1.5-30b-q2ks-mixed-autoround"),
    ("qwen35_moe",      "qwen/qwen3.6-35b-a3b"),
    ("gpt_oss_heretic", "openai-gpt-oss-20b-claude-4.5-opus-heretic-uncensored-i1"),
]

LOG_DIR = REPO / "data" / "logs" / "full_standard"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_one(label: str, model: str, timeout_min: int = 300) -> tuple[int, float]:
    log = LOG_DIR / f"{label}.log"
    cmd = [PY, "-m", "lm_optimizer", "optimize", model,
           "--preset", "standard",
           # Quality context follows --max-context; 8192 is the engine's
           # documented default. The CLI default (32768) made the quality
           # suite 4x slower on a 6 GB GPU with no additional evidence value.
           "--max-context", "8192",
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
                print("  STALL (no log output 30 min), killing")
                p.kill()
                p.wait()
                rc = -2
                break
    dt = time.monotonic() - t0
    print(f"  finished in {dt/60:.1f} min, rc={rc}")
    return rc, dt


def main():
    print("=== Complete standard runs (no sandbox): all models except qwen27b/flint ===")
    for label, model in MODELS:
        run_one(label, model)
    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
