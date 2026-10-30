"""Final pass:
- flint at 4096 (worked once) and 8192 (scale check)
- reasoning sweep on the remaining Swift variants (q2 already done, q3 dropped)
"""

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG_DIR = REPO / "data" / "logs" / "final_pass"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_one(label: str, cmd: list[str], timeout_min: int = 120) -> tuple[int, float]:
    log = LOG_DIR / f"{label}.log"
    print(f"\n=== {label} ===")
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


def main() -> int:
    # Flint: try higher context now that 4096 works; 8192 may still fit with KV on GPU
    run_one("flint_4096", [PY, "-m", "lm_optimizer", "optimize", "recursivesi/flint",
                          "--speed-only", "--context", "4096",
                          "--min-generation-tok-s", "0.2", "--no-enforce-min-generation-speed",
                          "--yes", "--non-interactive"], timeout_min=60)
    run_one("flint_8192", [PY, "-m", "lm_optimizer", "optimize", "recursivesi/flint",
                          "--speed-only", "--context", "8192",
                          "--min-generation-tok-s", "0.2", "--no-enforce-min-generation-speed",
                          "--yes", "--non-interactive"], timeout_min=60)
    # Reasoning sweep on the remaining Swift variant (q2 already done, q3 dropped)
    run_one("swift_q4_reasoning", [PY, "-m", "lm_optimizer", "optimize",
                          "swift-qwen3.8-27b@q4_k_s", "--reasoning-only",
                          "--max-context", "8192", "--yes", "--non-interactive"],
            timeout_min=240)
    print("\n=== ALL DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
