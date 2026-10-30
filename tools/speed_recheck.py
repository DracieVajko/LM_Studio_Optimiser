"""Re-check suspicious speed numbers with identical settings.

gpt-oss-20b heretic and qwen3.8-9b measured 3.50 / 11.63 (median) at 16K but
10.50 / (no clean 4K run) at 4096 - both contexts are now measured again with
the CURRENT LM Studio settings (KV cache on, 8 threads) and the same
`--preset quick --speed-only` path, so the delta is apples-to-apples.

flint runs last: its earlier load failed with HTTP 500 "insufficient system
resources"; we retry 4096 then 2048 so the small-context case is at least tried.
"""

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

BASE = [PY, "-m", "lm_optimizer", "optimize"]
COMMON = ["--preset", "quick", "--speed-only",
          "--min-generation-tok-s", "0.2",
          "--no-enforce-min-generation-speed",
          "--yes", "--non-interactive"]

JOBS = [
    # gpt-oss already re-measured at both contexts this morning (8.38 / 11.30)
    ("qwen_9b",         "qwen3.8-9b-distill",                                        [16384]),
    ("flint_4k",        "recursivesi/flint",                                         [4096]),
    ("flint_2k",        "recursivesi/flint",                                         [2048]),
]

LOG_DIR = REPO / "data" / "logs" / "speed_recheck"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_one(label: str, cmd: list[str], timeout_min: int = 150) -> tuple[int, float]:
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
    started = time.time()
    print(f"RECHECK_STARTED {started:.0f}")
    for label, model, contexts in JOBS:
        for ctx in contexts:
            run_one(f"{label}_ctx{ctx}",
                    BASE + [model, "--context", str(ctx)] + COMMON,
                    timeout_min=150)
    print(f"RECHECK_FINISHED {time.time():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
