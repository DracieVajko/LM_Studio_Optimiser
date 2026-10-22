"""Run only the bartowski Qwen3.8-27B IQ3_XS @16K (skipping flint 40B)."""

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

LOG_DIR = REPO / "data" / "logs" / "speed16k"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log = LOG_DIR / "bartowski_iq3xs.log"
cmd = [PY, "-m", "lm_optimizer", "optimize", "qwen3.8-27b",
       "--speed-only", "--context", "16384",
       "--yes", "--non-interactive"]
print(f"log: {log}")
t0 = time.monotonic()
deadline = t0 + 180 * 60
with open(log, "w", encoding="utf-8") as fh:
    p = subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    while True:
        time.sleep(30)
        rc = p.poll()
        if rc is not None:
            break
        now = time.monotonic()
        if now > deadline:
            print("TIMEOUT, killing")
            p.kill()
            p.wait()
            rc = -1
            break
        try:
            quiet = now - log.stat().st_mtime
        except OSError:
            quiet = 0
        if quiet > 1800:
            print("STALL, killing")
            p.kill()
            p.wait()
            rc = -2
            break
    dt = time.monotonic() - t0
    print(f"finished in {dt/60:.1f} min, rc={rc}")