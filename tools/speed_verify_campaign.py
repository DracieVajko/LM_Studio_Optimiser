"""Speed campaign + implementation verification.

Phase A: speed-only @4096 with `--preset quick` for every model except
         qwen3.8-9b-distill (excluded by request) and flint (unloadable).
         Exercises: preset resolution, probe cap (12), speed-only skips,
         resolved_settings persistence, model key/variant resolution.
Phase B: three synthetic edge cases on the fast bonsai model:
         B1 high floor enforced  -> explicit NO_CONFIG_MEETS_SPEED_FLOOR
         B2 high floor disabled  -> informational, no forced no-winner reason
         B3 custom --stage speed -> custom stage list honoured
"""

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

LOG_DIR = REPO / "data" / "logs" / "speed_verify"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PHASE_A = [
    ("swift_q2_k",      "swift-qwen3.8-27b@q2_k"),
    ("swift_q3_k_m",    "swift-qwen3.8-27b@q3_k_m"),
    ("swift_q4_k_s",    "swift-qwen3.8-27b@q4_k_s"),
    ("qwen_coder",      "qwen/qwen3-coder-30b"),
    ("bonsai",          "prism-ml/bonsai-27b"),
    ("gemma_26b",       "google/gemma-4-26b-a4b-qat"),
    ("mirothinker",     "mirothinker-v1.5-30b-q2ks-mixed-autoround"),
    ("qwen35_moe",      "qwen/qwen3.6-35b-a3b"),
    ("gpt_oss_heretic", "openai-gpt-oss-20b-claude-4.5-opus-heretic-uncensored-i1"),
]

BASE = [PY, "-m", "lm_optimizer", "optimize"]

PHASE_B = [
    (
        "B1_floor_enforced_high",
        BASE + ["prism-ml/bonsai-27b", "--preset", "quick", "--speed-only",
                "--context", "4096", "--min-generation-tok-s", "999",
                "--yes", "--non-interactive"],
    ),
    (
        "B2_floor_disabled_high",
        BASE + ["prism-ml/bonsai-27b", "--preset", "quick", "--speed-only",
                "--context", "4096", "--min-generation-tok-s", "999",
                "--no-enforce-min-generation-speed",
                "--yes", "--non-interactive"],
    ),
    (
        "B3_custom_stage_speed_cap5",
        BASE + ["prism-ml/bonsai-27b", "--preset", "custom",
                "--stage", "speed", "--speed-only", "--context", "4096",
                "--max-speed-probes", "5", "--benchmark-repetitions", "2",
                "--min-generation-tok-s", "0.2",
                "--no-enforce-min-generation-speed",
                "--yes", "--non-interactive"],
    ),
]


def run_one(label: str, cmd: list[str], timeout_min: int) -> tuple[int, float]:
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
    print(f"CAMPAIGN_STARTED {started:.0f}")
    print("=== Phase A: speed @4096 (preset quick) ===")
    for label, model in PHASE_A:
        run_one(
            label,
            BASE + [model, "--preset", "quick", "--speed-only",
                    "--context", "4096",
                    "--min-generation-tok-s", "0.2",
                    "--no-enforce-min-generation-speed",
                    "--yes", "--non-interactive"],
            timeout_min=90,
        )

    print("\n=== Phase B: implementation edge cases (bonsai) ===")
    for label, cmd in PHASE_B:
        run_one(label, cmd, timeout_min=45)

    print(f"\nCAMPAIGN_FINISHED {time.time():.0f} started {started:.0f}")
    print("=== ALL DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
