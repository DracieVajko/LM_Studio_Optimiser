"""Live end-to-end validation of the new workflows on one real model.

Runs each workflow that until now only had offline tests:
  1. context-explorer (1.5x bracketing + retrieval)
  2. quality-lab (deterministic tasks + hard gates)
  3. duel (controlled, blind)
  4. optimize --test-only (test-name selection)
  5. optimize --quality-lab (Quality Lab inside the optimizer)
  6. mtp_test (OFF baseline vs n=2/3/4)  [only when the model has MTP]
  7. sandbox web task (static validation + repair loop)
  8. full-optimize (orchestration, context+quality skipped to stay short)

Usage:
    python tools/live_validation.py <model> [--budget short|full]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG_DIR = REPO / "data" / "logs" / "live_validation"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run(label: str, cmd: list[str], timeout_min: int) -> dict:
    log = LOG_DIR / f"{label}.log"
    print(f"\n=== {label} ===")
    print(f"  {' '.join(cmd[1:8])} ...")
    started = time.monotonic()
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=timeout_min * 60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            rc = -1
    duration = time.monotonic() - started
    ok = rc == 0
    print(f"  {'OK' if ok else 'FAIL'} rc={rc} in {duration/60:.1f} min -> {log.name}")
    return {"label": label, "rc": rc, "ok": ok, "minutes": round(duration / 60, 1),
            "log": str(log)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model id to validate the workflows on")
    parser.add_argument("--budget", choices=("short", "full"), default="short")
    parser.add_argument("--skip-sandbox", action="store_true")
    parser.add_argument("--skip-duel", action="store_true")
    args = parser.parse_args()

    model = args.model
    short = args.budget == "short"
    ctx = 8192 if short else 16384
    results: list[dict] = []

    # 1. Context explorer: 1.5x bracketing + fill ratios + retrieval.
    #    Under a loaded host the absolute tok/s is low; a short-budget run
    #    lowers the interactive floor so the fill/retrieval paths still run.
    floor = "0.2" if short else "1.0"
    results.append(
        run(
            "context_explorer",
            [PY, "-m", "lm_optimizer", "lab", "context-explorer", model,
             "--min-context", "4096", "--max-context", str(ctx),
             "--repetitions", "1", "--min-tps", floor],
            timeout_min=30 if short else 90,
        )
    )

    # 2. Quality Lab: two seeds keep it short but still exercise reliability.
    results.append(
        run(
            "quality_lab",
            [PY, "-m", "lm_optimizer", "lab", "quality-lab", model,
             "--seeds", "2", "--context", str(ctx)],
            timeout_min=30 if short else 90,
        )
    )

    # 3. Duel: controlled micro-quality, blind.
    if not args.skip_duel:
        results.append(
            run(
                "duel",
                [PY, "-m", "lm_optimizer", "lab", "duel", model, model,
                 "--type", "micro_quality", "--context", str(ctx)],
                timeout_min=20 if short else 60,
            )
        )

    # 4. test-only selection: exactly one test, no speed phase.
    results.append(
        run(
            "test_only",
            [PY, "-m", "lm_optimizer", "optimize", model,
             "--test-only", "--test", "short_instruction",
             "--max-context", str(ctx), "--yes", "--non-interactive"],
            timeout_min=20 if short else 60,
        )
    )

    # 5. Quality Lab inside the optimizer (needs the quality phase).
    results.append(
        run(
            "optimize_quality_lab",
            [PY, "-m", "lm_optimizer", "optimize", model,
             "--quality-only", "--quality-lab", "--max-context", str(ctx),
             "--yes", "--non-interactive"],
            timeout_min=40 if short else 120,
        )
    )

    # 6. MTP comparison (only meaningful when the model ships an MTP head).
    results.append(
        run(
            "mtp_test",
            [PY, str(REPO / "tools" / "mtp_test.py"), model],
            timeout_min=30 if short else 90,
        )
    )

    # 7. Sandbox web task (static validation + repair loop).
    if not args.skip_sandbox:
        results.append(
            run(
                "sandbox_web",
                [PY, "-m", "lm_optimizer", "lab", "sandbox",
                 "--task", "web_dashboard", "--model", model],
                timeout_min=20 if short else 60,
            )
        )

    # 8. Full optimization orchestration (quality + context stages included).
    results.append(
        run(
            "full_optimize",
            [PY, "-m", "lm_optimizer", "lab", "full-optimize", model,
             "--max-context", str(ctx), "--quality-threshold", "0.9"],
            timeout_min=60 if short else 240,
        )
    )

    # Summary
    print("\n" + "=" * 64)
    print("LIVE VALIDATION SUMMARY")
    print("=" * 64)
    ok = sum(1 for r in results if r["ok"])
    for result in results:
        mark = "OK  " if result["ok"] else "FAIL"
        print(f"  {mark} {result['label']:<24s} {result['minutes']:>5.1f} min")
    print(f"\n{ok}/{len(results)} workflows passed")
    out = REPO / "results" / "live-validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "budget": args.budget,
        "note": (
            "Purpose: prove the workflows work end-to-end, not to measure clean "
            "speed. Any background load (ffmpeg/upscale) depresses the numbers; "
            "see docs/MEASUREMENT_CONTAMINATION.md."
        ),
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Report: {out}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
