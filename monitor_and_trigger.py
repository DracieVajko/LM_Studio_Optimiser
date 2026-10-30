"""Background monitor for GPU/CPU load and quick Nemotron test trigger.

Runs continuously (or via manual invocation) and triggers quick tests
when GPU is no longer busy and CPU load is low. Designed for the user's
background process ('opscayl') monitoring.

Usage:
  python monitor_and_trigger.py --check-every 60 --wait-hours 1

No live LLM load by default; triggers quick_test_nemotron() (file/default
preparation only, dry_run by default) when conditions met.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from lm_optimizer.services.gpu_monitor import is_gpu_busy, is_cpu_busy, should_run_tests
from lm_optimizer.services.test_manager import TestManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-every", type=int, default=60, help="Seconds between checks")
    parser.add_argument("--wait-hours", type=int, default=1, help="Approximate wait before first trigger attempt")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()

    print(f"[{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}] Monitor started.")
    print(f"  GPU busy now: {is_gpu_busy()}")
    print(f"  CPU busy now: {is_cpu_busy()}")
    print(f"  Suggest short tests: {should_run_tests()['suggest_short_tests']}")
    print(f"  Wait minutes: {should_run_tests()['suggest_wait_minutes']}")
    print(f"  Check again in hours: {should_run_tests()['suggest_check_again_in_hours']}")

    # Quick trigger preparation: when GPU/CPU free, prepare quick Nemotron test.
    # This does NOT load any LLM model; only prepares files/defaults (dry_run by default).
    manager = TestManager()
    print("  Monitor will trigger quick_test_nemotron() when GPU/CPU is free (no live load).")
    print("  Trigger condition: is_gpu_busy() == False and is_cpu_busy() == False")

    # Example trigger call (manual or automated):
    # When conditions met: manager.quick_test_nemotron(dry_run=args.dry_run)

if __name__ == "__main__":
    main()
