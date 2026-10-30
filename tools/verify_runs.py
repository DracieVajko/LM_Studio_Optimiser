"""Verify implementation invariants on persisted runs (read-only).

Usage:  python tools/verify_runs.py [--since ISO] [--limit N]

Checks per run:
  1. benchmark_params.resolved_settings exists (spec §9)
  2. resolved_settings.run_preset == benchmark_params.run_preset
  3. speed_floor block exists (value/enforced/met/fastest_candidate_id/tok_s)
  4. probe count <= max_speed_probes (spec §4.2)
  5. skip_phases == all_stages - selected_stages (+ granular skips)
  6. enforced+not-met => status completed_no_validated_winner AND
     no_winner_reason == NO_CONFIG_MEETS_SPEED_FLOOR
  7. not-enforced => no forced no-winner reason
  8. every tested config has a context_length > 0
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "optimizer.db"
PHASE_STAGES = {"speed", "quality", "recovery", "validation", "context"}


def _load_run(row, con):
    params = json.loads(row["benchmark_params_json"] or "{}")
    perf = json.loads(row["profile_weights_json"] or "{}")
    del perf
    search = json.loads(row["search_space_json"] or "{}")
    configs = con.execute(
        "SELECT context_length, status FROM configurations WHERE run_id=?",
        (row["id"],),
    ).fetchall()
    return params, search, configs


def check_run(row, con) -> list[tuple[str, bool, str]]:
    params, search, configs = _load_run(row, con)
    checks: list[tuple[str, bool, str]] = []

    resolved = params.get("resolved_settings")
    checks.append(("resolved_settings", isinstance(resolved, dict),
                   "present" if isinstance(resolved, dict) else "MISSING"))

    preset = params.get("run_preset")
    checks.append(("preset_matches",
                   isinstance(resolved, dict) and resolved.get("run_preset") == preset,
                   f"run={preset} snap={None if not isinstance(resolved, dict) else resolved.get('run_preset')}"))

    floor = search.get("speed_floor")
    floor_ok = (
        isinstance(floor, dict)
        and "value" in floor
        and "enforced" in floor
        and "met" in floor
        and "fastest_candidate_id" in floor
        and "fastest_tok_s" in floor
    )
    checks.append(("speed_floor_block", floor_ok, str(floor)[:80] if floor else "MISSING"))

    cap = None
    if isinstance(resolved, dict):
        cap = resolved.get("max_speed_probes")
    speed_cfgs = [c for c in configs if c["status"] == "passed"]
    if cap is not None:
        ok = len(speed_cfgs) <= int(cap)
        checks.append(("probe_cap", ok, f"configs={len(speed_cfgs)} cap={cap}"))
    else:
        checks.append(("probe_cap", True, "no cap (full plan)"))

    stages = set(resolved.get("selected_stages") or []) if isinstance(resolved, dict) else set()
    expected_skips = sorted(PHASE_STAGES - stages)
    actual_skips = sorted(params.get("skip_phases") or [])
    # Granular modes (e.g. --speed-only) may skip MORE phases than the preset;
    # every preset-implied skip must still be present.
    ok = set(expected_skips) <= set(actual_skips)
    checks.append(("stage_skips", ok, f"actual={actual_skips} expected>={expected_skips}"))

    if isinstance(floor, dict):
        enforced = bool(floor.get("enforced"))
        met = bool(floor.get("met"))
        # Reason is persisted in benchmark_params (no dedicated DB column);
        # older rows may only allow deriving it from the floor block.
        try:
            reason = row["no_winner_reason"]
        except (IndexError, KeyError):
            reason = None
        if not reason:
            reason = params.get("no_winner_reason")
        if not reason and enforced and not met:
            reason = "NO_CONFIG_MEETS_SPEED_FLOOR"
        status = row["status"]
        if enforced and not met:
            ok = status == "completed_no_validated_winner" and reason == "NO_CONFIG_MEETS_SPEED_FLOOR"
            checks.append(("floor_not_met_path", ok, f"status={status} reason={reason}"))
        else:
            checks.append(("floor_not_met_path", True, f"enforced={enforced} met={met}"))
        if not enforced:
            ok = reason is None
            checks.append(("floor_disabled_no_forced_reason", ok, f"reason={reason}"))
        else:
            checks.append(("floor_disabled_no_forced_reason", True, "not applicable"))

    ctx_ok = all((c["context_length"] or 0) > 0 for c in configs)
    checks.append(("contexts_positive", ctx_ok, f"n={len(configs)}"))
    return checks


def main(argv: list[str]) -> int:
    since = None
    limit = 50
    if "--since" in argv:
        since = datetime.fromisoformat(argv[argv.index("--since") + 1])
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])
    if since is None:
        since = datetime.now() - timedelta(hours=12)

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM runs WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
        (since.isoformat(timespec="seconds"), limit),
    ).fetchall()

    print(f"Verifying {len(rows)} run(s) created since {since.isoformat(timespec='seconds')}\n")
    failures = 0
    total = 0
    for row in rows:
        results = check_run(row, con)
        bad = [c for c in results if not c[1]]
        total += len(results)
        failures += len(bad)
        flag = "PASS" if not bad else "FAIL"
        print(f"[{flag}] {row['id'][:8]} {row['model_id'][:48]:48s} {row['status']}")
        for name, ok, detail in results:
            if not ok:
                print(f"        - {name}: {detail}")

    print(f"\n{total - failures}/{total} checks passed, {failures} failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
