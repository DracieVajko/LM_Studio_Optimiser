#!/usr/bin/env python3
"""Project cleanup: archive + delete runs/benchmarks, logs, caches.

Safety model:
- LM Studio models are NEVER touched (they live outside this repo).
- Curated artifacts are kept: results/ reports, data/presets, the DB tables
  models/presets/settings/migrations.
- Refuses to run while the optimizer job lock is alive (a run is active) -
  deleting logs or the DB under an active campaign would corrupt it.
- Runs + benchmarks are ARCHIVED before deletion: the whole optimizer.db is
  snapshotted into data/archive/<timestamp>/ (same precedent as the manual
  optimizer.db.*.bak files), together with data/reports and data/checkpoints
  content. Logs and caches are deleted without archive (user choice).

Usage:
    python tools/cleanup.py --dry-run          # show what would happen
    python tools/cleanup.py --yes              # do it
    python tools/cleanup.py --yes --logs-only  # logs + caches only

Options:
    --data-dir DIR    default <repo>/data (override for tests)
    --repo-root DIR   default the repo root (override for tests)
    --logs-only       skip the DB archive/delete, only logs + caches
Exit codes: 0 clean(ed), 1 refused (active run / nothing to do), 2 usage.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[1]

# Run/benchmark tables to empty. Children of configurations first, then runs,
# then snapshots. models/presets/settings/migrations are intentionally kept.
RUN_TABLES = (
    "benchmarks",
    "quality_results",
    "configurations",
    "capability_snapshots",
    "judge_verdicts",
    "duels",
    "runs",
    "hardware_snapshots",
)

# Cache hit list (globs relative to repo root, never entering .venv / node).
CACHE_GLOBS = (
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "**/__pycache__",
)


def lock_alive() -> bool:
    """True when an optimizer run holds the job lock (best effort)."""
    try:
        from lm_optimizer.services import joblock

        holder = joblock._read_holder(joblock._lock_path())
        return bool(holder and joblock._holder_alive(holder))
    except Exception:
        return False  # cannot verify -> assume free, do not block cleanup


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def shutil_delete(target: Path) -> None:
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
    elif target.exists():
        target.unlink(missing_ok=True)


def shutil_delete_contents(directory: Path) -> None:
    """Delete everything INSIDE a directory, keeping the directory itself."""
    if not directory.is_dir():
        return
    for child in directory.iterdir():
        shutil_delete(child)


def empty_runs(db_path: Path) -> dict[str, int]:
    """Delete run/benchmark rows; keep models/presets/settings/migrations."""
    deleted: dict[str, int] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for table in RUN_TABLES:
            if table not in existing:
                deleted[table] = 0
                continue
            try:
                cur = conn.execute(f"DELETE FROM [{table}]")
                deleted[table] = cur.rowcount if cur.rowcount >= 0 else 0
            except sqlite3.Error as e:
                print(f"  [warn] {table}: {e}")
                deleted[table] = -1
        conn.commit()
        conn.execute("VACUUM")
        conn.commit()
    finally:
        conn.close()
    return deleted


def collect_caches(root: Path) -> list[Path]:
    found: list[Path] = []
    for pat in CACHE_GLOBS:
        for hit in root.glob(pat):
            if ".venv" in hit.parts or "node_modules" in hit.parts:
                continue
            found.append(hit)
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=REPO_DEFAULT)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--logs-only", action="store_true",
                    help="only logs + caches (no DB archive/delete)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)

    root = args.repo_root
    data = args.data_dir or (root / "data")
    db_path = data / "optimizer.db"
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = data / "archive" / f"cleanup_{now}"

    if lock_alive():
        print("[REFUSED] An optimizer run holds the job lock.")
        print("Stop/pause it (or finish cleanup after it ends), then retry.")
        return 1
    if not db_path.exists():
        print(f"[WARN] No DB at {db_path} - cleaning logs/caches only.")
        args.logs_only = True

    # --- build the action plan ------------------------------------------
    logs_dir = data / "logs"
    reports_dir = data / "reports"
    checkpoints_dir = data / "checkpoints"
    caches = collect_caches(root)

    print(f"Repo:       {root}")
    print(f"Data dir:   {data}")
    print(f"Logs:       {human(dir_size(logs_dir))}")
    print(f"Reports:    {human(dir_size(reports_dir))}")
    print(f"Checkpoints:{human(dir_size(checkpoints_dir))}")
    print(f"DB:         {human(db_path.stat().st_size) if db_path.exists() else 'none'}")
    print(f"Cache dirs: {len(caches)} ({human(sum(dir_size(c) for c in caches))})")

    if args.dry_run:
        print("\n[DRY RUN] Would:")
        if not args.logs_only:
            print(f"  1. create archive {archive_dir}")
            print(f"     copy optimizer.db, move reports/, checkpoints/")
            print(f"  2. DELETE rows from: {', '.join(RUN_TABLES)}")
        print(f"  3. delete {logs_dir}/*")
        print(f"  4. delete {len(caches)} cache dir(s)")
        print("  Kept: models/presets/settings tables, results/, data/presets/,")
        print("        LM Studio models, the archive files themselves.")
        return 0

    if not args.yes:
        print("\nAdd --yes to execute (or --dry-run first to inspect).")
        return 2

    # --- execute ---------------------------------------------------------
    if not args.logs_only:
        archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db_path, archive_dir / "optimizer.db")
        for name in ("reports", "checkpoints"):
            src = data / name
            if src.is_dir() and any(src.iterdir()):
                shutil.move(str(src), str(archive_dir / name))
                src.mkdir(parents=True, exist_ok=True)
        print(f"[archive] {archive_dir}")
        deleted = empty_runs(db_path)
        total = sum(v for v in deleted.values() if v > 0)
        for table, count in deleted.items():
            print(f"  [db] {table}: {'-' if count < 0 else count} rows deleted")
        print(f"[db] {total} rows total, DB now "
              f"{human(db_path.stat().st_size)}")

    shutil_delete_contents(logs_dir)
    if not logs_dir.exists():
        logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"[logs] cleared {logs_dir}")

    for cache in caches:
        shutil_delete(cache)
    print(f"[cache] removed {len(caches)} dir(s)")
    print("[done] Models and presets were NOT touched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
