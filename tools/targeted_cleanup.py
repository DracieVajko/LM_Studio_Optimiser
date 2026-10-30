"""Targeted cleanup: remove DB runs/configs for models not currently in LM Studio.

Keeps results only for models that are currently downloaded.
Archives the removed runs' data to data/archive/cleanup_targeted_<ts>/ first.
"""

import json
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DB = REPO / "data" / "optimizer.db"
ARCHIVE = REPO / "data" / "archive"

# Models currently in LM Studio (from /api/v1/models)
CURRENT_MODELS = {
    "qwen/qwen3.6-35b-a3b",
    "qwen/qwen3-coder-30b",
    "openai-gpt-oss-20b-claude-4.5-opus-heretic-uncensored-i1",
    "mirothinker-v1.5-30b-q2ks-mixed-autoround",
    "google/gemma-4-26b-a4b-qat",
    "qwen3.8-9b-distill",
    "prism-ml/bonsai-27b",
    "text-embedding-nomic-embed-text-v1.5",
}


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    # Find runs for models not currently in LM Studio
    all_runs = con.execute("SELECT id, model_id, status FROM runs").fetchall()
    to_remove = [r for r in all_runs if r["model_id"] not in CURRENT_MODELS]
    to_keep = [r for r in all_runs if r["model_id"] in CURRENT_MODELS]

    print(f"Total runs: {len(all_runs)}")
    print(f"Runs to keep (current models): {len(to_keep)}")
    print(f"Runs to remove (deleted models): {len(to_remove)}")

    if not to_remove:
        print("Nothing to remove.")
        con.close()
        return

    models_to_remove = set(r["model_id"] for r in to_remove)
    print("\nModels to remove:")
    for m in sorted(models_to_remove):
        count = sum(1 for r in to_remove if r["model_id"] == m)
        print(f"  {m} ({count} runs)")

    con.close()

    # Archive first
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = ARCHIVE / f"cleanup_targeted_{ts}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Archive DB snapshot
    shutil.copy2(DB, archive_dir / "optimizer.db.snapshot")
    print(f"\nDB snapshot archived: {archive_dir / 'optimizer.db.snapshot'}")

    # Now delete
    con = sqlite3.connect(str(DB))
    con.execute("PRAGMA foreign_keys = ON")

    run_ids = [r["id"] for r in to_remove]
    # Get all config IDs for these runs
    config_ids = []
    for run_id in run_ids:
        rows = con.execute("SELECT id FROM configurations WHERE run_id = ?", (run_id,)).fetchall()
        config_ids.extend([r[0] for r in rows])

    # Delete in FK order:
    # 1. benchmarks, quality_results (reference configurations)
    for config_id in config_ids:
        con.execute("DELETE FROM benchmarks WHERE config_id = ?", (config_id,))
        con.execute("DELETE FROM quality_results WHERE config_id = ?", (config_id,))

    # 2. presets referencing these runs
    for run_id in run_ids:
        con.execute("DELETE FROM presets WHERE run_id = ?", (run_id,))

    # 3. configurations
    for run_id in run_ids:
        con.execute("DELETE FROM configurations WHERE run_id = ?", (run_id,))

    # 4. runs
    for run_id in run_ids:
        con.execute("DELETE FROM runs WHERE id = ?", (run_id,))

    con.commit()
    # VACUUM
    con.execute("VACUUM")
    con.close()

    # Clean up logs directory (keep the directory, remove contents)
    logs_dir = REPO / "data" / "logs"
    if logs_dir.exists():
        for f in logs_dir.rglob("*"):
            if f.is_file():
                f.unlink()
        print("Logs cleaned.")

    print(f"\nDone. {len(run_ids)} runs removed, archived to {archive_dir}")
    print(f"Kept {len(to_keep)} runs for {len(CURRENT_MODELS)} current models.")


if __name__ == "__main__":
    main()