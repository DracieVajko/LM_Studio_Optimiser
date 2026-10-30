"""Night status: live runs, checkpoint age, LM Studio, foreign models. Read-only."""

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

DB = "file:data/optimizer.db?mode=ro"


def main() -> None:
    now = time.time()
    print(f"===== {datetime.now():%Y-%m-%d %H:%M:%S} =====")

    con = sqlite3.connect(DB, uri=True)
    con.row_factory = sqlite3.Row

    rows = con.execute(
        "SELECT id, model_id, status, stage, started_at FROM runs "
        "WHERE status IN ('running', 'resumed') ORDER BY started_at DESC"
    ).fetchall()
    print(f"\nLIVE RUNS ({len(rows)}):")
    for x in rows:
        n = con.execute(
            "SELECT COUNT(*) FROM configurations WHERE run_id = ?", (x["id"],)
        ).fetchone()[0]
        cp = Path("data/reports/checkpoints") / f"run_{x['id']}.json"
        age = f"{(now - cp.stat().st_mtime) / 60:.1f} min ago" if cp.exists() else "none"
        print(f"  {x['id'][:8]}  {x['model_id']:<44} {x['status']:<8} "
              f"{x['stage']:<18} cfg={n:<4} ckpt={age}")

    day = con.execute(
        "SELECT id, model_id, status, stage, duration_seconds FROM runs "
        "WHERE started_at > datetime('now', '-1 day') AND status NOT IN "
        "('running','resumed') ORDER BY started_at DESC"
    ).fetchall()
    print(f"\nFINISHED TODAY ({len(day)}):")
    for x in day:
        d = x["duration_seconds"] or 0
        print(f"  {x['id'][:8]}  {x['model_id']:<44} {x['status']:<16} "
              f"{x['stage']:<18} {d / 60:.0f} min")
    con.close()

    try:
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:1234/api/v1/models", timeout=15) as r:
            data = json.loads(r.read())
        loaded = [m for m in data.get("data", data.get("models", [])) if m.get("state") == "loaded"]
        print(f"\nLM STUDIO: OK ({len(loaded)} loaded)")
        for m in loaded:
            print(f"    loaded: {m.get('id')}")
    except Exception as e:
        print(f"\nLM STUDIO: DOWN ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()