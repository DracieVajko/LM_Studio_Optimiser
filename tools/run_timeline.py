"""Timeline of one run: configurations, timestamps, gaps (read-only)."""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "optimizer.db"
run_id = sys.argv[1] if len(sys.argv) > 1 else None

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row

if not run_id:
    row = con.execute(
        "SELECT id, model_id FROM runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    run_id = row["id"]
    print(f"run {run_id} ({row['model_id']})")

run = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
print(
    f"status={run['status']} stage={run['stage']} created={run['created_at']} "
    f"started={run['started_at']} completed={run['completed_at']} "
    f"duration={run['duration_seconds']}s"
)
print(f"benchmark_params keys: {list(__import__('json').loads(run['benchmark_params_json'] or '{}').keys())}")

rows = con.execute(
    "SELECT id, status, context_length, tested_at, duration_ms, error "
    "FROM configurations WHERE run_id=? ORDER BY tested_at",
    (run_id,),
).fetchall()

print(f"\n{len(rows)} configurations:")
prev = None
for r in rows:
    ts = r["tested_at"] or "?"
    gap = ""
    try:
        t = datetime.fromisoformat(ts)
        if prev:
            gap = f"  gap={(t - prev).total_seconds():7.1f}s"
        prev = t
    except Exception:
        pass
    err = (r["error"] or "")[:50]
    print(
        f"  {ts[:19]}  {r['status']:18s} ctx={r['context_length']:<7} "
        f"{int((r['duration_ms'] or 0)/1000):5d}s{gap} {err}"
    )
