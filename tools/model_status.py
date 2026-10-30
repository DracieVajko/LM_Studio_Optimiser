"""Status of the most recent run per available model (read-only)."""

import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "optimizer.db"
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row

rows = con.execute(
    """
    SELECT r.model_id, r.id, r.status, r.created_at, r.duration_seconds,
           r.best_config_id, r.stage
    FROM runs r
    JOIN (
        SELECT model_id, MAX(created_at) AS mx
        FROM runs GROUP BY model_id
    ) latest ON latest.model_id = r.model_id AND latest.mx = r.created_at
    ORDER BY r.created_at DESC
    """
).fetchall()

print(f"{'model':55s} {'status':22s} {'created':20s} {'min':>6s} configs")
print("-" * 110)
for r in rows:
    n = con.execute(
        "SELECT COUNT(*) FROM configurations WHERE run_id=?", (r["id"],)
    ).fetchone()[0]
    dur = (r["duration_seconds"] or 0) / 60.0
    print(
        f"{r['model_id'][:55]:55s} {r['status']:22s} "
        f"{(r['created_at'] or '')[:19]:20s} {dur:6.1f} {n:>6d}"
    )
