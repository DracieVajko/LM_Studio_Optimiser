"""Median measured generation tok/s per model from the latest run (read-only)."""

import json
import sqlite3
import statistics
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "optimizer.db"
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row

rows = con.execute(
    """
    SELECT r.model_id, r.id, r.created_at, r.status
    FROM runs r
    JOIN (SELECT model_id, MAX(created_at) mx FROM runs GROUP BY model_id) l
      ON l.model_id = r.model_id AND l.mx = r.created_at
    ORDER BY r.model_id
    """
).fetchall()

print(f"{'model':58s} {'status':30s} {'n':>3s} {'median':>8s} {'max':>8s} ctx")
print("-" * 118)
for r in rows:
    speeds = []
    ctxs = set()
    for c in con.execute(
        "SELECT metrics_json, context_length FROM configurations "
        "WHERE run_id=? AND status='passed'",
        (r["id"],),
    ):
        try:
            for m in json.loads(c["metrics_json"] or "[]"):
                g = m.get("generation_tok_s")
                if isinstance(g, (int, float)) and g > 0:
                    speeds.append(g)
                    ctxs.add(c["context_length"])
        except Exception:
            continue
    if speeds:
        med = statistics.median(speeds)
        print(
            f"{r['model_id'][:58]:58s} {r['status']:30s} {len(speeds):3d} "
            f"{med:8.2f} {max(speeds):8.2f} {sorted(c for c in ctxs if c)}"
        )
    else:
        print(f"{r['model_id'][:58]:58s} {r['status']:30s}   0        -        - -")
