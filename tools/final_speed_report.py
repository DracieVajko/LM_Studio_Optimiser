import json
import sqlite3
import statistics
from pathlib import Path

DB = Path("data/optimizer.db")
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row

print("=== All @16K speed runs (final tally) ===")
runs = con.execute("""
    SELECT r.id, r.model_id, r.status, r.stage, r.created_at
    FROM runs r
    WHERE r.status IN ('completed_no_validated_winner','success','completed')
    ORDER BY r.created_at DESC
""").fetchall()

for r in runs:
    speeds = []
    for c in con.execute(
        "SELECT metrics_json FROM configurations WHERE run_id=? AND status='passed'",
        (r["id"],),
    ):
        raw = c["metrics_json"]
        if not raw: continue
        try:
            for t in json.loads(raw):
                g = t.get("generation_tok_s")
                if isinstance(g, (int, float)) and t.get("category") == "speed":
                    speeds.append(g)
        except: pass
    if speeds:
        sp = sorted(speeds)
        n = len(sp)
        med = sp[n//2] if n % 2 else (sp[n//2-1] + sp[n//2]) / 2
        print(f"  {r['model_id'][:50]:50s} n={n:3d} med={med:.2f} min={min(sp):.2f} max={max(sp):.2f} @ctx {r['context_length'] or '?'}")