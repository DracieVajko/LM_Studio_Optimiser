import json
import sqlite3
import statistics

con = sqlite3.connect("file:data/optimizer.db?mode=ro", uri=True)
con.row_factory = sqlite3.Row

runs = con.execute("""
    SELECT id, model_id FROM runs
    WHERE status IN ('completed_no_validated_winner','success','completed')
    ORDER BY created_at DESC
""").fetchall()

for r in runs:
    speeds = []
    for c in con.execute("SELECT metrics_json FROM configurations WHERE run_id=? AND status='passed'", (r["id"],)):
        raw = c["metrics_json"]
        if not raw:
            continue
        try:
            for t in json.loads(raw):
                g = t.get("generation_tok_s")
                if isinstance(g, (int, float)) and t.get("category") == "speed":
                    speeds.append(g)
        except Exception:
            pass
    if speeds:
        sp = sorted(speeds)
        n = len(sp)
        med = sp[n//2] if n % 2 else (sp[n//2-1] + sp[n//2]) / 2
        print(f"{r['model_id'][:50]:50s} n={len(sp):3d} med={statistics.median(sp):.2f} min={min(sp):.2f} max={max(sp):.2f}")