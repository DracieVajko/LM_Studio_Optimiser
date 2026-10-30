import sqlite3

con = sqlite3.connect("file:data/optimizer.db?mode=ro", uri=True)
con.row_factory = sqlite3.Row

runs = con.execute("""
    SELECT id, model_id, status, created_at
    FROM runs
    WHERE status IN ('completed_no_validated_winner','success','completed')
    ORDER BY created_at DESC
""").fetchall()

for r in runs:
    print(f"Run: {r['id'][:8]} Model: {r['model_id'][:40]:40s} Status: {r['status']} Created: {r['created_at']}")