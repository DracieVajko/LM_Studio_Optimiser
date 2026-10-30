import sqlite3

con = sqlite3.connect("file:data/optimizer.db?mode=ro", uri=True)
con.row_factory = sqlite3.Row
rows = con.execute(
    "SELECT id, model_id, status, stage FROM runs "
    "WHERE status IN ('running','resumed') ORDER BY created_at DESC"
).fetchall()
for r in rows:
    print(r["id"], r["model_id"], r["status"], r["stage"])
if not rows:
    print("no active run")
