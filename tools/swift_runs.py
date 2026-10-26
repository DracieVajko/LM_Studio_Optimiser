import sqlite3

con = sqlite3.connect("file:data/optimizer.db?mode=ro", uri=True)
con.row_factory = sqlite3.Row
for r in con.execute(
    "SELECT id, model_id, status, created_at, completed_at, duration_seconds "
    "FROM runs WHERE model_id LIKE 'swift%' ORDER BY created_at DESC"
):
    print(
        f"{r['id'][:8]} {r['model_id'][:30]:30s} {r['status']:32s} "
        f"created={str(r['created_at'])[:19]} completed={str(r['completed_at'])[:19]} "
        f"dur={r['duration_seconds']}"
    )
