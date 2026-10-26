import sqlite3
from pathlib import Path

con = sqlite3.connect(f"file:{Path('data/optimizer.db')}?mode=ro", uri=True)
con.row_factory = sqlite3.Row

print("qwen3.8-9b runs:")
for r in con.execute(
    "SELECT id, status, stage, created_at, started_at, completed_at, duration_seconds "
    "FROM runs WHERE model_id LIKE 'qwen3.8-9b%' ORDER BY created_at DESC LIMIT 6"
):
    print(
        f"  {r['id'][:8]} {r['status']:32s} created={str(r['created_at'])[:19]} "
        f"started={str(r['started_at'])[:19]} completed={str(r['completed_at'])[:19]} "
        f"dur={r['duration_seconds']}"
    )

print("\nlatest runs:")
for r in con.execute(
    "SELECT id, model_id, status, created_at FROM runs ORDER BY created_at DESC LIMIT 8"
):
    print(f"  {r['id'][:8]} {r['model_id'][:45]:45s} {r['status']:32s} {str(r['created_at'])[:19]}")
