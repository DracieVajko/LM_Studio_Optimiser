"""Repair a run row that finished per checkpoint but kept 'running' in DB."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

RUN_ID = "417fda08-ad15-460a-b897-d4e797b75c23"
DB = Path("data/optimizer.db")
CKPT = Path(f"data/reports/checkpoints/run_{RUN_ID}.json")

ck = json.loads(CKPT.read_text(encoding="utf-8"))
if not str(ck.get("reason", "")).startswith("final:"):
    raise SystemExit(f"checkpoint not final ({ck.get('reason')}); refusing")

con = sqlite3.connect(DB)
row = con.execute("SELECT status, started_at FROM runs WHERE id=?", (RUN_ID,)).fetchone()
if row is None or row[0] != "running":
    print(f"nothing to do: status={None if row is None else row[0]}")
else:
    completed = datetime.fromtimestamp(CKPT.stat().st_mtime)
    started = datetime.fromisoformat(row[1])
    con.execute(
        "UPDATE runs SET status=?, stage=?, completed_at=?, duration_seconds=? WHERE id=?",
        (ck["status"], "complete", completed.isoformat(),
         (completed - started).total_seconds(), RUN_ID),
    )
    con.commit()
    print(f"repaired: running -> {ck['status']} completed_at={completed}")
