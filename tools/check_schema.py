#!/usr/bin/env python3
from lm_optimizer.database.manager import db_manager

with db_manager.get_connection() as conn:
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    for t in tables:
        print(f"Table: {t[0]}")
        cols = conn.execute(f"PRAGMA table_info({t[0]})").fetchall()
        for c in cols:
            print(f"  {c[1]} ({c[2]})")
        print()