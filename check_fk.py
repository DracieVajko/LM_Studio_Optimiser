#!/usr/bin/env python3
from lm_optimizer.database.manager import db_manager

with db_manager.get_connection() as conn:
    tables = [row[0] for row in conn.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()]
    for table in tables:
        fks = conn.execute(f'PRAGMA foreign_key_list({table})').fetchall()
        for fk in fks:
            if fk[2] == 'runs':
                print(f'Table {table} references runs via {fk[3]} -> {fk[4]}')