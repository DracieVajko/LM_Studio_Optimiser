"""Per-level reasoning comparison for the Swift reasoning sweeps."""

import json
import sqlite3

con = sqlite3.connect("file:data/optimizer.db?mode=ro", uri=True)
con.row_factory = sqlite3.Row


def show(prefix: str, label: str):
    run = con.execute(
        "SELECT id, status, duration_seconds FROM runs WHERE id LIKE ?", (prefix + "%",)
    ).fetchone()
    if not run:
        print(f"{label}: run not found")
        return
    print(f"\n=== {label} (run {run['id'][:8]}, {run['status']}, "
          f"{round((run['duration_seconds'] or 0)/60,1)} min) ===")
    best = con.execute(
        "SELECT quality_json, context_length, status FROM configurations "
        "WHERE run_id=? AND quality_json IS NOT NULL ORDER BY id DESC LIMIT 1",
        (run["id"],),
    ).fetchone()
    if best and best["quality_json"]:
        try:
            q = json.loads(best["quality_json"])
            print(f"winner: ctx={best['context_length']} overall={q.get('overall')} "
                  f"checks={q.get('checks_passed')}/{q.get('checks_total')}")
        except Exception:
            pass
    rows = con.execute(
        "SELECT metrics_json FROM configurations WHERE run_id=? AND status='passed'",
        (run["id"],),
    ).fetchall()
    levels = {}
    for r in rows:
        try:
            for m in json.loads(r["metrics_json"] or "[]"):
                name = str(m.get("test_name") or "")
                level = name.rsplit("_", 1)[-1]
                g = m.get("generation_tok_s")
                if level in ("off", "low", "medium", "high", "xhigh") and g:
                    levels.setdefault(level, []).append((g, m.get("completion_tokens") or 0))
        except Exception:
            pass
    order = ["off", "low", "medium", "high", "xhigh"]
    print(f"{'level':8s} {'gen tok/s':>10s} {'avg tokens':>10s}  samples")
    for lv in order:
        vals = levels.get(lv, [])
        if not vals:
            print(f"{lv:8s} {'-':>10s} {'-':>10s}  0")
        else:
            sp = sum(v[0] for v in vals) / len(vals)
            tk = sum(v[1] for v in vals) / len(vals)
            print(f"{lv:8s} {sp:10.2f} {tk:10.0f}  {len(vals)}")


show("48fc0924", "swift q2_k reasoning")
show("d8bd9255", "swift q4_k_s reasoning")
