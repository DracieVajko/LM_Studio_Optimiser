"""Final KEEP/DELETE report from the measured data.

Reads every speed run from the database, groups it per model, and prints
the comparison table that drives the keep/delete decision.

Usage:
    python tools/final_report.py [--context 16384] [--min-probes 8]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DB = REPO / "data" / "optimizer.db"

#: Models the user explicitly wants to keep (tools / utilities).
KEEP_ALWAYS = {
    "prism-ml/bonsai-27b": "tools/odysseus utilities",
}

#: Model families that are pure duplicates of another entry (same base model).
FAMILY_HINTS = {
    "swift": "27B dense",
    "qwen/qwen3.8-27b": "27B dense (base)",
    "qwen3.8-27b": "27B dense (base, bartowski)",
    "fable": "27B dense (fable)",
    "hauhaucs": "27B dense (hauhaucs)",
    "qwen3.8-9b": "9B dense",
    "gpt-oss": "20B MoE",
    "mirothinker": "30B MoE (reasoning)",
    "coder-30b": "30B MoE (coder)",
    "35b-a3b": "35B MoE",
    "gemma-4": "26B MoE",
    "flint": "40B dense",
}


def load_runs(con) -> list[dict]:
    rows = con.execute(
        "SELECT id, model_id, status, created_at FROM runs "
        "WHERE status IN ('completed_no_validated_winner','success','completed') "
        "ORDER BY created_at DESC"
    ).fetchall()
    out = []
    for row in rows:
        speeds: list[float] = []
        contexts: list[int] = []
        for cfg in con.execute(
            "SELECT metrics_json, context_length FROM configurations "
            "WHERE run_id=? AND status='passed'",
            (row["id"],),
        ):
            raw = cfg["metrics_json"]
            if raw:
                try:
                    for t in json.loads(raw):
                        g = t.get("generation_tok_s")
                        if isinstance(g, (int, float)) and t.get("category") == "speed":
                            speeds.append(g)
                except Exception:
                    pass
            if cfg["context_length"]:
                contexts.append(int(cfg["context_length"]))
        if not speeds:
            continue
        out.append(
            {
                "run_id": row["id"],
                "model_id": row["model_id"],
                "status": row["status"],
                "created_at": row["created_at"],
                "n": len(speeds),
                "median": statistics.median(speeds),
                "min": min(speeds),
                "max": max(speeds),
                "context": max(contexts) if contexts else None,
            }
        )
    return out


def newest_per_model(runs: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for run in runs:
        if run["model_id"] in seen:
            continue
        seen.add(run["model_id"])
        out.append(run)
    return out


def model_meta(con) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    try:
        for row in con.execute(
            "SELECT id, quantization, size_bytes, context_limit FROM models"
        ):
            meta[row["id"]] = {
                "quant": row["quantization"],
                "size_gb": (row["size_bytes"] or 0) / 1e9,
                "context_limit": row["context_limit"],
            }
    except Exception:
        pass
    return meta


def family(model_id: str) -> str:
    for key, label in FAMILY_HINTS.items():
        if key.lower() in model_id.lower():
            return label
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--min-probes", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    runs = load_runs(con)
    meta = model_meta(con)

    # Prefer the measurement at the requested context; fall back to the
    # newest measurement of any context (the context is always reported).
    at_ctx = [r for r in runs if r["context"] == args.context and r["n"] >= args.min_probes]
    others = [r for r in runs if r not in at_ctx and r["n"] >= args.min_probes]
    selected = newest_per_model(at_ctx) or newest_per_model(others)
    selected.sort(key=lambda r: -(r["median"] or 0))

    if args.json:
        print(json.dumps(selected, indent=2, ensure_ascii=False))
        return 0

    print(f"Speed comparison @ ctx {args.context} "
          f"(median generation tok/s, {args.min_probes}+ probes)")
    print("=" * 104)
    print(
        f"{'model':46s} {'quant':10s} {'size':>7s} {'N':>4s} "
        f"{'median':>8s} {'min':>7s} {'max':>7s}  {'ctx':>7s}  note"
    )
    print("-" * 104)
    for run in selected:
        info = meta.get(run["model_id"], {})
        note = ""
        if run["model_id"] in KEEP_ALWAYS:
            note = f"KEEP ({KEEP_ALWAYS[run['model_id']]})"
        elif run["context"] and run["context"] != args.context:
            note = f"measured @ctx {run['context']}"
        print(
            f"{run['model_id'][:46]:46s} "
            f"{str(info.get('quant') or '?')[:10]:10s} "
            f"{info.get('size_gb', 0):6.1f}G "
            f"{run['n']:4d} "
            f"{run['median']:8.2f} {run['min']:7.2f} {run['max']:7.2f}  "
            f"{run['context'] or 0:7d}  {note}"
        )
    print("=" * 104)
    print(f"{len(selected)} models with {args.min_probes}+ passed speed probes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
