#!/usr/bin/env python3
"""Normalize one-off smoke-test JSON into self-describing versioned objects.

Provenance: `qwen-variants-smoke.json` was produced by an ad-hoc probe script
that is no longer in the repository. Its records are positional 4-tuples:

    [model_id, generation_tok_s, elapsed_seconds, error | null]

Field meanings (established from the probe flow and the campaign logs):
- index 0: model identifier as passed to LM Studio (string)
- index 1: aggregate generation speed measured by the smoke probe.
  0.0 is the sentinel for a FAILED LOAD (not a measurement).
- index 2: probe wall-clock total (load + generate), seconds (float)
- index 3: error details (string) or null on success

The converter emits a versioned, self-describing document and preserves only
collected values (no invented quality/context numbers). The source file is
left untouched; write to a NEW path with `-o`.

Usage:
    python tools/smoke_json_normalize.py results/qwen-variants-smoke.json \\
        -o results/qwen-variants-smoke.v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA_VERSION = 2
SOURCE_SCHEMA = "positional-v1-smoke-probe"


def normalize_record(record: list, index: int) -> dict:
    """Convert one positional tuple to a versioned object (no data invented)."""
    if not isinstance(record, list) or len(record) != 4:
        return {
            "row_index": index,
            "schema_version": SCHEMA_VERSION,
            "source_schema": SOURCE_SCHEMA,
            "status": "malformed_record",
            "error": f"expected 4-element array, got {type(record).__name__} "
            f"of length {len(record) if isinstance(record, (list, dict, str)) else '?'}",
        }
    model_id, gen_tok_s, elapsed_s, error = record
    measured_speed = isinstance(gen_tok_s, (int, float))
    load_failed = error is not None or (measured_speed and gen_tok_s == 0.0)
    status = (
        "failed"
        if load_failed or not measured_speed
        else "success"
    )
    return {
        "row_index": index,
        "schema_version": SCHEMA_VERSION,
        "source_schema": SOURCE_SCHEMA,
        "model_id": model_id,
        "status": status,
        "generation_tok_s": gen_tok_s if measured_speed else None,
        "elapsed_seconds": elapsed_s if isinstance(elapsed_s, (int, float)) else None,
        "error": error,
        "field_notes": {
            "generation_tok_s": (
                "probe aggregate speed; 0.0 is a load-failure sentinel, not a "
                "measurement; aggregation = probe (single short test)"
            ),
            "elapsed_seconds": "whole probe wall time incl. model load",
            "error": None,
        },
    }


def normalize_file(inp: Path, out: Path) -> int:
    data = json.loads(inp.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print(f"error: {inp} is not a JSON array", file=sys.stderr)
        return 2
    doc = {
        "schema_version": SCHEMA_VERSION,
        "source_schema": SOURCE_SCHEMA,
        "source_file": str(inp),
        "converted_at_note": "measurement timestamps were never recorded by "
        "the source probe; no timestamps are invented",
        "records": [normalize_record(r, i) for i, r in enumerate(data)],
        "summary": {
            "rows": len(data),
            "success": sum(1 for r in data if isinstance(r, list) and len(r) == 4
                           and r[3] is None and r[1] != 0.0),
            "failed": sum(1 for r in data if not (isinstance(r, list) and len(r) == 4
                        and r[3] is None and r[1] != 0.0)),
        },
    }
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({doc['summary']['rows']} rows)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True,
                    help="destination (new file; the source stays untouched)")
    args = ap.parse_args()
    return normalize_file(args.input, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
