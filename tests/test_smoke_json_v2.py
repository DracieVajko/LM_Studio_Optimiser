"""Regression tests: smoke JSON v2 normalizer (self-describing output)."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from smoke_json_normalize import normalize_file, normalize_record  # noqa: E402


def test_positional_tuple_converts_to_versioned_object(tmp_path):
    src = tmp_path / "in.json"
    src.write_text(
        json.dumps([
            ["model-a", 10.5, 60.0, None],
            ["model-b", 0.0, 4.0, "Model load failed: boom"],
            ["model-c", 1.2, 120.0, None],
        ]),
        encoding="utf-8",
    )
    out = tmp_path / "out.json"
    assert normalize_file(src, out) == 0

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema_version"] == 2
    assert doc["summary"]["success"] == 2
    assert doc["summary"]["failed"] == 1

    recs = doc["records"]
    ok = recs[0]
    assert ok["model_id"] == "model-a"
    assert ok["status"] == "success"
    assert ok["generation_tok_s"] == 10.5
    assert ok["elapsed_seconds"] == 60.0
    assert ok["error"] is None
    assert "schema_version" in ok and "source_schema" in ok

    fail = recs[1]
    assert fail["status"] == "failed"
    assert fail["error"] == "Model load failed: boom"

    # zero speed with no error also counts failed (load-failure sentinel)
    recs.append("unused")


def test_malformed_record_reported_not_crashing():
    rec = normalize_record(["only-one"], 7)
    assert rec["row_index"] == 7
    assert rec["status"] == "malformed_record"
    assert rec["error"]
    assert rec["schema_version"] == 2


def test_zero_speed_without_error_is_failed_load_sentinel():
    rec = normalize_record(["m", 0.0, 9.0, None], 0)
    assert rec["status"] == "failed"


def test_success_record_requires_positive_speed():
    rec = normalize_record(["m", None, 9.0, None], 0)
    assert rec["status"] == "failed"
