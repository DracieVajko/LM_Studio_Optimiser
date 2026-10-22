"""Community benchmark import/export tests (P5, spec 16.4)."""
import json

import pytest

from lm_optimizer.services.community_benchmarks import COMMUNITY_BENCHMARKS
from lm_optimizer.services.community_import import (
    ImportError_,
    export_records,
    import_records,
    load_json_file,
    record_from_dict,
    record_to_dict,
)


def _valid_record(**kw):
    base = {
        "hardware": {
            "gpu_name": "NVIDIA GeForce RTX 3060 Laptop GPU",
            "vram_gb": 6.0,
            "ram_channels": 1,
            "ram_gb": 32.0,
        },
        "model_repository": "test/model",
        "model_filename": "model-q4.gguf",
        "quant": "Q4_K_M",
        "context_length": 16384,
        "decode_tps": 9.5,
        "prompt_tps": 120.0,
        "source": "unit test",
        "confidence": "C",
    }
    base.update(kw)
    return base


def test_record_roundtrip():
    record = record_from_dict(_valid_record())
    payload = record_to_dict(record)
    restored = record_from_dict(payload)
    assert restored.model_filename == record.model_filename
    assert restored.decode_tps == record.decode_tps
    assert restored.confidence == record.confidence


def test_missing_required_field_is_rejected():
    data = _valid_record()
    del data["model_filename"]
    with pytest.raises(ImportError_) as exc:
        record_from_dict(data)
    assert "model_filename" in str(exc.value)


def test_invalid_confidence_is_rejected():
    with pytest.raises(ImportError_) as exc:
        record_from_dict(_valid_record(confidence="X"))
    assert "confidence" in str(exc.value)


def test_zero_decode_tps_is_rejected():
    with pytest.raises(ImportError_):
        record_from_dict(_valid_record(decode_tps=0))


def test_missing_hardware_gpu_name_is_rejected():
    data = _valid_record()
    data["hardware"] = {"vram_gb": 6.0}
    with pytest.raises(ImportError_) as exc:
        record_from_dict(data)
    assert "gpu_name" in str(exc.value)


def test_import_skips_duplicates():
    base = [record_from_dict(_valid_record())]
    merged, warnings = import_records([_valid_record()], existing=base)
    assert len(merged) == len(base)
    assert warnings and "duplicate" in warnings[0]


def test_import_adds_new_records():
    base = [record_from_dict(_valid_record())]
    new = _valid_record(model_filename="other-q3.gguf", quant="Q3_K_M")
    merged, warnings = import_records([new], existing=base)
    assert len(merged) == len(base) + 1
    assert not warnings


def test_import_reports_malformed_records_without_failing():
    base = []
    payload = [{"model_filename": "broken"}]  # missing almost everything
    merged, warnings = import_records(payload, existing=base)
    assert merged == []
    assert warnings and "record 0" in warnings[0]


def test_import_accepts_wrapped_payload():
    merged, warnings = import_records({"records": [_valid_record()]}, existing=[])
    assert len(merged) == 1
    assert not warnings


def test_import_rejects_non_list_payload():
    with pytest.raises(ImportError_):
        import_records({"nope": 1}, existing=[])


def test_export_import_roundtrip():
    exported = export_records(COMMUNITY_BENCHMARKS)
    assert exported["records"]
    merged, warnings = import_records(exported, existing=[])
    assert len(merged) == len(COMMUNITY_BENCHMARKS)
    assert not warnings


def test_load_json_file(tmp_path):
    path = tmp_path / "records.json"
    path.write_text(json.dumps([_valid_record()]), encoding="utf-8")
    assert len(load_json_file(path)) == 1

    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"records": [_valid_record()]}), encoding="utf-8")
    assert len(load_json_file(wrapped)) == 1


def test_load_json_file_rejects_bad_root(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps("not a list"), encoding="utf-8")
    with pytest.raises(ImportError_):
        load_json_file(path)
