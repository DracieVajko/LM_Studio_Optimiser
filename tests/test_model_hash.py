"""Model hashing tests (spec §16.3 / §21)."""
import hashlib
import json

import pytest

from lm_optimizer.services.model_hash import (
    _candidate_paths,
    hash_all_models,
    model_sha256,
    models_root,
    sha256_file,
)


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "model.gguf"
    data = b"fake gguf bytes" * 1000
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_sha256_streaming_large(tmp_path):
    path = tmp_path / "big.bin"
    data = b"x" * (3 * 1024 * 1024)
    path.write_bytes(data)
    assert sha256_file(path, chunk_size=4096) == hashlib.sha256(data).hexdigest()


def test_model_sha256_missing_file_returns_none(tmp_path):
    result = model_sha256(
        "publisher/does-not-exist",
        cache_path=tmp_path / "cache.json",
    )
    assert result["sha256"] is None
    assert result["error"]


def test_model_sha256_uses_cache(tmp_path, monkeypatch):
    import lm_optimizer.services.model_hash as mod

    root = tmp_path / "models"
    model_dir = root / "publisher" / "model"
    model_dir.mkdir(parents=True)
    gguf = model_dir / "model-q4.gguf"
    gguf.write_bytes(b"weights")
    monkeypatch.setattr(mod, "models_root", lambda: root)

    cache = tmp_path / "cache.json"
    first = model_sha256("publisher/model", cache_path=cache)
    assert first["sha256"]
    assert first["cached"] is False
    assert first["size_bytes"] == len(b"weights")

    second = model_sha256("publisher/model", cache_path=cache)
    assert second["sha256"] == first["sha256"]
    assert second["cached"] is True


def test_model_sha256_refresh_recomputes(tmp_path, monkeypatch):
    import lm_optimizer.services.model_hash as mod

    root = tmp_path / "models"
    model_dir = root / "publisher" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.gguf").write_bytes(b"weights")
    monkeypatch.setattr(mod, "models_root", lambda: root)

    cache = tmp_path / "cache.json"
    model_sha256("publisher/model", cache_path=cache)
    again = model_sha256("publisher/model", cache_path=cache, refresh=True)
    assert again["cached"] is False


def test_candidate_paths_strips_quant_suffix(tmp_path, monkeypatch):
    import lm_optimizer.services.model_hash as mod

    root = tmp_path / "models"
    model_dir = root / "ukisai" / "swift-qwen3.8-27b"
    model_dir.mkdir(parents=True)
    (model_dir / "swift-q2_k.gguf").write_bytes(b"x")
    monkeypatch.setattr(mod, "models_root", lambda: root)

    candidates = _candidate_paths("ukisai/swift-qwen3.8-27b@q2_k")
    assert candidates
    assert candidates[0].name == "swift-q2_k.gguf"


def test_models_root_returns_none_when_missing(monkeypatch):
    import lm_optimizer.services.model_hash as mod

    monkeypatch.setattr(mod, "_LMSTUDIO_DIRS", ())
    assert models_root() is None


def test_hash_all_models(tmp_path, monkeypatch):
    import lm_optimizer.services.model_hash as mod

    root = tmp_path / "models"
    for name in ("a", "b"):
        d = root / "pub" / name
        d.mkdir(parents=True)
        (d / f"{name}.gguf").write_bytes(f"data-{name}".encode())
    monkeypatch.setattr(mod, "models_root", lambda: root)

    results = hash_all_models(["pub/a", "pub/b"], cache_path=tmp_path / "c.json")
    assert len(results) == 2
    assert all(r["sha256"] for r in results)
    assert results[0]["sha256"] != results[1]["sha256"]
