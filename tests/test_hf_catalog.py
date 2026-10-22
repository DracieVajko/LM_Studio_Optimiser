"""HuggingFace catalogue discovery tests (P5, spec §16.3) - offline."""
import pytest

from lm_optimizer.services.hf_catalog import (
    estimate_size_gb,
    parse_params_b,
    parse_quant,
    parse_repo_id,
    repo_to_entry,
    sync_catalogue,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Qwen3.8-27B-GGUF", 27.0),
        ("gemma-3-27b-it-GGUF", 27.0),
        ("Qwen3-8B-GGUF", 8.0),
        ("Qwen3-Coder-30B-A3B-Instruct-GGUF", 30.0),  # A3B must be skipped
        ("gpt-oss-20b-GGUF", 20.0),
        ("no params here", None),
    ],
)
def test_parse_params_b(name, expected):
    assert parse_params_b(name) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Qwen3.8-27B-Q4_K_M", "Q4_K_M"),
        ("model-IQ4_XS.gguf", "IQ4_XS"),
        ("model-q3_k_s", "Q3_K_S"),
        ("model-Q2_K", "Q2_K"),
        ("model-MXFP4", "MXFP4"),
        ("model-Q8_0", "Q8_0"),
        ("nothing", None),
    ],
)
def test_parse_quant(name, expected):
    assert parse_quant(name) == expected


def test_estimate_size_gb():
    assert estimate_size_gb(27.0, "Q4_K_M") == pytest.approx(16.2, abs=0.1)
    assert estimate_size_gb(27.0, "IQ3_XS") == pytest.approx(11.48, abs=0.1)
    assert estimate_size_gb(27.0, "UNKNOWN") is None
    assert estimate_size_gb(None, "Q4_K_M") is None


def test_parse_repo_id():
    info = parse_repo_id("bartowski/Qwen3.8-27B-GGUF")
    assert info["publisher"] == "bartowski"
    assert info["params_b"] == 27.0
    assert info["quant"] is None  # repo names usually omit the quant


def test_repo_to_entry_requires_quant():
    # A repo name without a quant tag cannot be turned into an entry.
    assert repo_to_entry("bartowski/Qwen3.8-27B-GGUF") is None


def test_repo_to_entry_with_quant():
    entry = repo_to_entry("bartowski/Qwen3.8-27B-IQ4_XS-GGUF")
    assert entry is not None
    assert entry.quant == "IQ4_XS"
    assert entry.total_params_b == 27.0
    assert entry.source == "huggingface"
    assert entry.confidence.value == "D"


def test_sync_skips_unparseable_and_duplicates():
    existing = [repo_to_entry("bartowski/Qwen3.8-27B-IQ4_XS-GGUF")]
    result = sync_catalogue(
        query="qwen",
        existing=existing,
        repo_ids=[
            "bartowski/Qwen3.8-27B-IQ4_XS-GGUF",  # duplicate
            "bartowski/Qwen3.8-27B-Q4_K_M-GGUF",  # new
            "someone/no-quant-here",              # unparseable
        ],
    )
    assert result.discovered == 3
    assert len(result.added) == 1
    assert result.added[0].quant == "Q4_K_M"
    assert result.skipped == 2
    assert result.errors


def test_sync_with_no_existing():
    result = sync_catalogue(
        query="x", repo_ids=["a/Qwen3-8B-Q4_K_M-GGUF"]
    )
    assert len(result.added) == 1
    assert result.added[0].file_size_gb == pytest.approx(4.8, abs=0.1)
