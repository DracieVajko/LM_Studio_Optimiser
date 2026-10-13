"""Minimal regression tests for DeepResearch Bench integration (no network)."""

import asyncio
from pathlib import Path

from lm_optimizer.services.deepresearch import DeepResearchConfig, run_model_on_queries


def test_config_defaults_off():
    cfg = DeepResearchConfig(model_name="m")
    assert cfg.limit is None
    assert cfg.skip_cleaning is False
    assert cfg.force is False
    assert cfg.max_workers == 5
    assert cfg.openrouter_api_key is None
    assert cfg.llm_backend == "openrouter"


def test_run_model_no_query_file_returns_false(tmp_path):
    missing = tmp_path / "nope" / "query.jsonl"
    cfg = DeepResearchConfig(model_name="m", query_file=missing)
    assert cfg.query_file.exists() is False
    ok = asyncio.run(run_model_on_queries(cfg, "http://127.0.0.1:1234"))
    assert ok is False


def test_default_paths_relative():
    assert Path(DeepResearchConfig(model_name="m").dr_bench_dir) == Path("deep_research_bench")
