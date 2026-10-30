"""Tests for Judge API endpoints."""

import json
import sqlite3
from contextlib import closing
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from lm_optimizer.api.main import app
from lm_optimizer.database.manager import DatabaseManager
from lm_optimizer.config import config as cfg


@pytest.fixture
def client():
    """Test client for the API."""
    return TestClient(app)


@pytest.fixture
def temp_db():
    """Create a temporary database with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = DatabaseManager(db_path)
        db._init_database()

        # Insert hardware
        with db.get_connection() as conn:
            conn.execute(
                "INSERT INTO hardware_snapshots (id, os, cpu_name, cpu_cores_physical, cpu_cores_logical, total_ram_gb, gpu_count, gpus_json, cuda_version, metal_available, vulkan_available, detected_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("test-hw", "Windows", "Test CPU", 8, 16, 32.0, 1, '[{"index": 0, "name": "RTX 3060", "vram_gb": 6}]', "12.0", 0, 0, "2026-01-01T00:00:00")
            )
            conn.execute(
                "INSERT INTO models (id, name, architecture, parameter_count, quantization, context_limit, is_moe, num_experts, size_bytes, metadata_json, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("test-model", "test-model", "test", 1000000000, "q4", 32768, 0, None, 1000000000, "{}", "2026-01-01T00:00:00", "2026-01-01T00:00:00")
            )

        yield db_path


def _create_test_run(db_path: Path, run_id: str, model_id: str = "test-model"):
    """Create a test run with configurations."""
    import json as _json
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(
            """INSERT INTO runs (id, model_id, hardware_id, profile, profile_weights_json, quality_threshold,
                status, stage, search_space_json, optimizer_version, benchmark_repetitions, validation_repetitions,
                created_at, started_at, completed_at, duration_seconds, error,
                baseline_config_id, baseline_metrics_json, is_experimental, experimental_reason, benchmark_params_json,
                best_config_id, pareto_config_ids_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, model_id, "test-hw", "balanced", "{}", 0.95,
                "success", "complete", "{}", "test", 1, 1,
                "2026-01-01T00:00:00", "2026-01-01T00:00:00", "2026-01-01T00:00:00", 60, None,
                None, None, 0, None, "{}", None, "[]"
            )
        )

        # Add a config
        cfg_id = str(uuid4())
        config_json = _json.dumps({
            "context_length": 4096, "gpu_ratio": 0.8, "eval_batch_size": 256,
            "flash_attention": True, "offload_kv_cache_to_gpu": True,
            "generation": {"reasoning": "off"}
        })
        metrics_json = _json.dumps([{
            "test_name": "coding_task", "category": "coding", "success": True,
            "generation_tok_s": 10.5, "prompt_tok_s": 100.0, "estimated_ttft_ms": 50.0,
            "prompt_text": "Write a Python function...",
            "output_text": "def find_duplicates(nums):...",
        }])
        quality_json = _json.dumps({"overall": 0.95, "checks_passed": 28, "checks_total": 30,
                                     "by_test": {"coding_task": 0.95}})
        conn.execute(
            """INSERT INTO configurations (id, run_id, config_json, metrics_json, quality_json,
                stability_score, peak_vram_gb, peak_ram_gb, score, status, tested_at, context_length)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (cfg_id, run_id, config_json, metrics_json, quality_json,
             0.9, 5.2, None, 0.85, "passed", "2026-01-01T00:00:00", 4096)
        )


def test_post_judge_returns_judge_id(client, temp_db):
    """Test POST /api/judge returns judge_id."""
    run_id = str(uuid4())
    _create_test_run(temp_db, run_id)

    original_path = cfg.storage.database_path
    cfg.storage.database_path = temp_db

    try:
        resp = client.post("/api/judge", json={
            "run_ids": [run_id],
            "use_case": "coding",
            "priority": "balanced",
            "backend": "heuristic",
            "judge_model": "heuristic",
            "top_n_detail": 3,
            "wait": False,
        })
        assert resp.status_code == 202
        data = resp.json()
        assert "judge_id" in data
        assert data["status"] == "pending"
        assert "poll_url" in data
    finally:
        cfg.storage.database_path = original_path


def test_post_judge_wait_true_returns_verdict(client, temp_db):
    """Test POST /api/judge?wait=true returns verdict synchronously."""
    run_id = str(uuid4())
    _create_test_run(temp_db, run_id)

    original_path = cfg.storage.database_path
    cfg.storage.database_path = temp_db

    try:
        resp = client.post("/api/judge?wait=true", json={
            "run_ids": [run_id],
            "use_case": "coding",
            "priority": "balanced",
            "backend": "heuristic",
            "judge_model": "heuristic",
            "top_n_detail": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert "verdict" in data
        assert data["verdict"] is not None
        assert "recommended_config_id" in data["verdict"]
    finally:
        cfg.storage.database_path = original_path


def test_get_judge_by_id(client, temp_db):
    """Test GET /api/judge/{id} returns verdict."""
    original_path = cfg.storage.database_path
    cfg.storage.database_path = temp_db

    try:
        # First insert a verdict directly
        judge_id = str(uuid4())
        verdict_json = json.dumps({
            "recommended_config_id": "cfg-1",
            "reasoning": "Test verdict",
            "confidence": 0.8,
            "pros": ["Good speed"],
            "cons": [],
            "alternative_config_ids": [],
            "warnings": [],
        })
        with closing(sqlite3.connect(temp_db)) as conn, conn:
            conn.execute(
                "INSERT INTO judge_verdicts (id, run_ids_json, config_ids_json, use_case, custom_prompt, priority, backend, judge_model, verdict_json, prompt_tokens, completion_tokens, latency_ms, status, error_message) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (judge_id, '["run-1"]', '["cfg-1"]', "coding", None, "balanced", "heuristic", "heuristic", verdict_json, 1000, 500, 2000, "completed", None)
            )

        resp = client.get(f"/api/judge/{judge_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["judge_id"] == judge_id
        assert data["status"] == "completed"
        assert data["verdict"] is not None
    finally:
        cfg.storage.database_path = original_path


def test_get_judge_history(client, temp_db):
    """Test GET /api/judge/history returns list of verdicts."""
    original_path = cfg.storage.database_path
    cfg.storage.database_path = temp_db

    try:
        # Insert some verdicts
        for i in range(3):
            judge_id = str(uuid4())
            verdict_json = json.dumps({
                "overall_winner": {"model_id": f"model-{i}"},
                "confidence": 0.8,
            })
            with closing(sqlite3.connect(temp_db)) as conn, conn:
                conn.execute(
                    "INSERT INTO judge_verdicts (id, run_ids_json, config_ids_json, use_case, custom_prompt, priority, backend, judge_model, verdict_json, prompt_tokens, completion_tokens, latency_ms, status, error_message, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (judge_id, f'["run-{i}"]', f'["cfg-{i}"]', "coding", None, "balanced", "heuristic", "heuristic", verdict_json, 1000, 500, 2000, "completed", None, f"2026-01-{i+1:02d}T00:00:00")
                )

        resp = client.get("/api/judge/history?limit=20")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        assert all("id" in item for item in data)
        assert all("run_ids" in item for item in data)
    finally:
        cfg.storage.database_path = original_path


def test_post_judge_missing_run_ids(client, temp_db):
    """Test POST /api/judge without run_ids or model_ids returns 400."""
    original_path = cfg.storage.database_path
    cfg.storage.database_path = temp_db

    try:
        resp = client.post("/api/judge", json={
            "use_case": "coding",
            "priority": "balanced",
            "backend": "heuristic",
            "judge_model": "heuristic",
        })
        assert resp.status_code == 400
        assert "run_ids or model_ids" in resp.json()["detail"]
    finally:
        cfg.storage.database_path = original_path


def test_post_judge_api_backend_without_key(client, temp_db):
    """Test POST /api/judge with API backend but no key returns 400."""
    original_path = cfg.storage.database_path
    cfg.storage.database_path = temp_db

    try:
        resp = client.post("/api/judge", json={
            "run_ids": ["run-1"],
            "use_case": "coding",
            "backend": "api",
            "judge_model": "gpt-4",
        })
        assert resp.status_code == 400
        assert "API key required" in resp.json()["detail"]
    finally:
        cfg.storage.database_path = original_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])