"""Tests for JudgeService data loading."""

import json
import tempfile
import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest

from lm_optimizer.database.manager import DatabaseManager


def _setup_test_db(db_path: Path):
    """Set up test database with hardware, models, runs, and configs."""
    with sqlite3.connect(db_path) as conn:
        # Insert hardware snapshot
        conn.execute(
            "INSERT INTO hardware_snapshots (id, os, cpu_name, cpu_cores_physical, cpu_cores_logical, total_ram_gb, gpu_count, gpus_json, cuda_version, metal_available, vulkan_available, detected_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("test-hw", "Windows", "Test CPU", 8, 16, 32.0, 1, '[{"index": 0, "name": "RTX 3060", "vram_gb": 6, "vendor": "NVIDIA", "driver_version": "test", "compute_capability": "8.6", "shared_memory_gb": 0}]', "12.0", 0, 0, "2026-01-01T00:00:00")
        )
        # Insert model
        conn.execute(
            "INSERT INTO models (id, name, architecture, parameter_count, quantization, context_limit, is_moe, num_experts, size_bytes, metadata_json, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("test-model", "test-model", "test", 1000000000, "q4", 32768, 0, None, 1000000000, "{}", "2026-01-01T00:00:00", "2026-01-01T00:00:00")
        )


def _create_test_run(db_path: Path, run_id: str, model_id: str = "test-model"):
    """Create a test run with configurations directly via SQL."""
    with sqlite3.connect(db_path) as conn:
        # Insert run
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

        # Add 3 configs
        config_ids = []
        for i in range(3):
            cfg_id = str(uuid4())
            config_ids.append(cfg_id)
            config_json = json.dumps({
                "context_length": 4096, "gpu_ratio": 0.8, "eval_batch_size": 256,
                "flash_attention": True, "offload_kv_cache_to_gpu": True,
                "generation": {"reasoning": "off"}
            })
            metrics_json = json.dumps([{
                "test_name": "coding_task", "category": "coding", "success": True,
                "generation_tok_s": 10.5, "prompt_tok_s": 100.0, "estimated_ttft_ms": 50.0,
                "prompt_text": "Write a Python function...",
                "output_text": "def find_duplicates(nums):...",
            }])
            quality_json = json.dumps({"overall": 0.95, "checks_passed": 28, "checks_total": 30,
                                       "by_test": {"coding_task": 0.95}})
            conn.execute(
                """INSERT INTO configurations (id, run_id, config_json, metrics_json, quality_json,
                    stability_score, peak_vram_gb, peak_ram_gb, score, status, tested_at, context_length)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (cfg_id, run_id, config_json, metrics_json, quality_json,
                 0.9, 5.2, None, 0.85, "passed", "2026-01-01T00:00:00", 4096)
            )
        return config_ids


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        # Initialize with migrations
        db = DatabaseManager(db_path)
        db._init_database()
        _setup_test_db(db_path)
        yield db_path


def test_load_run_data_returns_configs_with_metrics(temp_db):
    """Test that _load_run_data returns run data with configs and metrics."""
    from lm_optimizer.services.judge_service import JudgeService

    run_id = str(uuid4())
    _create_test_run(temp_db, run_id)

    svc = JudgeService(db_path=temp_db)
    import asyncio
    data = asyncio.run(svc._load_run_data([run_id]))

    assert len(data) == 1
    assert len(data[0].configs) == 3
    for cfg in data[0].configs:
        assert "metrics_json" in cfg
        assert "quality_json" in cfg
        assert "config_json" in cfg
        assert cfg["metrics_json"] is not None
        assert cfg["quality_json"] is not None
        assert cfg["config_json"] is not None


def test_load_run_data_filters_by_config_ids(temp_db):
    """Test that _load_run_data filters by config_ids when provided."""
    from lm_optimizer.services.judge_service import JudgeService

    run_id = str(uuid4())
    config_ids = _create_test_run(temp_db, run_id)

    svc = JudgeService(db_path=temp_db)
    import asyncio
    # Filter to only first config
    data = asyncio.run(svc._load_run_data([run_id], config_ids=[config_ids[0]]))

    assert len(data) == 1
    assert len(data[0].configs) == 1
    assert data[0].configs[0]["id"] == config_ids[0]


def test_load_run_data_empty_run_raises(temp_db):
    """Test that _load_run_data raises for run with no valid configs."""
    from lm_optimizer.services.judge_service import JudgeService

    run_id = str(uuid4())
    with sqlite3.connect(temp_db) as conn:
        conn.execute(
            """INSERT INTO runs (id, model_id, hardware_id, profile, profile_weights_json, quality_threshold,
                status, stage, search_space_json, optimizer_version, benchmark_repetitions, validation_repetitions,
                created_at, started_at, completed_at, duration_seconds, error,
                baseline_config_id, baseline_metrics_json, is_experimental, experimental_reason, benchmark_params_json,
                best_config_id, pareto_config_ids_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, "test-model", "test-hw", "balanced", "{}", 0.95,
             "success", "complete", "{}", "test", 1, 1,
             "2026-01-01T00:00:00", "2026-01-01T00:00:00", "2026-01-01T00:00:00", 60, None,
             None, None, 0, None, "{}", None, "[]")
        )

    svc = JudgeService(db_path=temp_db)
    import asyncio
    with pytest.raises(ValueError, match="No configurations to evaluate"):
        asyncio.run(svc._load_run_data([run_id]))


def test_load_run_data_multiple_runs(temp_db):
    """Test loading data from multiple runs."""
    from lm_optimizer.services.judge_service import JudgeService

    # Add second model
    with sqlite3.connect(temp_db) as conn:
        conn.execute(
            "INSERT INTO models (id, name, architecture, parameter_count, quantization, context_limit, is_moe, num_experts, size_bytes, metadata_json, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("model-b", "model-b", "test", 1000000000, "q4", 32768, 0, None, 1000000000, "{}", "2026-01-01T00:00:00", "2026-01-01T00:00:00")
        )

    run_id_1 = str(uuid4())
    _create_test_run(temp_db, run_id_1, "test-model")

    run_id_2 = str(uuid4())
    _create_test_run(temp_db, run_id_2, "model-b")

    svc = JudgeService(db_path=temp_db)
    import asyncio
    data = asyncio.run(svc._load_run_data([run_id_1, run_id_2]))

    assert len(data) == 2
    # Order may vary, check both models present
    model_ids = {d.model_id for d in data}
    assert model_ids == {"test-model", "model-b"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])