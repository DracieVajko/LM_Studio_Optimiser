"""Regression tests: a timed-out optimizer child must be reconciled.

The night runner hard-kills an optimizer child on timeout (subprocess.run).
No live process can write fresh checkpoints afterwards, so a run row still in
``running``/``resumed`` belongs to a dead process and must be marked
``interrupted`` immediately (never left ``running`` indefinitely).
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import pytest

from lm_optimizer.database.manager import DatabaseManager


def _load_night_runner():
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("night_runner", repo / "tools" / "night_runner.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("night_runner", mod)
    spec.loader.exec_module(mod)
    return mod


night_runner = _load_night_runner()


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Bind the repositories singleton to a temporary database."""
    import lm_optimizer.database.manager as manager_mod
    import lm_optimizer.database.repositories as repositories

    test_manager = DatabaseManager(tmp_path / "test.db")
    monkeypatch.setattr(repositories, "db_manager", test_manager, raising=False)
    monkeypatch.setattr(manager_mod, "db_manager", test_manager, raising=False)
    return test_manager


def _insert_running_run(db, model_id: str = "slow-model", run_status: str = "running") -> str:
    run_id = str(uuid4())
    with db.get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO hardware_snapshots (id, os, cpu_name, cpu_cores_physical, cpu_cores_logical, total_ram_gb, gpu_count, gpus_json, cuda_version, metal_available, vulkan_available, detected_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("hw", "Windows", "Test CPU", 8, 16, 32.0, 1, '[{"index": 0, "name": "RTX 3060", "vram_gb": 6}]', "12.0", 0, 0, "2026-01-01T00:00:00"),
        )
        conn.execute(
            "INSERT OR IGNORE INTO models (id, name, architecture, parameter_count, quantization, context_limit, is_moe, num_experts, size_bytes, metadata_json, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (model_id, model_id, "test", 1000000000, "q4", 32768, 0, None, 1000000000, "{}", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
        )
        conn.execute(
            """INSERT INTO runs (id, model_id, hardware_id, profile, profile_weights_json, quality_threshold,
                status, stage, search_space_json, optimizer_version, benchmark_repetitions, validation_repetitions,
                created_at, started_at, completed_at, duration_seconds, error,
                baseline_config_id, baseline_metrics_json, is_experimental, experimental_reason, benchmark_params_json,
                best_config_id, pareto_config_ids_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, model_id, "hw", "balanced", "{}", 0.95,
                run_status, "coarse_search", "{}", "test", 1, 1,
                "2026-01-01T00:00:00", "2026-01-01T00:00:00", None, None, None,
                None, None, 0, None, "{}", None, "[]",
            ),
        )
    return run_id


def _get_run_status(db, run_id: str) -> str | None:
    with db.get_connection() as conn:
        row = conn.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()
    return row[0] if row else None


def test_timed_out_run_reconciled_to_interrupted(db):
    run_id = _insert_running_run(db, "timed-out-model")

    result = night_runner.reconcile_dead_run(
        "timed-out-model", rc=-1, timed_out=True, success_run=False,
    )

    assert result["changed"] == 1
    change = result["changes"][0]
    assert change["from"] == "running"
    assert change["to"] == "interrupted"
    assert _get_run_status(db, run_id) == "interrupted"


def test_nonzero_exit_reconciles_stale_running(db):
    run_id = _insert_running_run(db, "crashed-model")

    result = night_runner.reconcile_dead_run(
        "crashed-model", rc=1, timed_out=False, success_run=False,
    )

    assert result["changed"] == 1
    assert _get_run_status(db, run_id) == "interrupted"


def test_clean_exit_does_not_touch_running_row(db):
    _insert_running_run(db, "clean-model")

    result = night_runner.reconcile_dead_run(
        "clean-model", rc=0, timed_out=False, success_run=False,
    )

    assert "skipped" in result
    assert result["changed"] == 0 if "changed" in result else True


def test_timeout_reconciliation_is_idempotent(db):
    run_id = _insert_running_run(db, "twice-model")

    first = night_runner.reconcile_dead_run("twice-model", rc=-1, timed_out=True, success_run=False)
    second = night_runner.reconcile_dead_run("twice-model", rc=-1, timed_out=True, success_run=False)

    assert first["changed"] == 1
    assert second["changed"] == 0
    assert _get_run_status(db, run_id) == "interrupted"


def test_reconcile_scoped_to_model_only(db):
    victim = _insert_running_run(db, "victim-model")
    other = _insert_running_run(db, "other-model")

    night_runner.reconcile_dead_run("victim-model", rc=-1, timed_out=True, success_run=False)

    assert _get_run_status(db, victim) == "interrupted"
    assert _get_run_status(db, other) == "running"
