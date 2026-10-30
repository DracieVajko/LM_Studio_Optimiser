"""Tests for judge_verdicts table migration."""

import tempfile
from pathlib import Path

import pytest

from lm_optimizer.database.manager import DatabaseManager


def test_judge_verdicts_table_exists(tmp_path: Path):
    """Test that judge_verdicts table is created with correct schema."""
    db = DatabaseManager(tmp_path / "test.db")
    db._init_database()
    with db.get_connection() as conn:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(judge_verdicts)")]
        expected = {
            "id", "run_ids_json", "config_ids_json", "use_case", "custom_prompt",
            "priority", "backend", "judge_model", "verdict_json", "prompt_tokens",
            "completion_tokens", "latency_ms", "status", "error_message", "created_at"
        }
        assert expected.issubset(set(cols))


def test_judge_verdicts_indexes_exist(tmp_path: Path):
    """Test that indexes are created on judge_verdicts table."""
    db = DatabaseManager(tmp_path / "test.db")
    db._init_database()
    with db.get_connection() as conn:
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='judge_verdicts'"
        ).fetchall()
        index_names = {row[0] for row in indexes}
        assert "idx_judge_verdicts_created" in index_names
        assert "idx_judge_verdicts_run" in index_names


def test_judge_verdicts_insert_and_query(tmp_path: Path):
    """Test inserting and querying judge verdicts."""
    db = DatabaseManager(tmp_path / "test.db")
    db._init_database()
    with db.get_connection() as conn:
        conn.execute("""
            INSERT INTO judge_verdicts (
                id, run_ids_json, config_ids_json, use_case, custom_prompt,
                priority, backend, judge_model, verdict_json,
                prompt_tokens, completion_tokens, latency_ms, status, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "test-verdict-1",
            '["run-1", "run-2"]',
            '["config-1"]',
            "coding",
            None,
            "balanced",
            "heuristic",
            "heuristic",
            '{"overall_winner": {"model_id": "test"}}',
            1000, 500, 2000, "completed", None
        ))
        row = conn.execute("SELECT * FROM judge_verdicts WHERE id = ?", ("test-verdict-1",)).fetchone()
        assert row is not None
        assert row["use_case"] == "coding"
        assert row["backend"] == "heuristic"
        assert row["prompt_tokens"] == 1000


def test_migration_version_11_in_migrations_list(tmp_path: Path):
    """Test that migration 11 is registered and runs."""
    db = DatabaseManager(tmp_path / "test.db")
    db._init_database()
    with db.get_connection() as conn:
        # Check migrations table has version 11
        versions = {row["version"] for row in conn.execute("SELECT version FROM migrations")}
        assert 11 in versions
        # Check the migration name
        name = conn.execute("SELECT name FROM migrations WHERE version = 11").fetchone()
        assert name is not None
        assert name["name"] == "add_judge_verdicts"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])