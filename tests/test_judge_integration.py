"""Judge integration test: full heuristic flow (plan Task 7, 2026-10-19).

Runs the real service against a temporary database with two models and
three configurations each, then verifies the verdict structure and that a
row lands in judge_verdicts.
"""
import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from lm_optimizer.database.manager import DatabaseManager


def _seed_db(path: Path) -> tuple[list[str], list[str]]:
    """Create 2 runs x 3 configs with measurable quality/speed data."""
    db = DatabaseManager(path)
    run_ids: list[str] = []
    config_ids: list[str] = []
    with db.get_connection() as conn:
        # The runs table references hardware_snapshots via hardware_id.
        conn.execute(
            """
            INSERT INTO hardware_snapshots (
                id, os, cpu_name, cpu_cores_physical, cpu_cores_logical,
                total_ram_gb, gpu_count, gpus_json, metal_available,
                vulkan_available, detected_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "hw-test", "Windows", "Test CPU", 8, 16, 32.0, 1,
                "[]", 0, 0, "2026-10-22T09:00:00",
            ),
        )
        for model_index, model_id in enumerate(("model/alpha", "model/beta")):
            conn.execute(
                """
                INSERT INTO models (
                    id, name, architecture, parameter_count, quantization,
                    context_limit, is_moe, size_bytes, metadata_json,
                    first_seen, last_seen
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    model_id, model_id, "qwen35", 27_000_000_000, "Q4_K_M",
                    32768, 0, 16_000_000_000, "{}",
                    "2026-10-22T09:00:00", "2026-10-22T09:00:00",
                ),
            )
            run_id = str(uuid.uuid4())
            run_ids.append(run_id)
            conn.execute(
                """
                INSERT INTO runs (
                    id, model_id, hardware_id, profile, profile_weights_json,
                    quality_threshold, status, stage, optimizer_version,
                    benchmark_repetitions, validation_repetitions,
                    created_at, started_at, completed_at,
                    search_space_json, benchmark_params_json,
                    pareto_config_ids_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id, model_id, "hw-test", "balanced", "{}",
                    0.80, "success", "complete", "test",
                    3, 5,
                    "2026-10-22T10:00:00", "2026-10-22T10:00:00",
                    "2026-10-22T10:30:00", "{}", "{}", "[]",
                ),
            )
            for cfg_index in range(3):
                config_id = str(uuid.uuid4())
                config_ids.append(config_id)
                # beta is slightly better at coding, worse at reasoning.
                base = 0.85 + 0.03 * cfg_index + (0.05 if model_index == 1 else 0.0)
                quality = {
                    "overall": round(base, 3),
                    "task_completion": round(base, 3),
                    "factual_consistency": round(base + 0.02, 3),
                    "format_compliance": round(base - 0.01, 3),
                    "coding_correctness": round(
                        base + (0.06 if model_index == 1 else 0.0), 3
                    ),
                    "no_truncation": 1.0,
                    "no_malformed": 1.0,
                    "confident": True,
                    "details": {},
                    "checks_passed": 5,
                    "checks_total": 6,
                }
                speed = 5.0 + cfg_index + (2.0 if model_index == 0 else 0.0)
                metrics = [
                    {
                        "test_name": "short_instruction",
                        "category": "instruction",
                        "success": True,
                        "generation_tok_s": speed,
                        "prompt_tok_s": speed * 10,
                        "estimated_ttft_ms": 400.0,
                        "completion_tokens": 200,
                    },
                    {
                        "test_name": "coding_task",
                        "category": "coding",
                        "success": True,
                        "generation_tok_s": speed * 0.9,
                        "prompt_tok_s": speed * 9,
                        "estimated_ttft_ms": 450.0,
                        "completion_tokens": 300,
                    },
                ]
                conn.execute(
                    """
                    INSERT INTO configurations (
                        id, run_id, config_json, context_length, status,
                        metrics_json, quality_json, stability_score,
                        peak_vram_gb, peak_ram_gb, score, tested_at, duration_ms
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        config_id,
                        run_id,
                        json.dumps({"context_length": 8192, "flash_attention": True}),
                        8192,
                        "passed",
                        json.dumps(metrics),
                        json.dumps(quality),
                        0.98,
                        5.5,
                        20.0,
                        round(base, 3),
                        "2026-10-22T10:10:00",
                        1000,
                    ),
                )
        conn.commit()
    return run_ids, config_ids


@pytest.mark.asyncio
async def test_full_judge_flow_heuristic(tmp_path):
    from lm_optimizer.services.judge_prompt import JudgeRequest
    from lm_optimizer.services.judge_service import JudgeService

    db_path = tmp_path / "judge.db"
    run_ids, config_ids = _seed_db(db_path)

    service = JudgeService(db_path=db_path)
    request = JudgeRequest(
        run_ids=run_ids,
        use_case="coding",
        backend="heuristic",
        judge_model="heuristic",
        priority="balanced",
    )
    verdict = await service.run_judge(request)
    payload = verdict.to_dict()

    # Real contract of the heuristic backend (the plan's older shape was
    # superseded): one recommended config, reasoning, confidence, pros/cons.
    assert payload["recommended_config_id"]
    assert payload["reasoning"]
    assert payload["confidence"] > 0
    assert payload["pros"]
    assert "warnings" in payload
    assert "alternative_config_ids" in payload

    # The recommended config must belong to one of the seeded runs.
    with sqlite3.connect(db_path) as conn:
        owner = conn.execute(
            "SELECT run_id FROM configurations WHERE id = ?",
            (payload["recommended_config_id"],),
        ).fetchone()
    assert owner is not None
    assert owner[0] in run_ids

    # Speed and quality must be real numbers, not N/A (they were dropped by
    # the old _build_comparison wiring).
    assert not any("N/A" in pro for pro in payload["pros"]), payload["pros"]

    # A row was persisted.
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT id, run_ids_json, verdict_json FROM judge_verdicts").fetchall()
    assert len(rows) == 1
    stored = json.loads(rows[0][2])
    assert stored["recommended_config_id"] == payload["recommended_config_id"]
    assert json.loads(rows[0][1]) == run_ids


@pytest.mark.asyncio
async def test_judge_rejects_empty_request(tmp_path):
    from lm_optimizer.services.judge_service import JudgeService

    db_path = tmp_path / "empty.db"
    DatabaseManager(db_path)
    service = JudgeService(db_path=db_path)
    with pytest.raises(Exception):
        await service.run_judge({"run_ids": [], "use_case": "general"})
