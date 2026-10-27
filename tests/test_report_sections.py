"""Report sections for speed floor, context strategies and artifact test."""

from __future__ import annotations

from uuid import uuid4

from lm_optimizer.domain.models import (
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    HardwareInfo,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
    QualityScore,
)
from lm_optimizer.services.reporting import save_best_report


def _quality():
    return QualityScore(
        overall=0.99,
        task_completion=1.0,
        factual_consistency=1.0,
        format_compliance=1.0,
        coding_correctness=1.0,
        no_truncation=1.0,
        no_malformed=1.0,
        checks_passed=30,
        checks_total=30,
    )


def _run(*, floor_met=True, strategies=None, artifact=None):
    best = ConfigurationResult(
        id=uuid4(),
        config=LoadConfiguration(context_length=8192, flash_attention=True),
        context_length=8192,
        status=ConfigurationStatus.PASSED,
        score=0.9,
        quality_score=_quality(),
        metrics=[
            BenchmarkMetrics(
                test_name="short_instruction", category="instruction", success=True,
                generation_tok_s=12.0, completion_tokens=64,
            )
        ],
    )
    run = OptimizationRun(
        model=ModelIdentity(id="vendor/model", name="Model"),
        hardware=HardwareInfo(
            os="w", cpu_name="c", cpu_cores_physical=1,
            cpu_cores_logical=1, total_ram_gb=32, gpu_count=1,
        ),
        profile=OptimizationProfile.BALANCED,
    )
    run.configurations = [best]
    run.best_config_id = best.id
    run.search_space = {
        "speed_floor": {
            "value": 2.0,
            "enforced": True,
            "met": floor_met,
            "fastest_candidate_id": str(best.id),
            "fastest_tok_s": 1.4 if not floor_met else 12.0,
        }
    }
    pab = {"speed_context": 4096, "quality_context": 8192}
    if strategies is not None:
        pab["phase_b_result"] = {"context_strategies": strategies}
    run.benchmark_params = {"phase_ab": pab}
    if artifact is not None:
        run.benchmark_params["artifact_test"] = artifact
    return run


def test_report_shows_speed_floor_met(tmp_path):
    text = save_best_report(_run(floor_met=True), out_dir=tmp_path).read_text(encoding="utf-8")
    assert "## Speed floor" in text
    assert "Floor: 2.0 tok/s (enforced)" in text
    assert "- Met: True" in text
    assert "NO CONFIG MEETS SPEED FLOOR" not in text


def test_report_explains_no_winner_when_floor_not_met(tmp_path):
    text = save_best_report(_run(floor_met=False), out_dir=tmp_path).read_text(encoding="utf-8")
    assert "- Met: False" in text
    assert "NO CONFIG MEETS SPEED FLOOR" in text
    assert "diagnostic only" in text


def test_report_lists_context_strategies(tmp_path):
    strategies = {
        "winner": {"available": True, "max_stable_context": 8192, "max_usable_context": 8192},
        "model_first_kv_cpu": {"available": False, "reason": "capability not supported"},
    }
    text = save_best_report(
        _run(strategies=strategies), out_dir=tmp_path
    ).read_text(encoding="utf-8")
    assert "## Context strategies" in text
    assert "| winner |" in text
    assert "model_first_kv_cpu" in text
    assert "capability not supported" in text


def test_report_shows_artifact_test(tmp_path):
    artifact = {
        "task_id": "web_dashboard",
        "task_kind": "web",
        "success": True,
        "score": 0.86,
        "config_id": str(uuid4()),
        "unloaded": True,
    }
    text = save_best_report(
        _run(artifact=artifact), out_dir=tmp_path
    ).read_text(encoding="utf-8")
    assert "## Artifact test (post-run)" in text
    assert "web_dashboard" in text
    assert "Score: 0.86" in text
    assert "- Unloaded: True" in text


def test_report_manual_review_artifact_has_no_fake_score(tmp_path):
    artifact = {
        "task_id": "custom_abc",
        "task_kind": "text",
        "success": True,
        "manual_review_only": True,
        "score": None,
    }
    text = save_best_report(
        _run(artifact=artifact), out_dir=tmp_path
    ).read_text(encoding="utf-8")
    assert "manual review only" in text
    assert "Score: 0.0" not in text
