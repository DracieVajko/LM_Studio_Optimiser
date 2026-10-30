"""Scheduled artifact test (fourth option): validation, execution, history."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from lm_optimizer.domain.models import (
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    HardwareInfo,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
)
from lm_optimizer.services.optimizer import (
    AdaptiveOptimizer,
    OptimizationState,
    build_resolved_settings,
    validate_artifact_test_id,
)


def _run(**kw):
    run = OptimizationRun(
        model=ModelIdentity(id="m", name="m", context_limit=32768),
        hardware=HardwareInfo(
            os="w", cpu_name="c", cpu_cores_physical=1,
            cpu_cores_logical=1, total_ram_gb=32, gpu_count=0,
        ),
        profile=OptimizationProfile.BALANCED,
    )
    for key, value in kw.items():
        setattr(run, key, value)
    return run


def _best():
    return ConfigurationResult(
        config=LoadConfiguration(context_length=4096, flash_attention=True),
        context_length=4096,
        status=ConfigurationStatus.PASSED,
        score=0.9,
        metrics=[
            BenchmarkMetrics(
                test_name="short_instruction", category="speed", success=True,
                generation_tok_s=10.0, completion_tokens=64,
            )
        ],
    )


# --- validation -------------------------------------------------------------

def test_builtin_ids_accepted():
    assert validate_artifact_test_id("web_dashboard") == "web_dashboard"
    assert validate_artifact_test_id("blender_skyblock") == "blender_skyblock"
    assert validate_artifact_test_id(None) is None


def test_unknown_id_fails_fast():
    with pytest.raises(ValueError) as exc:
        validate_artifact_test_id("nope")
    assert "unknown artifact test" in str(exc.value)


def test_custom_missing_file_fails():
    with pytest.raises(ValueError) as exc:
        validate_artifact_test_id("custom:does-not-exist.json")
    assert "not found" in str(exc.value)


def test_custom_valid_json_is_accepted(tmp_path):
    p = tmp_path / "task.json"
    p.write_text(
        json.dumps({"prompt": "write a haiku about GPUs", "kind": "text"}),
        encoding="utf-8",
    )
    assert validate_artifact_test_id(f"custom:{p}") == f"custom:{p}"


def test_custom_invalid_json_fails(tmp_path):
    p = tmp_path / "task.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        validate_artifact_test_id(f"custom:{p}")
    assert "valid JSON" in str(exc.value)


# --- freeze / resume --------------------------------------------------------

def test_artifact_selector_is_in_resolved_settings():
    run = _run(artifact_test_id="web_dashboard")
    snap = build_resolved_settings(run)
    assert snap["artifact_test_id"] == "web_dashboard"


# --- execution --------------------------------------------------------------

def _optimizer(run, best):
    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    opt.state = OptimizationState(run=run, search_space=MagicMock())
    opt.state.run.search_space = {}
    opt.state.best_config = best
    opt.state.tested_configs = [best]
    run.configurations = [best]
    run.best_config_id = best.id
    opt.client = MagicMock()
    opt._log_event = lambda *a, **k: None
    opt._checkpoint = lambda *a, **k: None
    opt._sync_phase_params = lambda: None
    opt._broadcast = AsyncMock()
    opt._pause_point = AsyncMock()
    return opt


@pytest.fixture(autouse=True)
def _no_repo_writes(monkeypatch):
    from lm_optimizer.services import optimizer as _opt

    monkeypatch.setattr(_opt.run_repo, "save", lambda r: str(r.id))


def test_finalize_runs_selected_artifact_test(monkeypatch):
    from lm_optimizer.services import sandbox_runner

    calls = {}

    async def fake_run(client, model_id, cfg, task, run_id="", **kw):
        calls["task"] = task.id
        calls["config"] = cfg
        return {
            "task_id": task.id,
            "task_kind": task.kind.value,
            "success": True,
            "result": {"penalized_score": 0.86, "artifacts": ["index.html"]},
            "config_applied": cfg.to_dict(),
            "unloaded": True,
            "duration_s": 12.0,
        }

    monkeypatch.setattr(sandbox_runner, "run_sandbox_task", fake_run)
    best = _best()
    run = _run(artifact_test_id="web_dashboard", benchmark_params={"fingerprint": "fp"})
    opt = _optimizer(run, best)

    asyncio.run(opt._finalize())

    assert calls["task"] == "web_dashboard"
    payload = run.benchmark_params["artifact_test"]
    assert payload["task_id"] == "web_dashboard"
    assert payload["config_id"] == str(best.id)
    assert payload["success"] is True
    assert payload["score"] == pytest.approx(0.86)
    assert payload["config_applied"]["context_length"] == 4096
    assert payload["unloaded"] is True
    # never mutates optimizer scores
    assert best.score == 0.9


def test_finalize_skips_artifact_when_not_selected(monkeypatch):
    from lm_optimizer.services import sandbox_runner

    async def boom(*a, **k):  # pragma: no cover
        raise AssertionError("must not run")

    monkeypatch.setattr(sandbox_runner, "run_sandbox_task", boom)
    run = _run(artifact_test_id=None)
    opt = _optimizer(run, _best())
    asyncio.run(opt._finalize())
    assert "artifact_test" not in run.benchmark_params


def test_manual_review_custom_prompt_has_no_score(monkeypatch, tmp_path):
    from lm_optimizer.services import sandbox_runner

    path = tmp_path / "task.json"
    path.write_text(json.dumps({"prompt": "say hello", "kind": "text"}), encoding="utf-8")

    async def fake_manual(client, model_id, cfg, prompt, **kw):
        return {
            "task_id": "custom",
            "success": True,
            "manual_review_only": True,
            "quality_score": None,
            "output_text": "hello",
            "config_applied": cfg.to_dict(),
            "unloaded": True,
        }

    monkeypatch.setattr(sandbox_runner, "run_manual_prompt", fake_manual)
    run = _run(
        artifact_test_id=f"custom:{path}",
        benchmark_params={"fingerprint": "fp"},
    )
    opt = _optimizer(run, _best())
    asyncio.run(opt._finalize())

    payload = run.benchmark_params["artifact_test"]
    assert payload["manual_review_only"] is True
    assert payload["score"] is None
    assert payload["success"] is True


# --- API surface ------------------------------------------------------------

def test_api_run_exposes_artifact_summary():
    from lm_optimizer.api.routes import _convert_run

    run = _run(
        benchmark_params={
            "phase_ab": {"phase": "final_validation"},
            "artifact_test": {
                "task_id": "web_dashboard",
                "success": True,
                "score": 0.86,
            },
        }
    )
    resp = _convert_run(run)
    assert resp.artifact_test["task_id"] == "web_dashboard"
    assert resp.artifact_test["score"] == pytest.approx(0.86)


def test_api_schema_accepts_artifact_selector():
    from lm_optimizer.api.schemas import AdvancedSettingsSchema

    s = AdvancedSettingsSchema(artifact_test_id="web_dashboard")
    assert s.artifact_test_id == "web_dashboard"
    assert AdvancedSettingsSchema().artifact_test_id is None
