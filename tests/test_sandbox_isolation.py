"""Sandbox isolation, sequential execution and failure-safe unload (spec §6, §10)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from lm_optimizer.domain.models import (
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
)
from lm_optimizer.services.sandbox_lab import (
    SANDBOX_TASKS,
    SandboxTask,
    SandboxTaskKind,
)
from lm_optimizer.services.sandbox_runner import SandboxRunner


def _task(artifact: str) -> SandboxTask:
    return SandboxTask(
        id="t",
        kind=SandboxTaskKind.TEXT,
        title="t",
        prompt="p",
        artifacts=(artifact,),
        max_repair_attempts=0,
    )


def _runner(tmp_path: Path, artifact: str) -> SandboxRunner:
    from lm_optimizer.services.sandbox_lab import SandboxPolicy

    workdir = tmp_path / "sandbox"
    return SandboxRunner(
        task=_task(artifact),
        policy=SandboxPolicy(workdir=workdir),
        workdir=workdir,
    )


# --- artifact isolation -----------------------------------------------------

def test_parent_traversal_artifact_is_contained(tmp_path):
    runner = _runner(tmp_path, "../escape.txt")
    runner._materialise("payload")

    assert (tmp_path / "sandbox" / "output.txt").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_absolute_artifact_path_is_contained(tmp_path):
    outside = tmp_path / "outside.txt"
    runner = _runner(tmp_path, str(outside))
    runner._materialise("payload")

    assert not outside.exists()
    assert (tmp_path / "sandbox" / "output.txt").read_text(encoding="utf-8") == "payload"


def test_nested_legitimate_artifact_stays_inside(tmp_path):
    runner = _runner(tmp_path, "sub/dir/page.html")
    runner._materialise("<html></html>")

    target = tmp_path / "sandbox" / "sub" / "dir" / "page.html"
    assert target.exists()
    assert target.is_relative_to((tmp_path / "sandbox").resolve())


# --- unload on every exit path ---------------------------------------------

def _client(chat_error: Exception | None = None):
    client = MagicMock()
    client.ensure_unloaded = AsyncMock(return_value=True)
    loaded = MagicMock()
    loaded.success = True
    loaded.error = None
    loaded.loaded_config = LoadConfiguration(context_length=4096)
    loaded.echo_mismatches = None
    client.load_model = AsyncMock(return_value=loaded)
    if chat_error is not None:
        client.chat_completion = AsyncMock(side_effect=chat_error)
    else:
        client.chat_completion = AsyncMock(
            return_value={"choices": [{"message": {"content": "x"}}]}
        )
    return client


def _check_task() -> SandboxTask:
    from lm_optimizer.services.sandbox_lab import SandboxCheck

    return SandboxTask(
        id="t",
        kind=SandboxTaskKind.TEXT,
        title="t",
        prompt="p",
        checks=[SandboxCheck("f", "files", ["output.txt"], critical=True)],
        max_repair_attempts=0,
        artifacts=("output.txt",),
    )


def test_run_sandbox_task_unloads_after_generation_error(tmp_path):
    from lm_optimizer.services.sandbox_runner import run_sandbox_task

    client = _client(chat_error=RuntimeError("engine died"))
    result = asyncio.run(
        run_sandbox_task(
            client, "m", LoadConfiguration(), _check_task(),
            run_id="r", base_dir=tmp_path,
        )
    )
    # The runner records the per-attempt error and still unloads.
    assert result["unloaded"] is True
    assert client.ensure_unloaded.await_count >= 2
    assert result["config_requested"] == {}


def test_run_manual_prompt_unloads_after_generation_error(tmp_path):
    from lm_optimizer.services.sandbox_runner import run_manual_prompt

    client = _client(chat_error=RuntimeError("boom"))
    result = asyncio.run(
        run_manual_prompt(
            client, "m", LoadConfiguration(), "p", run_id="r", base_dir=tmp_path
        )
    )
    assert result["unloaded"] is True
    assert result["quality_score"] is None
    assert result["success"] is False


# --- sequential, registry-ordered execution --------------------------------

def _optimizer(run):
    from lm_optimizer.services.optimizer import AdaptiveOptimizer, OptimizationState

    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    opt.client = MagicMock()
    opt.state = OptimizationState(run=run, search_space=MagicMock())
    opt.state.run.search_space = {}
    opt._log_event = lambda *a, **k: None
    opt._checkpoint = lambda *a, **k: None
    opt._sync_phase_params = lambda: None
    opt._broadcast = AsyncMock()
    opt._pause_point = AsyncMock()
    return opt


def test_all_mode_runs_tasks_sequentially_in_registry_order(monkeypatch):
    from lm_optimizer.services import optimizer as _opt
    from lm_optimizer.services import sandbox_runner

    monkeypatch.setattr(_opt.run_repo, "save", lambda r: str(r.id))
    order = []
    active = {"now": 0, "max": 0}

    async def fake_run(client, model_id, cfg, task, run_id="", **kw):
        order.append(task.id)
        active["now"] += 1
        active["max"] = max(active["max"], active["now"])
        await asyncio.sleep(0)
        active["now"] -= 1
        return {"task_id": task.id, "success": True, "unloaded": True}

    monkeypatch.setattr(sandbox_runner, "run_sandbox_task", fake_run)

    run = OptimizationRun(
        model=ModelIdentity(id="m", name="m"),
        profile=OptimizationProfile.BALANCED,
        final_sandbox_mode="all",
    )
    winner = ConfigurationResult(
        config=LoadConfiguration(context_length=4096),
        context_length=4096,
        status=ConfigurationStatus.PASSED,
    )
    asyncio.run(_optimizer(run)._run_final_sandbox(winner))

    assert order == list(SANDBOX_TASKS)
    assert active["max"] == 1  # never concurrent by default
    payload = run.benchmark_params["final_sandbox"]
    assert payload["affects_optimizer_score"] is False
    assert all(t["config"] == "winner" for t in payload["tasks"])


def test_sandbox_mode_off_does_nothing(monkeypatch):
    from lm_optimizer.services import optimizer as _opt
    from lm_optimizer.services import sandbox_runner

    monkeypatch.setattr(_opt.run_repo, "save", lambda r: str(r.id))
    called = []

    async def fake_run(*a, **k):
        called.append(1)

    monkeypatch.setattr(sandbox_runner, "run_sandbox_task", fake_run)
    run = OptimizationRun(
        model=ModelIdentity(id="m", name="m"),
        profile=OptimizationProfile.BALANCED,
        final_sandbox_mode="off",
    )
    winner = ConfigurationResult(
        config=LoadConfiguration(), context_length=4096,
        status=ConfigurationStatus.PASSED,
    )
    asyncio.run(_optimizer(run)._run_final_sandbox(winner))
    assert called == []
    assert "final_sandbox" not in run.benchmark_params
