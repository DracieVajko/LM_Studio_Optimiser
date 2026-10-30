"""Final sandbox: winner-config execution, custom prompts, isolation (§6, §10)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from lm_optimizer.domain.models import (
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
)
from lm_optimizer.services.sandbox_lab import (
    BUILTIN_TASKS,
    CUSTOM_PROMPT_MAX_CHARS,
    SANDBOX_TASKS,
    SandboxCheck,
    SandboxTask,
    SandboxTaskKind,
    validate_custom_prompt,
)


# --- custom prompt schema ---------------------------------------------------

def test_valid_custom_prompt_normalises_limits():
    spec = validate_custom_prompt(
        {
            "id": "p1",
            "name": "Explain",
            "prompt": "Explain KV cache",
            "kind": "text",
            "max_output_tokens": 512,
            "temperature": 0.1,
            "timeout_s": 60,
        }
    )
    assert spec["id"] == "p1"
    assert spec["max_output_tokens"] == 512
    assert spec["manual_review_only"] is True
    assert spec["checks"] == []


def test_custom_prompt_with_checks_is_scored():
    spec = validate_custom_prompt(
        {
            "prompt": "Return JSON with keys a and b",
            "kind": "text",
            "checks": [{"name": "keys", "kind": "json", "value": ["a", "b"]}],
        }
    )
    assert spec["manual_review_only"] is False
    assert len(spec["checks"]) == 1


@pytest.mark.parametrize(
    "spec, needle",
    [
        ({}, "non-empty"),
        ({"prompt": "x" * (CUSTOM_PROMPT_MAX_CHARS + 1)}, "limit"),
        ({"prompt": "x", "kind": "video"}, "kind"),
        ({"prompt": "x", "max_output_tokens": 0}, "max_output_tokens"),
        ({"prompt": "x", "max_output_tokens": 999999}, "max_output_tokens"),
        ({"prompt": "x", "timeout_s": 1}, "timeout_s"),
        ({"prompt": "x", "timeout_s": 99999}, "timeout_s"),
        ({"prompt": "x", "temperature": 9.0}, "temperature"),
        ({"prompt": "x", "checks": [{"name": "n"}]}, "kind"),
        ({"prompt": "x", "checks": "nope"}, "list"),
    ],
)
def test_invalid_custom_prompts_are_rejected(spec, needle):
    with pytest.raises(ValueError) as exc:
        validate_custom_prompt(spec)
    assert needle in str(exc.value)


def test_builtin_registry_is_single_source_of_truth():
    assert BUILTIN_TASKS is SANDBOX_TASKS
    assert set(BUILTIN_TASKS) >= {
        "web_dashboard",
        "research_synthesis",
        "blender_skyblock",
    }


# --- fake chat client -------------------------------------------------------

def _client(load_ok: bool = True, content: str = "hello"):
    client = MagicMock()
    client.ensure_unloaded = AsyncMock(return_value=True)
    loaded = MagicMock()
    loaded.success = load_ok
    loaded.error = None if load_ok else "boom"
    loaded.loaded_config = LoadConfiguration(context_length=4096, flash_attention=True)
    loaded.echo_mismatches = None
    client.load_model = AsyncMock(return_value=loaded)
    client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": content}}]}
    )
    return client


def _task(checks=None):
    return SandboxTask(
        id="custom_text",
        kind=SandboxTaskKind.TEXT,
        title="t",
        prompt="Write the word hello",
        checks=checks
        or [SandboxCheck("file", "files", ["output.txt"], critical=True)],
        max_repair_attempts=0,
        artifacts=("output.txt",),
    )


def test_run_sandbox_task_loads_exact_config_and_unloads(tmp_path):
    from lm_optimizer.services.sandbox_runner import run_sandbox_task

    client = _client()
    cfg = LoadConfiguration(context_length=8192, flash_attention=True)
    result = asyncio.run(
        run_sandbox_task(
            client,
            "m",
            cfg,
            _task(),
            run_id="run1",
            base_dir=tmp_path,
        )
    )
    assert result["success"] is True
    assert result["unloaded"] is True
    assert result["config_requested"]["context_length"] == 8192
    assert result["config_applied"]["context_length"] == 4096  # server echo wins
    # The load used the optimizer's exact config object.
    client.load_model.assert_awaited_once()
    assert client.load_model.await_args.args[1] is cfg
    # Unloaded in finally, exactly once per boundary.
    assert client.ensure_unloaded.await_count >= 2
    assert str(tmp_path) in result["workdir"]


def test_run_sandbox_task_reports_load_failure_without_raising(tmp_path):
    from lm_optimizer.services.sandbox_runner import run_sandbox_task

    result = asyncio.run(
        run_sandbox_task(
            _client(load_ok=False), "m", LoadConfiguration(), _task(),
            run_id="r", base_dir=tmp_path,
        )
    )
    assert result["success"] is False
    assert "load failed" in result["error"]
    assert result["unloaded"] is True


def test_run_manual_prompt_never_synthesises_a_score(tmp_path):
    from lm_optimizer.services.sandbox_runner import run_manual_prompt

    result = asyncio.run(
        run_manual_prompt(
            _client(content="the answer"),
            "m",
            LoadConfiguration(context_length=2048),
            "Explain",
            run_id="r",
            base_dir=tmp_path,
        )
    )
    assert result["manual_review_only"] is True
    assert result["quality_score"] is None
    assert result["success"] is True
    assert result["output_text"] == "the answer"
    assert (tmp_path / "r" / "custom").exists()


# --- optimizer sandbox stage ------------------------------------------------

def _optimizer_for_sandbox(run, tmp_path):
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


def _winner():
    return ConfigurationResult(
        config=LoadConfiguration(context_length=4096, flash_attention=True),
        context_length=4096,
        status=ConfigurationStatus.PASSED,
        score=0.91,
        metrics=[
            BenchmarkMetrics(
                test_name="t", category="speed", success=True,
                generation_tok_s=9.0, completion_tokens=32,
            )
        ],
    )


def _run(**kw):
    run = OptimizationRun(
        model=ModelIdentity(id="m", name="m", context_limit=32768),
        profile=OptimizationProfile.BALANCED,
    )
    run.id = "11111111-1111-1111-1111-111111111111" if hasattr(run.id, "__str__") else run.id
    for key, value in kw.items():
        setattr(run, key, value)
    run.pareto_config_ids = []
    return run


@pytest.fixture(autouse=True)
def _no_repo_writes(monkeypatch):
    from lm_optimizer.services import optimizer as _opt

    monkeypatch.setattr(_opt.run_repo, "save", lambda r: str(r.id))


def test_all_mode_runs_every_registered_builtin(monkeypatch, tmp_path):
    from lm_optimizer.services import sandbox_runner

    seen = []

    async def fake_run(client, model_id, cfg, task, run_id="", **kw):
        seen.append(task.id)
        return {
            "task_id": task.id,
            "success": True,
            "config_applied": cfg.to_dict(),
            "unloaded": True,
        }

    monkeypatch.setattr(sandbox_runner, "run_sandbox_task", fake_run)
    run = _run(final_sandbox_mode="all")
    winner = _winner()
    opt = _optimizer_for_sandbox(run, tmp_path)

    asyncio.run(opt._run_final_sandbox(winner))

    assert set(seen) == set(SANDBOX_TASKS)
    payload = run.benchmark_params["final_sandbox"]
    assert payload["mode"] == "all"
    assert payload["affects_optimizer_score"] is False
    assert payload["winner_config_id"] == str(winner.id)
    assert len(payload["tasks"]) == len(SANDBOX_TASKS)
    # Sandbox never mutates optimizer scores/Pareto.
    assert winner.score == 0.91
    assert run.pareto_config_ids == []


def test_invalid_custom_prompt_is_recorded_not_fatal(monkeypatch, tmp_path):
    from lm_optimizer.services import sandbox_runner

    async def failing(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("no task should run")

    monkeypatch.setattr(sandbox_runner, "run_sandbox_task", failing)
    monkeypatch.setattr(sandbox_runner, "run_manual_prompt", failing)
    run = _run(
        final_sandbox_mode="custom",
        final_sandbox_custom_prompts=[
            {"prompt": ""},
            {"prompt": "x", "kind": "video"},
        ],
    )
    opt = _optimizer_for_sandbox(run, tmp_path)

    asyncio.run(opt._run_final_sandbox(_winner()))

    payload = run.benchmark_params["final_sandbox"]
    assert payload["tasks"] == []
    assert len(payload["invalid_custom_prompts"]) == 2
    assert "non-empty" in payload["invalid_custom_prompts"][0]["error"]
    assert "kind" in payload["invalid_custom_prompts"][1]["error"]


def test_manual_review_prompt_records_output_and_no_score(monkeypatch, tmp_path):
    from lm_optimizer.services import sandbox_runner

    called = {}

    async def fake_manual(client, model_id, cfg, prompt, **kw):
        called["prompt"] = prompt
        return {
            "task_id": "custom",
            "success": True,
            "manual_review_only": True,
            "quality_score": None,
            "output_text": "ok",
            "config_applied": cfg.to_dict(),
            "unloaded": True,
        }

    monkeypatch.setattr(sandbox_runner, "run_manual_prompt", fake_manual)
    run = _run(
        final_sandbox_mode="custom",
        final_sandbox_custom_prompts=[
            {"id": "p", "prompt": "Say hello", "max_output_tokens": 64}
        ],
    )
    opt = _optimizer_for_sandbox(run, tmp_path)

    asyncio.run(opt._run_final_sandbox(_winner()))

    assert called["prompt"] == "Say hello"
    task = run.benchmark_params["final_sandbox"]["tasks"][0]
    assert task["result"]["quality_score"] is None
    assert task["result"]["manual_review_only"] is True


def test_one_mode_runs_only_the_selected_preset(monkeypatch, tmp_path):
    from lm_optimizer.services import sandbox_runner

    seen = []

    async def fake_run(client, model_id, cfg, task, run_id="", **kw):
        seen.append(task.id)
        return {"task_id": task.id, "success": True, "unloaded": True}

    monkeypatch.setattr(sandbox_runner, "run_sandbox_task", fake_run)
    run = _run(
        final_sandbox_mode="one",
        final_sandbox_preset_ids=["web_dashboard", "research_synthesis"],
    )
    opt = _optimizer_for_sandbox(run, tmp_path)

    asyncio.run(opt._run_final_sandbox(_winner()))
    assert seen == ["web_dashboard"]
