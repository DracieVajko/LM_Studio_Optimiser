"""Single-model deep runner + CLI (Task 2).

TDD: mocked client only, no live server. The DB fixture in conftest routes
run lookups to a tmp database; report output defaults under the (patched)
configured results dir so tests never touch the real results/.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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


def _seed_best_run(model_id="m"):
    from lm_optimizer.database import repositories as _repo

    hw = HardwareInfo(
        os="T",
        cpu_name="C",
        cpu_cores_physical=1,
        cpu_cores_logical=2,
        total_ram_gb=8,
        gpu_count=0,
    )
    model = ModelIdentity(id=model_id, name=model_id, context_limit=131072)
    _repo.hardware_repo.save(hw)
    _repo.model_repo.save(model)
    run = OptimizationRun(model=model, hardware=hw, profile=OptimizationProfile.BALANCED)
    _repo.run_repo.save(run)
    best = ConfigurationResult(
        config=LoadConfiguration(
            context_length=4096,
            flash_attention=True,
            offload_kv_cache_to_gpu=True,
            eval_batch_size=256,
        ),
        context_length=4096,
        status=ConfigurationStatus.PASSED,
        metrics=[
            BenchmarkMetrics(
                test_name="seed",
                category="instruction",
                success=True,
                generation_tok_s=30.0,
                prompt_tok_s=500.0,
                estimated_ttft_ms=100.0,
                prompt_tokens=10,
                completion_tokens=50,
                total_tokens=60,
                output_text="seeded best",
            )
        ],
        quality_score=QualityScore(
            overall=1.0,
            task_completion=1.0,
            factual_consistency=1.0,
            format_compliance=1.0,
            coding_correctness=1.0,
            no_truncation=1.0,
            no_malformed=1.0,
            checks_passed=6,
            checks_total=6,
        ),
        score=9.1,
    )
    best.run_id = run.id
    _repo.config_repo.save(best)
    run.configurations = [best]
    run.best_config_id = best.id
    _repo.run_repo.save(run)
    return run


def _mock_client(thinking="mock reasoning trace", fail=False, load_fail=False):
    c = MagicMock()
    ok = MagicMock()
    ok.success = not load_fail
    ok.identifier = "mock-id"
    ok.loaded_config = None
    ok.error = "mock load refused" if load_fail else None
    c.load_model = AsyncMock(return_value=ok)
    c.ensure_unloaded = AsyncMock(return_value=True)
    c.unload_all = AsyncMock(return_value={})
    c.get_loaded_instances = AsyncMock(return_value=[])
    c.model_known_no_reasoning = lambda model_id: False

    async def _chat(**kwargs):
        if fail:
            raise RuntimeError("mock generation exploded")
        return {
            "choices": [
                {"message": {"content": "Line 1\nLine 2 compass\nLine 3\nLine 4."}}
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            "_stats": {"tokens_per_second": 25.0, "time_to_first_token_seconds": 0.05},
            "thinking_text": thinking,
        }

    c.chat_completion = AsyncMock(side_effect=_chat)
    return c


@pytest.fixture
def mock_client_with_best():
    _seed_best_run("m")
    return _mock_client()


def test_deep_run_uses_best_config_and_logs_thinking(mock_client_with_best):
    from lm_optimizer.services.deep import run_deep_model

    out = run_deep_model(mock_client_with_best, "m")
    assert out["tasks"][0]["thinking_chars"] >= 0
    assert "deep" in out["report_path"]
    # Best config from the DB was actually used (never a silent default).
    cfgs = [call.args[1] for call in mock_client_with_best.load_model.call_args_list]
    assert cfgs, "expected the model to be loaded with the stored best config"
    assert all(getattr(cfg, "eval_batch_size", None) == 256 for cfg in cfgs)
    # Single load per model: all 5 tasks run under that one load.
    assert mock_client_with_best.load_model.call_count == 1
    assert mock_client_with_best.chat_completion.call_count == 5
    assert mock_client_with_best.ensure_unloaded.call_count >= 1
    assert out["status"] == "completed"


def test_deep_run_refuses_without_best_config():
    from lm_optimizer.services.deep import run_deep_model

    client = _mock_client()
    with pytest.raises(ValueError, match="no stored best config"):
        run_deep_model(client, "never-optimized-model")
    client.load_model.assert_not_called()


def test_deep_run_aborts_after_three_consecutive_errors():
    from lm_optimizer.services.deep import run_deep_model

    _seed_best_run("m")
    client = _mock_client(fail=True)
    out = run_deep_model(client, "m")
    assert out["status"] == "aborted"
    assert "consecutive" in (out["reason"] or "").lower()
    # Exactly 1 load; 3 tasks attempted, remainder aborted (skipped).
    assert client.load_model.call_count == 1
    assert client.chat_completion.call_count == 3
    assert [t["status"] for t in out["tasks"]] == ["failed"] * 3 + ["skipped"] * 2
    assert "deep" in out["report_path"]


def test_deep_run_load_failure_fails_tasks_without_chats():
    from lm_optimizer.services.deep import run_deep_model

    _seed_best_run("m")
    client = _mock_client(load_fail=True)
    out = run_deep_model(client, "m")
    assert out["status"] == "failed"
    assert out["reason"], "expected the load error as reason"
    assert client.load_model.call_count == 1
    assert client.chat_completion.call_count == 0
    assert [t["status"] for t in out["tasks"]] == ["failed"] * 5
    assert "deep" in out["report_path"]


def test_deep_run_unload_failure_raises_instead_of_silently_passing():
    from lm_optimizer.config import config as _cfg
    from lm_optimizer.services.deep import run_deep_model
    from lm_optimizer.services.unload_guard import UnloadNotClean

    _seed_best_run("m")
    client = _mock_client()
    client.ensure_unloaded = AsyncMock(return_value=False)
    with pytest.raises(UnloadNotClean, match="[Hh]ost not clean"):
        run_deep_model(client, "m")
    # Evidence is still preserved: the per-model report was written first.
    deep_dir = Path(_cfg.storage.results_dir) / "deep"
    assert list(deep_dir.glob("m-deep-*.md")), "expected the deep report despite unload failure"


def test_deep_run_caps_output_tokens_per_task():
    from lm_optimizer.services.deep import MAX_OUTPUT_TOKENS_PER_TASK, cap_deep_cases
    from lm_optimizer.domain.models import BenchmarkCase

    big = BenchmarkCase(
        name="x", category="recall", prompt="hi", max_tokens=100000, temperature=0.0
    )
    capped = cap_deep_cases([big])
    assert capped[0].max_tokens == MAX_OUTPUT_TOKENS_PER_TASK


def test_deep_cli_registered_with_model_option():
    import typer

    from lm_optimizer.cli.main import app

    info = typer.main.get_command(app)
    assert "deep" in info.commands, "CLI must expose a 'deep' command"
    params = [p.name for p in info.commands["deep"].params]
    assert "model" in params, "deep must accept --model"


# --- Task 3: multi-model batch with resume (TDD: must FAIL before impl) ---


@pytest.fixture
def mock_client_one_fail():
    from unittest.mock import AsyncMock, MagicMock

    _seed_best_run("m1")
    _seed_best_run("m2")
    c = _mock_client()
    ok = MagicMock()
    ok.success = True
    ok.identifier = "mock-id"
    ok.loaded_config = None
    ok.error = None
    bad = MagicMock()
    bad.success = False
    bad.identifier = None
    bad.loaded_config = None
    bad.error = "mock load refused for m1"

    async def _load(model_id, cfg=None, **kwargs):
        if model_id == "m1":
            return bad
        return ok

    c.load_model = AsyncMock(side_effect=_load)
    return c


def test_batch_continues_after_model_failure(mock_client_one_fail):
    from lm_optimizer.services.deep import run_deep_batch

    out = run_deep_batch(mock_client_one_fail, ["m1", "m2"])
    assert out["models"]["m1"]["status"] == "failed"
    assert out["models"]["m2"]["status"] == "completed"


def test_batch_skips_model_without_best_config():
    from lm_optimizer.services.deep import run_deep_batch

    _seed_best_run("has-best")
    client = _mock_client()
    out = run_deep_batch(client, ["has-best", "never-optimized-model"])
    assert out["models"]["has-best"]["status"] == "completed"
    assert out["models"]["never-optimized-model"]["status"] == "skipped"
    assert "no stored best" in (out["models"]["never-optimized-model"]["reason"] or "").lower()


def test_batch_resume_reuses_completed_report():
    from lm_optimizer.services.deep import run_deep_batch

    _seed_best_run("r1")
    client = _mock_client()
    first = run_deep_batch(client, ["r1"])
    assert first["models"]["r1"]["status"] == "completed"
    n_loads = client.load_model.call_count
    assert n_loads == 1

    # Second run with a client that would explode if asked to measure.
    client2 = _mock_client(fail=True)
    client2.load_model = AsyncMock(side_effect=AssertionError("must not reload resumed model"))
    client2.chat_completion = AsyncMock(side_effect=AssertionError("must not re-measure"))
    second = run_deep_batch(client2, ["r1"])
    assert second["models"]["r1"]["report_path"] == first["models"]["r1"]["report_path"]
    assert second["models"]["r1"].get("resumed") is True


def test_batch_unload_failure_records_and_continues():
    from unittest.mock import AsyncMock

    from lm_optimizer.services.deep import run_deep_batch

    _seed_best_run("u1")
    _seed_best_run("u2")
    client = _mock_client()
    orig_unloaded = client.ensure_unloaded

    async def _unloaded(model_id=None, **kwargs):
        if model_id == "u1":
            return False  # force UnloadNotClean from the single-model runner
        return True

    client.ensure_unloaded = AsyncMock(side_effect=_unloaded)
    out = run_deep_batch(client, ["u1", "u2"])
    assert out["models"]["u1"]["status"] == "failed"
    assert "unload" in (out["models"]["u1"]["reason"] or "").lower()
    assert out["models"]["u2"]["status"] == "completed"


def test_batch_dirty_host_skips_remaining():
    from unittest.mock import AsyncMock

    from lm_optimizer.services.deep import run_deep_batch

    _seed_best_run("d1")
    _seed_best_run("d2")
    client = _mock_client()
    client.ensure_unloaded = AsyncMock(return_value=False)
    client.get_loaded_instances = AsyncMock(
        return_value=[{"model": "d1", "instance_id": "stub-1"}]
    )
    out = run_deep_batch(client, ["d1", "d2"])
    assert out["models"]["d1"]["status"] == "failed"
    assert out["models"]["d2"]["status"] == "skipped"
    assert "not clean" in (out["models"]["d2"]["reason"] or "").lower()


def test_batch_writes_summary_md():
    from pathlib import Path

    from lm_optimizer.services.deep import run_deep_batch

    _seed_best_run("s1")
    client = _mock_client()
    out = run_deep_batch(client, ["s1"])
    assert "summary_path" in out
    p = Path(out["summary_path"])
    assert p.exists(), "batch must write a summary .md"
    text = p.read_text(encoding="utf-8")
    assert "s1" in text
    text.encode("ascii")  # ASCII-only reports


def test_deep_cli_has_models_and_all_optimized_options():
    import typer

    from lm_optimizer.cli.main import app

    info = typer.main.get_command(app)
    params = [p.name for p in info.commands["deep"].params]
    assert "models" in params, "deep must accept --models"
    assert "all_optimized" in params, "deep must accept --all-optimized"


def test_list_optimized_models_resolves_only_with_best():
    from lm_optimizer.services.deep import list_optimized_models

    _seed_best_run("opt-a")
    got = list_optimized_models()
    assert "opt-a" in got
    assert "never-optimized-model" not in got


# --- Task 9: Deep Research default prompts YAML + example (TDD) ---


def test_load_deep_prompts_yaml_from_cwd(tmp_path, monkeypatch):
    """YAML file in CWD takes priority over config/ and defaults."""
    from lm_optimizer.services.deep import load_deep_prompts

    yaml_content = """prompts:
  - name: "Custom Recall Test"
    prompt: "Custom recall prompt content"
    category: "recall"
    max_tokens: 4096
  - name: "multi_hop"
    prompt: "Override multi-hop prompt"
    category: "reasoning"
    max_tokens: 2048
"""
    (tmp_path / "deep_prompts.yaml").write_text(yaml_content, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    prompts = load_deep_prompts()
    # Should have 2 prompts: custom recall + overridden multi_hop (others from defaults)
    names = [p["name"] for p in prompts]
    assert "Custom Recall Test" in names
    assert "multi_hop" in names
    # Override should take effect
    multi_hop = next(p for p in prompts if p["name"] == "multi_hop")
    assert multi_hop["prompt"] == "Override multi-hop prompt"


def test_load_deep_prompts_yaml_from_config_dir(tmp_path, monkeypatch):
    """YAML file in config/ is used when CWD has none."""
    from lm_optimizer.services.deep import load_deep_prompts
    from lm_optimizer.config import config as app_config

    # Create config dir with YAML
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    yaml_content = """prompts:
  - name: "Config Dir Test"
    prompt: "Prompt from config dir"
    category: "custom"
    max_tokens: 1024
"""
    (config_dir / "deep_prompts.yaml").write_text(yaml_content, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # Patch the config's storage.config_dir to point to our temp config dir
    monkeypatch.setattr(app_config.storage, "config_dir", str(config_dir))

    prompts = load_deep_prompts()
    names = [p["name"] for p in prompts]
    assert "Config Dir Test" in names


def test_load_deep_prompts_falls_back_to_defaults(monkeypatch, tmp_path):
    """When no YAML files exist, bundled defaults are returned."""
    from lm_optimizer.services.deep import load_deep_prompts, DEEP_PROMPT_DEFAULTS

    monkeypatch.chdir(tmp_path)
    # Ensure no YAML files exist
    assert not (tmp_path / "deep_prompts.yaml").exists()

    prompts = load_deep_prompts()
    # Should return all 5 built-in defaults
    names = [p["name"] for p in prompts]
    assert names == ["long_context_recall", "multi_hop", "json_discipline", "coding_precision", "instruction_follow"]
    # Content should match defaults
    for p in prompts:
        default = next(d for d in DEEP_PROMPT_DEFAULTS if d["name"] == p["name"])
        assert p["prompt"] == default["prompt"]
        assert p["category"] == default["category"]
        assert p["max_tokens"] == default["max_tokens"]


def test_load_deep_prompts_override_by_name():
    """Override replaces by name; other defaults preserved."""
    from lm_optimizer.services.deep import load_deep_prompts, DEEP_PROMPT_DEFAULTS

    # Create a YAML that only overrides one prompt
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        yaml_content = """prompts:
  - name: "json_discipline"
    prompt: "Custom JSON prompt"
    category: "format"
    max_tokens: 512
"""
        (td_path / "deep_prompts.yaml").write_text(yaml_content, encoding="utf-8")
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(td_path)
            prompts = load_deep_prompts()
        finally:
            os.chdir(old_cwd)

    # Should have all 5 prompts
    names = [p["name"] for p in prompts]
    assert len(names) == 5
    # json_discipline should be overridden
    json_prompt = next(p for p in prompts if p["name"] == "json_discipline")
    assert json_prompt["prompt"] == "Custom JSON prompt"
    assert json_prompt["max_tokens"] == 512
    # Others should be defaults
    for p in prompts:
        if p["name"] != "json_discipline":
            default = next(d for d in DEEP_PROMPT_DEFAULTS if d["name"] == p["name"])
            assert p["prompt"] == default["prompt"]
            assert p["category"] == default["category"]
            assert p["max_tokens"] == default["max_tokens"]


def test_run_deep_model_async_accepts_custom_prompts():
    """run_deep_model_async accepts a custom prompts list and uses it instead of defaults."""
    from lm_optimizer.services.deep import run_deep_model_async
    from unittest.mock import AsyncMock, MagicMock

    # This test will fail until we modify run_deep_model_async to accept custom prompts
    # For now, we just check the signature accepts the parameter
    import inspect
    sig = inspect.signature(run_deep_model_async)
    assert "prompts" in sig.parameters, "run_deep_model_async must accept 'prompts' parameter"


def test_run_deep_model_uses_custom_prompts(mock_client_with_best):
    """Custom prompts passed to run_deep_model_async are used for task execution."""
    import asyncio
    from lm_optimizer.services.deep import run_deep_model_async

    client = mock_client_with_best

    # Custom prompts - only 2 tasks instead of 5
    custom_prompts = [
        {
            "name": "custom_task_1",
            "prompt": "Custom prompt 1",
            "category": "custom",
            "max_tokens": 1024,
            "temperature": 0.0,
        },
        {
            "name": "custom_task_2",
            "prompt": "Custom prompt 2",
            "category": "custom",
            "max_tokens": 1024,
            "temperature": 0.0,
        },
    ]

    async def _test():
        out = await run_deep_model_async(
            client, "m", load_config=None, out_dir="results/deep", prompts=custom_prompts
        )
        # Should only have 2 tasks
        assert len(out["tasks"]) == 2
        assert out["tasks"][0]["name"] == "custom_task_1"
        assert out["tasks"][1]["name"] == "custom_task_2"
        # Verify chat_completion was called with custom prompts (2 calls)
        calls = client.chat_completion.call_args_list
        assert len(calls) == 2
        # The prompt should be in the call kwargs (check both input_text and messages)
        for i, call in enumerate(calls):
            kwargs = call.kwargs
            prompt_found = False
            for key in ("input_text", "messages", "prompt"):
                if key in kwargs:
                    val = kwargs[key]
                    if isinstance(val, str) and custom_prompts[i]["prompt"] in val:
                        prompt_found = True
                        break
                    if isinstance(val, list):
                        for msg in val:
                            if isinstance(msg, dict) and "content" in msg:
                                if custom_prompts[i]["prompt"] in msg["content"]:
                                    prompt_found = True
                                    break
            assert prompt_found, f"Custom prompt {i} not found in call kwargs: {kwargs}"

    asyncio.run(_test())


def test_run_deep_batch_uses_custom_prompts(tmp_path):
    """Custom prompts passed to run_deep_batch_async are used for all models."""
    import asyncio
    from lm_optimizer.services.deep import run_deep_batch_async
    from lm_optimizer.tests.test_deep_run import _seed_best_run, _mock_client

    _seed_best_run("m1")
    _seed_best_run("m2")

    custom_prompts = [
        {
            "name": "batch_task_1",
            "prompt": "Batch custom prompt",
            "category": "custom",
            "max_tokens": 1024,
            "temperature": 0.0,
        },
    ]

    client = _mock_client()
    out_dir = tmp_path / "deep_test_batch"

    async def _test():
        out = await run_deep_batch_async(
            client, ["m1", "m2"], out_dir=str(out_dir), prompts=custom_prompts
        )
        # Both models should have run with 1 custom task
        assert out["models"]["m1"]["status"] == "completed"
        assert out["models"]["m2"]["status"] == "completed"
        # Check the task name
        # Note: We can't easily access tasks from batch output, but we can verify
        # the chat_completion was called correctly
        calls = client.chat_completion.call_args_list
        assert len(calls) == 2  # 1 per model
        for call in calls:
            kwargs = call.kwargs
            prompt_found = False
            for key in ("input_text", "messages", "prompt"):
                if key in kwargs:
                    val = kwargs[key]
                    if isinstance(val, str) and "Batch custom prompt" in val:
                        prompt_found = True
                        break
                    if isinstance(val, list):
                        for msg in val:
                            if isinstance(msg, dict) and "content" in msg:
                                if "Batch custom prompt" in msg["content"]:
                                    prompt_found = True
                                    break
            assert prompt_found, f"Batch custom prompt not found in call kwargs: {kwargs}"

    asyncio.run(_test())
