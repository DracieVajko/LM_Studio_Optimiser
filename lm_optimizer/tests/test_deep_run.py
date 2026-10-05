"""Single-model deep runner + CLI (Task 2).

TDD: mocked client only, no live server. The DB fixture in conftest routes
run lookups to a tmp database; report output defaults under the (patched)
configured results dir so tests never touch the real results/.
"""

import pytest
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


def _mock_client(thinking="mock reasoning trace", fail=False):
    c = MagicMock()
    ok = MagicMock()
    ok.success = True
    ok.identifier = "mock-id"
    ok.loaded_config = None
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
    assert mock_client_with_best.load_model.call_count == 5


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
    # Aborted after 3 consecutive errors, not all 5 tasks.
    assert client.load_model.call_count == 3
    assert "deep" in out["report_path"]


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
