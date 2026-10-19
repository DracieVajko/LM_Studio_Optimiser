"""Tests for ModelJudge backends."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from lm_optimizer.services.model_judge import (
    HeuristicJudge,
    LocalJudge,
    APIJudge,
    JudgeConfig,
    JudgeBackend,
    JudgeVerdict,
    ModelComparison,
)
from lm_optimizer.domain.models import ConfigurationResult, ConfigurationStatus, QualityScore, LoadConfiguration


def _make_quality_score(overall=0.95):
    """Create a QualityScore with all required fields."""
    return QualityScore(
        overall=overall,
        checks_passed=28,
        checks_total=30,
        task_completion=overall,
        factual_consistency=overall,
        format_compliance=overall,
        coding_correctness=overall,
        no_truncation=overall,
        no_malformed=overall,
    )


def _make_comparison(configs=None):
    """Create a minimal ModelComparison for testing."""
    if configs is None:
        configs = []
    return ModelComparison(
        run=None,
        configs=configs,
        use_case="coding",
        hardware_constraints={"vram_gb": 6, "ram_gb": 32},
        priority="balanced",
    )


def _make_config(config_id="cfg-1", model_id="model-1", tok_s=10.0, quality=0.95, ctx=4096, vram=5.0, status="passed"):
    """Create a minimal ConfigurationResult-like object."""
    cfg = MagicMock()
    cfg.id = config_id
    cfg.model_id = model_id
    cfg.context_length = ctx
    cfg.status = status
    cfg.peak_vram_gb = vram
    cfg.peak_ram_gb = 16.0
    cfg.stability_score = 0.9
    cfg.get_avg_generation_tok_s = MagicMock(return_value=tok_s)
    cfg.quality_score = _make_quality_score(quality)
    cfg.config = LoadConfiguration(
        context_length=ctx,
        gpu_ratio=0.8,
        eval_batch_size=256,
        flash_attention=True,
        offload_kv_cache_to_gpu=True,
    )
    return cfg


class MockAsyncClient:
    """Mock httpx.AsyncClient for testing."""
    def __init__(self):
        self.post = AsyncMock()
        self.aclose = AsyncMock()


@pytest.mark.asyncio
async def test_heuristic_judge_returns_valid_verdict():
    """Test HeuristicJudge returns a valid verdict with expected structure."""
    judge = HeuristicJudge()
    configs = [
        _make_config("cfg-1", "model-a", tok_s=10.0, quality=0.90),
        _make_config("cfg-2", "model-a", tok_s=15.0, quality=0.85),
        _make_config("cfg-3", "model-b", tok_s=8.0, quality=0.95),
    ]
    comparison = _make_comparison(configs)

    verdict = await judge.evaluate(comparison)

    assert isinstance(verdict, JudgeVerdict)
    assert verdict.recommended_config_id in ["cfg-1", "cfg-2", "cfg-3"]
    assert isinstance(verdict.reasoning, str) and len(verdict.reasoning) > 0
    assert 0.0 <= verdict.confidence <= 1.0
    assert isinstance(verdict.pros, list)
    assert isinstance(verdict.cons, list)
    assert isinstance(verdict.alternative_config_ids, list)
    assert isinstance(verdict.warnings, list)
    await judge.close()


@pytest.mark.asyncio
async def test_heuristic_judge_speed_priority():
    """Test HeuristicJudge with speed priority picks fastest."""
    judge = HeuristicJudge()
    configs = [
        _make_config("slow-high-q", "model-a", tok_s=5.0, quality=0.99),
        _make_config("fast-low-q", "model-a", tok_s=20.0, quality=0.80),
    ]
    comparison = _make_comparison(configs)
    comparison.priority = "speed"

    verdict = await judge.evaluate(comparison)
    assert verdict.recommended_config_id == "fast-low-q"
    await judge.close()


@pytest.mark.asyncio
async def test_heuristic_judge_quality_priority():
    """Test HeuristicJudge with quality priority picks highest quality."""
    judge = HeuristicJudge()
    configs = [
        _make_config("slow-high-q", "model-a", tok_s=5.0, quality=0.99),
        _make_config("fast-low-q", "model-a", tok_s=20.0, quality=0.80),
    ]
    comparison = _make_comparison(configs)
    comparison.priority = "quality"

    verdict = await judge.evaluate(comparison)
    assert verdict.recommended_config_id == "slow-high-q"
    await judge.close()


@pytest.mark.asyncio
async def test_heuristic_judge_empty_configs():
    """Test HeuristicJudge with no configs returns appropriate verdict."""
    judge = HeuristicJudge()
    comparison = _make_comparison([])

    verdict = await judge.evaluate(comparison)

    assert verdict.recommended_config_id == ""
    assert verdict.confidence == 0.0
    assert "No configurations to evaluate" in verdict.reasoning
    await judge.close()


@pytest.mark.asyncio
async def test_heuristic_judge_no_passed_configs():
    """Test HeuristicJudge with only failed configs."""
    judge = HeuristicJudge()
    configs = [
        _make_config("cfg-1", "model-a", tok_s=10.0, quality=0.90, status="quality_failed"),
    ]
    comparison = _make_comparison(configs)

    verdict = await judge.evaluate(comparison)

    assert verdict.recommended_config_id == ""
    assert verdict.confidence == 0.0
    await judge.close()


@pytest.mark.asyncio
async def test_local_judge_calls_lm_studio():
    """Test LocalJudge calls LM Studio chat completions endpoint."""
    verdict_json = json.dumps({
        "recommended_config_id": "cfg-1",
        "reasoning": "Best balance of speed and quality",
        "confidence": 0.85,
        "pros": ["Good speed", "High quality"],
        "cons": ["Uses 5GB VRAM"],
        "alternative_config_ids": ["cfg-2"],
        "warnings": ["Monitor VRAM"],
    })

    # Mock the httpx client
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = MockAsyncClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": verdict_json}}]}
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        judge = LocalJudge(JudgeConfig(backend=JudgeBackend.LOCAL, model="test-model"))
        configs = [_make_config("cfg-1", "model-a", tok_s=10.0, quality=0.90)]
        comparison = _make_comparison(configs)

        verdict = await judge.evaluate(comparison)

        assert verdict.recommended_config_id == "cfg-1"
        assert verdict.confidence == 0.85
        # Verify request was made
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/chat/completions"
        body = call_args[1]["json"]
        assert body.get("response_format") == {"type": "json_object"}
        await judge.close()


@pytest.mark.asyncio
async def test_local_judge_retry_on_json_error():
    """Test LocalJudge retries once on JSON parse error."""
    verdict_json = json.dumps({
        "recommended_config_id": "cfg-1",
        "reasoning": "Retry worked",
        "confidence": 0.75,
        "pros": [], "cons": [], "alternative_config_ids": [], "warnings": [],
    })

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = MockAsyncClient()
        # First response: invalid JSON
        mock_response1 = MagicMock()
        mock_response1.json.return_value = {"choices": [{"message": {"content": "not valid json"}}]}
        mock_response1.raise_for_status = MagicMock()
        # Second response: valid JSON
        mock_response2 = MagicMock()
        mock_response2.json.return_value = {"choices": [{"message": {"content": verdict_json}}]}
        mock_response2.raise_for_status = MagicMock()
        mock_client.post.side_effect = [mock_response1, mock_response2]
        mock_client_class.return_value = mock_client

        judge = LocalJudge(JudgeConfig(backend=JudgeBackend.LOCAL, model="test-model"))
        configs = [_make_config("cfg-1", "model-a", tok_s=10.0, quality=0.90)]
        comparison = _make_comparison(configs)

        verdict = await judge.evaluate(comparison)

        assert verdict.recommended_config_id == "cfg-1"
        assert mock_client.post.call_count == 2  # Two calls due to retry
        await judge.close()


@pytest.mark.asyncio
async def test_api_judge_calls_openrouter():
    """Test APIJudge calls OpenRouter-compatible endpoint with auth."""
    verdict_json = json.dumps({
        "recommended_config_id": "cfg-1",
        "reasoning": "API judge verdict",
        "confidence": 0.9,
        "pros": [], "cons": [], "alternative_config_ids": [], "warnings": [],
    })

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = MockAsyncClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": verdict_json}}]}
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        judge = APIJudge(JudgeConfig(
            backend=JudgeBackend.API,
            model="anthropic/claude-3.5-sonnet",
            api_key="test-key",
            api_base="https://openrouter.ai/api/v1",
        ))
        configs = [_make_config("cfg-1", "model-a", tok_s=10.0, quality=0.90)]
        comparison = _make_comparison(configs)

        verdict = await judge.evaluate(comparison)

        assert verdict.recommended_config_id == "cfg-1"
        assert mock_client.post.call_count == 1
        call_args = mock_client.post.call_args
        # Check auth header was passed in client construction
        mock_client_class.assert_called_once()
        init_kwargs = mock_client_class.call_args[1]
        assert init_kwargs.get("headers", {}).get("Authorization") == "Bearer test-key"
        await judge.close()


@pytest.mark.asyncio
async def test_local_judge_handles_connection_error():
    """Test LocalJudge raises clear error when LM Studio unavailable."""
    # Create judge with mocked client
    judge = LocalJudge(JudgeConfig(backend=JudgeBackend.LOCAL, model="test-model"))
    judge.client = MockAsyncClient()
    judge.client.post.side_effect = httpx.ConnectError("Connection refused")

    configs = [_make_config("cfg-1", "model-a", tok_s=10.0, quality=0.90)]
    comparison = _make_comparison(configs)

    with pytest.raises(RuntimeError) as exc_info:
        await judge.evaluate(comparison)

    assert "LM Studio" in str(exc_info.value) or "connection" in str(exc_info.value).lower()
    await judge.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])