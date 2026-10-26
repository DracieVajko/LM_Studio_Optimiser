"""Load-config export service: REST params, vLLM engine config, instructions."""

from __future__ import annotations

from unittest.mock import MagicMock

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
from lm_optimizer.services.config_export import build_load_config_export


def _run(*, best: ConfigurationResult | None = None, name: str = "M") -> OptimizationRun:
    run = OptimizationRun(
        model=ModelIdentity(id="vendor/model", name=name),
        profile=OptimizationProfile.BALANCED,
    )
    if best is not None:
        run.configurations = [best]
        run.best_config_id = best.id
    return run


def _best(**cfg) -> ConfigurationResult:
    base = dict(
        context_length=8192,
        flash_attention=True,
        offload_kv_cache_to_gpu=True,
        eval_batch_size=1024,
    )
    base.update(cfg)
    config = LoadConfiguration(**base)
    return ConfigurationResult(
        config=config,
        context_length=8192,
        status=ConfigurationStatus.PASSED,
        score=0.9,
        metrics=[
            BenchmarkMetrics(
                test_name="t", category="speed", success=True,
                generation_tok_s=10.0, completion_tokens=64,
            )
        ],
    )


def test_export_contains_rest_and_vllm_views():
    best = _best()
    out = build_load_config_export(_run(best=best))

    assert out["model_id"] == "vendor/model"
    assert out["config_id"] == str(best.id)
    assert out["config"]["context_length"] == 8192
    assert out["config"]["flash_attention"] is True
    assert out["lmstudio_load_body"]["model"] == "vendor/model"

    vllm = out["vllm_engine_config"]
    assert vllm["load_params"]["n_ctx"] == 8192
    assert vllm["load_params"]["n_batch"] == 1024
    assert vllm["load_params"]["flash_attn"] is True
    # KV on GPU => no_kv_offload false
    assert vllm["load_params"]["no_kv_offload"] is False

    assert out["instructions"]
    assert out["generated_at"]


def test_kv_on_cpu_maps_to_no_kv_offload_true():
    out = build_load_config_export(
        _run(best=_best(offload_kv_cache_to_gpu=False))
    )
    assert out["vllm_engine_config"]["load_params"]["no_kv_offload"] is True


def test_unknown_kv_placement_is_omitted_not_guessed():
    out = build_load_config_export(
        _run(best=_best(offload_kv_cache_to_gpu=None))
    )
    assert "no_kv_offload" not in out["vllm_engine_config"]["load_params"]


def test_no_validated_winner_raises():
    with pytest.raises(ValueError) as exc:
        build_load_config_export(_run(best=None))
    assert "No validated winner" in str(exc.value)


def test_export_does_not_invent_model_name():
    best = _best()
    out = build_load_config_export(_run(best=best, name=""))
    assert out["vllm_engine_config"]["name"].startswith("vendor/model")
