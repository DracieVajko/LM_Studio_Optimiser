"""Context strategy gating and configuration building (spec §5, §10).

The strategies must never silently fall back to a different mode: a mode the
runtime cannot apply is reported unavailable. KV-on-CPU is a separate control
from "model on CPU" and must be built explicitly.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from lm_optimizer.domain.models import (
    LoadConfiguration,
    LMStudioCapabilities,
    ModelIdentity,
)
from lm_optimizer.services.optimizer import (
    AdaptiveOptimizer,
    VALID_CONTEXT_STRATEGIES,
)


def _opt(caps=None, gpu_via_cli: bool = False):
    opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    client = MagicMock()
    client._capabilities = caps
    opt.client = client
    opt.benchmark = MagicMock()
    opt.benchmark.gpu_via_cli = gpu_via_cli
    return opt


def _build(opt, base, strategy):
    return asyncio.run(
        opt._build_strategy_config(base, strategy, ModelIdentity(id="m", name="m"))
    )


def test_strategy_names_are_the_documented_ones():
    assert VALID_CONTEXT_STRATEGIES == {
        "winner",
        "gpu_resident",
        "model_first_kv_cpu",
        "system_spillover",
    }


def test_winner_strategy_is_unchanged():
    base = LoadConfiguration(
        context_length=8192,
        flash_attention=True,
        offload_kv_cache_to_gpu=True,
        eval_batch_size=2048,
    )
    cfg = _build(_opt(), base, "winner")
    assert cfg is not None
    assert cfg.to_dict() == base.to_dict()


def test_model_first_kv_cpu_keeps_weights_on_gpu_and_moves_kv():
    base = LoadConfiguration(
        context_length=8192,
        flash_attention=True,
        offload_kv_cache_to_gpu=True,
        eval_batch_size=2048,
    )
    cfg = _build(_opt(), base, "model_first_kv_cpu")
    assert cfg is not None
    assert cfg.offload_kv_cache_to_gpu is False
    # Model weights should not be spilled when the runtime allows it.
    assert cfg.gpu_ratio in (None, 1.0)


def test_gpu_resident_strategy_sets_kv_on_gpu():
    base = LoadConfiguration(
        context_length=8192,
        flash_attention=True,
        offload_kv_cache_to_gpu=False,
    )
    cfg = _build(_opt(), base, "gpu_resident")
    assert cfg is not None
    assert cfg.offload_kv_cache_to_gpu is True


def test_kv_placement_modes_unavailable_without_capability():
    caps = LMStudioCapabilities(supports_kv_cache_placement=False)
    base = LoadConfiguration(context_length=8192, flash_attention=True)
    assert _build(_opt(caps), base, "gpu_resident") is None
    assert _build(_opt(caps), base, "model_first_kv_cpu") is None


def test_system_spillover_requires_verified_cli_channel(monkeypatch):
    base = LoadConfiguration(context_length=8192, flash_attention=True)
    # CLI disabled in settings -> unavailable, never a silent REST fallback.
    assert _build(_opt(gpu_via_cli=False), base, "system_spillover") is None

    monkeypatch.setattr(
        "lm_optimizer.services.lms_cli.lms_available", lambda: False
    )
    assert _build(_opt(gpu_via_cli=True), base, "system_spillover") is None

    monkeypatch.setattr(
        "lm_optimizer.services.lms_cli.lms_available", lambda: True
    )
    cfg = _build(_opt(gpu_via_cli=True), base, "system_spillover")
    assert cfg is not None
    assert cfg.gpu_ratio == 0.5


def test_requested_vs_applied_is_recorded_by_the_client():
    # The strategy config is only a request; LM StudioClient.load_model
    # returns echo_mismatches which the run stores. This pins that the
    # client API exposes the verification channel the optimizer relies on.
    from lm_optimizer.services.lm_studio import LMStudioClient

    assert hasattr(LMStudioClient, "_compare_echo")
    assert hasattr(LMStudioClient, "_read_echo_config")
