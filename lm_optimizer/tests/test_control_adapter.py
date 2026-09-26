"""ParameterControlAdapter: channels, GPU offload, estimate, verification (§8-§20)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from lm_optimizer.services.control_adapter import (
    ControlChannel,
    channel_of,
    channel_table,
    is_programmatic,
    load_with_channel,
    prune_by_estimate,
    refine_gpu_boundary,
    unused_rest_keys,
    verify_applied,
)


class TestChannels:
    def test_rest_channel(self):
        assert channel_of("context_length") == ControlChannel.REST
        assert channel_of("flash_attention") == ControlChannel.REST
        assert channel_of("parallel") == ControlChannel.REST

    def test_cli_channel(self):
        assert channel_of("gpu_ratio") == ControlChannel.CLI
        assert channel_of("ttl") == ControlChannel.CLI

    def test_manual_only_gui_presets(self):
        assert channel_of("n_threads") == ControlChannel.MANUAL_ONLY
        assert channel_of("llama_k_cache_quantization_type") == ControlChannel.MANUAL_ONLY
        assert channel_of("llama_v_cache_quantization_type") == ControlChannel.MANUAL_ONLY
        assert channel_of("try_mmap") == ControlChannel.MANUAL_ONLY
        assert channel_of("keep_model_in_memory") == ControlChannel.MANUAL_ONLY

    def test_cpu_threads_manual_only(self):
        # Thread controls are GUI presets in practice: no verified REST/CLI
        # path, so MANUAL_ONLY (never injected). cpu_threads stays out of
        # programmatic control until apply->verify->benchmark proves otherwise.
        assert channel_of("cpu_thread_pool_size") == ControlChannel.MANUAL_ONLY
        assert channel_of("cpu_threads") == ControlChannel.MANUAL_ONLY
        assert is_programmatic("cpu_threads") is False

    def test_sdk_channel(self):
        # gpu.offload_ratio also has a CLI flag, so CLI wins by precedence;
        # pure schema-known surfaces (no CLI) are SDK.
        assert channel_of("gpu.offload_ratio") == ControlChannel.CLI
        assert channel_of("gpu.main") == ControlChannel.SDK
        assert channel_of("gpu.disabled") == ControlChannel.SDK

    def test_unsupported(self):
        assert channel_of("no_such_param_xyz") == ControlChannel.UNSUPPORTED
        assert channel_of("llama.device") == ControlChannel.UNSUPPORTED

    def test_llama_override_documented_not_controlled(self):
        from lm_optimizer.services.parameter_registry import get

        spec = get("llama_cpp_arguments_override")
        assert spec is not None
        assert channel_of("llama_cpp_arguments_override") == ControlChannel.UNSUPPORTED
        assert "GUI" in spec.default_behavior

    def test_programmatic_never_from_existence(self):
        assert is_programmatic("context_length") is True
        assert is_programmatic("gpu_ratio") is True  # CLI is programmatic
        assert is_programmatic("n_threads") is False
        assert is_programmatic("llama.device") is False

    def test_moe_and_speculative_classification(self):
        from lm_optimizer.services.parameter_registry import get

        assert get("num_experts").optimizer_enabled is True
        assert get("num_cpu_expert_layers_ratio").experimental is True
        # SDK schema-known is a programmatic surface (unverified, not blind).
        assert is_programmatic("num_cpu_expert_layers_ratio") is True
        assert get("num_cpu_expert_layers_ratio").optimizer_enabled is False
        assert channel_of("speculative_draft_model") == ControlChannel.CLI
        assert get("speculative_draft_max_tokens").optimizer_enabled is False

    def test_batch_params_distinct(self):
        from lm_optimizer.services.parameter_registry import get

        names = {
            "eval_batch_size",
            "physical_batch_size",
            "parallel",
            "unified_kv_cache",
            "context_checkpoints",
        }
        assert {n for n in names if get(n) is not None} == names

    def test_version_dependent_notes(self):
        from lm_optimizer.services.parameter_registry import get

        for name in ("physical_batch_size", "parallel", "context_checkpoints"):
            assert "version-dependent" in get(name).default_behavior


class TestVerification:
    def test_match(self):
        out = verify_applied({"a": 1}, {"a": 1, "b": 2})
        assert out["verification"] == "MATCH"

    def test_partial(self):
        out = verify_applied({"a": 1, "b": 2}, {"a": 1, "b": 9})
        assert out["verification"] == "PARTIAL_MATCH"

    def test_mismatch(self):
        out = verify_applied({"a": 1}, {"a": 2})
        assert out["verification"] == "MISMATCH"

    def test_unknown(self):
        out = verify_applied({"a": 1}, None)
        assert out["verification"] == "UNKNOWN"
        assert out["actual"] is None

    def test_unused_rest_keys(self):
        assert "gpu_ratio" in unused_rest_keys({"gpu_ratio": 0.5, "context_length": 1})
        assert "context_length" not in unused_rest_keys({"context_length": 1})


class TestGpuBoundary:
    def test_adaptive_sequence(self):
        seen = []

        def probe(r):
            seen.append(r)
            return r <= 0.9  # 0.85 pass, 0.925 fail

        out = refine_gpu_boundary(probe, 0.7, 1.0, steps=3)
        assert [p["ratio"] for p in out] == [0.85, 0.925, 0.8875]
        assert [p["ok"] for p in out] == [True, False, True]
        assert seen == [0.85, 0.925, 0.8875]

    def test_probe_errors_are_failures(self):
        out = refine_gpu_boundary(lambda r: 1 / 0, 0.7, 1.0, steps=1)
        assert out[0] == {"ratio": 0.85, "ok": False}


class TestEstimatePrune:
    def test_unknown_prunes_nothing(self):
        kept, pruned = prune_by_estimate([4096, 8192], None, None)
        assert (kept, pruned) == ([4096, 8192], [])

    def test_prune_keeps_boundary_empirical(self):
        kept, pruned = prune_by_estimate([4096, 8192, 16384], 8192, "high")
        assert kept == [4096, 8192, 16384]  # first beyond limit kept for proof
        assert pruned == []

    def test_prune_beyond_boundary(self):
        kept, pruned = prune_by_estimate([4096, 8192, 16384, 32768], 8192, "high")
        assert kept == [4096, 8192, 16384]
        assert pruned == [32768]

    def test_low_confidence_prunes_nothing(self):
        kept, pruned = prune_by_estimate([4096, 8192], 4096, "low")
        assert (kept, pruned) == ([4096, 8192], [])


class TestEstimateFor:
    def test_parses_numbers(self, monkeypatch):
        from lm_optimizer.services import lms_cli

        monkeypatch.setattr(lms_cli, "lms_available", lambda: True)
        monkeypatch.setattr(
            lms_cli,
            "run_lms",
            lambda *a, **k: (
                0,
                "Estimated GPU Memory: 5.5 GiB\nEstimated Total Memory: 9.1 GiB\nConfidence: high",
            ),
        )
        out = lms_cli.estimate_for("m", context=8192, gpu=0.8)
        assert out["gpu_gib"] == 5.5 and out["total_gib"] == 9.1
        assert out["confidence"] == "high" and out["returncode"] == 0

    def test_unavailable_or_refused(self, monkeypatch):
        from lm_optimizer.services import lms_cli

        monkeypatch.setattr(lms_cli, "lms_available", lambda: False)
        out = lms_cli.estimate_for("m")
        assert out["gpu_gib"] is None and "not found" in out["raw"]
        monkeypatch.setattr(lms_cli, "lms_available", lambda: True)
        monkeypatch.setattr(lms_cli, "run_lms", lambda *a, **k: (1, "boom"))
        out = lms_cli.estimate_for("m")
        assert out["returncode"] == 1 and out["gpu_gib"] is None


class TestChannelLoad:
    def test_rest_default(self):
        from lm_optimizer.domain.models import LoadConfiguration

        client = MagicMock()
        res = MagicMock()
        res.success = True
        res.identifier = "iid"
        res.loaded_config = LoadConfiguration(context_length=2048)
        client.load_model = AsyncMock(return_value=res)
        out = asyncio.run(
            load_with_channel(client, "m", LoadConfiguration(context_length=2048, gpu_ratio=0.8))
        )
        assert out[1] == "REST" and out[0].success is True

    def test_cli_path_records_mode(self, monkeypatch):
        from lm_optimizer.domain.models import LoadConfiguration
        from lm_optimizer.services import control_adapter as ca

        monkeypatch.setattr(ca.lms_cli, "lms_available", lambda: True)
        monkeypatch.setattr(ca, "_run_cli_load", AsyncMock(return_value=("iid-1", "")))
        monkeypatch.setattr(ca, "_verify_cli_instance", AsyncMock(return_value=True))
        client = MagicMock()
        result, channel, applied = asyncio.run(
            load_with_channel(
                client, "m", LoadConfiguration(context_length=4096, gpu_ratio=0.8), gpu_via_cli=True
            )
        )
        assert channel == "CLI" and result.success is True
        assert applied["gpu_ratio"] == "0.8"
        client.load_model.assert_not_called()

    def test_cli_failure_falls_back(self, monkeypatch):
        from lm_optimizer.domain.models import LoadConfiguration
        from lm_optimizer.services import control_adapter as ca

        monkeypatch.setattr(ca.lms_cli, "lms_available", lambda: True)
        monkeypatch.setattr(ca, "_run_cli_load", AsyncMock(return_value=(None, "cli boom")))
        client = MagicMock()
        res = MagicMock()
        res.success = False
        res.error = "refused"
        res.loaded_config = None
        client.load_model = AsyncMock(return_value=res)
        result, channel, _applied = asyncio.run(
            load_with_channel(
                client, "m", LoadConfiguration(context_length=4096, gpu_ratio=0.5), gpu_via_cli=True
            )
        )
        assert channel == "REST" and result.success is False

    def test_channel_table(self):
        rows = channel_table(["context_length", "n_threads", "nope"])
        assert rows[0]["channel"] == "REST" and rows[0]["programmatic"] is True
        assert rows[1]["channel"] == "MANUAL_ONLY"
        assert rows[2]["channel"] == "UNSUPPORTED"
