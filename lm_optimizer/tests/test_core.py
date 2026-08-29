"""Tests for LM Studio Optimizer."""

import pytest

from lm_optimizer.benchmark.suite import BENCHMARK_SUITE, create_benchmark_cases
from lm_optimizer.config import HardwareConfig, LMStudioConfig, OptimizationConfig, config
from lm_optimizer.discovery.inspector import ModelCapabilities
from lm_optimizer.hardware.detection import get_cpu_info, get_memory_info
from lm_optimizer.profiles.registry import ProfileRegistry


class TestConfig:
    """Test configuration loading."""

    def test_default_config(self):
        assert config.lm_studio.base_url == "http://127.0.0.1:1234"
        assert config.optimization.default_profile == "balanced"
        assert config.optimization.minimum_quality_score == 0.97

    def test_hardware_config_defaults(self):
        hw = HardwareConfig()
        assert hw.gpu_vram_gb is None
        assert hw.system_ram_gb is None

    def test_lm_studio_config_defaults(self):
        lm = LMStudioConfig()
        assert lm.timeout == 120.0
        assert lm.max_retries == 3

    def test_optimization_config_defaults(self):
        opt = OptimizationConfig()
        assert opt.benchmark_runs == 3
        assert opt.warmup_runs == 1


class TestHardwareDetection:
    """Test hardware detection."""

    def test_cpu_info(self):
        cpu = get_cpu_info()
        assert cpu.cores_physical >= 1
        assert cpu.cores_logical >= cpu.cores_physical
        assert cpu.name is not None

    def test_memory_info(self):
        mem = get_memory_info()
        assert mem.total_gb > 0
        assert mem.available_gb <= mem.total_gb


class TestBenchmarkSuite:
    """Test benchmark suite."""

    def test_suite_not_empty(self):
        assert len(BENCHMARK_SUITE) == 5

    def test_categories_present(self):
        categories = {p.category for p in BENCHMARK_SUITE}
        assert "instruction" in categories
        assert "reasoning" in categories
        assert "context" in categories
        assert "coding" in categories
        assert "format" in categories

    def test_create_cases(self):
        cases = create_benchmark_cases(4096)
        assert len(cases) == 5
        for case in cases:
            assert case.max_tokens > 0
            assert case.context_length == 4096


class TestProfiles:
    """Test optimization profiles."""

    def test_default_profiles_exist(self):
        registry = ProfileRegistry()
        profiles = registry.list_profiles()
        names = {p.name for p in profiles}
        assert "speed" in names
        assert "balanced" in names
        assert "quality" in names

    def test_profile_weights_sum(self):
        registry = ProfileRegistry()
        for profile in registry.list_profiles():
            total = sum(profile.weights.values())
            assert abs(total - 1.0) < 0.01, f"{profile.name} weights sum to {total}"

    def test_custom_profile(self):
        registry = ProfileRegistry()
        custom = registry.create_custom(
            "test", "Test profile", {"generation_speed": 0.5, "quality": 0.5}, minimum_quality=0.9
        )
        assert custom.name == "test"
        assert custom.description == "Test profile"
        assert custom.minimum_quality == 0.9


class TestModelCapabilities:
    """Test model capabilities."""

    def test_context_candidates(self):
        from unittest.mock import MagicMock

        from lm_optimizer.api.client import LMStudioClient
        from lm_optimizer.discovery.inspector import ModelDiscovery

        client = MagicMock(spec=LMStudioClient)
        client.capabilities = MagicMock()
        client.capabilities.supports_flash_attention = True
        client.capabilities.get_supported_load_params.return_value = [
            "context_length",
            "gpu_ratio",
            "flash_attention",
            "offload_kv_cache_to_gpu",
            "eval_batch_size",
        ]

        discovery = ModelDiscovery(client)
        caps = ModelCapabilities(
            model_id="test",
            max_context_length=16384,
            estimated_vram_gb=4.0,
        )
        candidates = discovery.generate_context_candidates(caps)
        assert 4096 in candidates
        assert 8192 in candidates
        assert 16384 in candidates
        assert all(c <= 16384 for c in candidates)

    def test_gpu_ratio_candidates(self):
        from unittest.mock import MagicMock, patch

        from lm_optimizer.api.client import LMStudioClient
        from lm_optimizer.config import config
        from lm_optimizer.discovery.inspector import ModelDiscovery

        client = MagicMock(spec=LMStudioClient)
        client.capabilities = MagicMock()
        client.capabilities.supports_flash_attention = True
        client.capabilities.get_supported_load_params.return_value = [
            "context_length",
            "gpu_ratio",
            "flash_attention",
            "offload_kv_cache_to_gpu",
            "eval_batch_size",
        ]

        discovery = ModelDiscovery(client)
        caps = ModelCapabilities(
            model_id="test",
            max_context_length=8192,
            estimated_vram_gb=4.0,
        )

        # Mock hardware config to have GPU VRAM
        with patch.object(config.hardware, "gpu_vram_gb", 6.0):
            candidates = discovery.generate_gpu_ratio_candidates(caps)

        assert all(0.0 <= c <= 1.0 for c in candidates)
        assert 1.0 in candidates

    def test_batch_size_candidates(self):
        from unittest.mock import MagicMock

        from lm_optimizer.api.client import LMStudioClient
        from lm_optimizer.discovery.inspector import ModelDiscovery

        client = MagicMock(spec=LMStudioClient)
        client.capabilities = MagicMock()
        client.capabilities.supports_flash_attention = True
        client.capabilities.get_supported_load_params.return_value = [
            "context_length",
            "gpu_ratio",
            "flash_attention",
            "offload_kv_cache_to_gpu",
            "eval_batch_size",
        ]

        discovery = ModelDiscovery(client)
        caps = ModelCapabilities(model_id="test")
        candidates = discovery.generate_batch_size_candidates(caps)
        assert 256 in candidates
        assert 512 in candidates
        assert all(c > 0 for c in candidates)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
