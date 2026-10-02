"""Stage-4 grid unification (Task 1)."""

from unittest.mock import MagicMock

from lm_optimizer.domain.models import HardwareInfo, GPUInfo, ModelIdentity, OptimizationProfile
from lm_optimizer.services.lm_studio import LMStudioCapabilities
from lm_optimizer.services.search_space import SearchSpaceGenerator


def _client(**cap_kw) -> MagicMock:
    cap = LMStudioCapabilities()
    cap.supports_context_length = True
    cap.supports_gpu_ratio = True
    cap.supports_flash_attention = True
    cap.supports_kv_cache_placement = True
    cap.supports_eval_batch_size = True
    cap.supports_physical_batch_size = True
    cap.supports_parallel = True
    cap.supports_context_checkpoints = True
    for k, v in cap_kw.items():
        setattr(cap, k, v)
    client = MagicMock()
    client.capabilities = cap
    return client


def _model() -> ModelIdentity:
    return ModelIdentity(id="m", name="m", context_limit=8192)


def _hw() -> HardwareInfo:
    return HardwareInfo(
        os="T",
        cpu_name="C",
        cpu_cores_physical=1,
        cpu_cores_logical=2,
        total_ram_gb=8,
        gpu_count=1,
        gpus=[GPUInfo(index=0, name="G", vram_gb=24, vendor="NVIDIA")],
    )


_PROFILE = OptimizationProfile.BALANCED


def test_eval_grid_includes_2048():
    space = SearchSpaceGenerator(_client()).generate(_model(), _hw(), _PROFILE)
    assert 2048 in space.batch_sizes


def test_parallel_8_throughput_only():
    c = _client()
    assert 8 in SearchSpaceGenerator(c).generate(
        _model(), _hw(), _PROFILE, {"workload_type": "throughput"}
    ).parallels
    assert SearchSpaceGenerator(c).generate(
        _model(), _hw(), _PROFILE, {"workload_type": "interactive"}
    ).parallels == [1]


def test_extended_values_opt_in_only():
    c = _client()
    space = SearchSpaceGenerator(c).generate(
        _model(), _hw(), _PROFILE, {"physical_batches": [128, 256, 512, 1024, 2048]}
    )
    assert 128 in space.physical_batch_sizes
