"""Ollama registry + search space tests (Task 3) — static table, no live calls."""

from __future__ import annotations

import inspect

import httpx
import pytest

from lm_optimizer.backends.ollama.registry import OLLAMA_OPTIONS, ollama_space
from lm_optimizer.domain.models import GPUInfo, HardwareInfo


def show_fixture(model: str = "qwen3-8b") -> dict:
    """Raw /api/show details shape (as returned by OllamaClient.show())."""
    assert model == "qwen3-8b"
    return {
        "modelfile": 'FROM qwen3:8b\nPARAMETER num_ctx 8192\n',
        "details": {"parameter_size": "8.2B", "quantization_level": "Q4_K_M"},
        "model_info": {
            "general.architecture": "qwen3",
            "general.parameter_count": 8_200_000_000,
            "llama.context_length": 131072,
        },
    }


def hw_8thread(gpus: int = 1) -> HardwareInfo:
    return HardwareInfo(
        os="T",
        cpu_name="C",
        cpu_cores_physical=4,
        cpu_cores_logical=8,
        total_ram_gb=32,
        gpu_count=gpus,
        gpus=[GPUInfo(index=i, name="G", vram_gb=24, vendor="NVIDIA") for i in range(gpus)],
    )


def test_ollama_space_defaults():
    space = ollama_space(show_fixture("qwen3-8b"), hw_8thread())
    assert space["num_ctx"] == [2048, 4096, 8192]


def test_ollama_options_exact_list():
    assert OLLAMA_OPTIONS == [
        "num_ctx",
        "num_batch",
        "num_thread",
        "num_gpu",
        "use_mmap",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "repeat_penalty",
    ]


def test_ollama_space_batch_thread_gpu_defaults():
    space = ollama_space(show_fixture("qwen3-8b"), hw_8thread())
    assert space["num_batch"] == [64, 256, 512]
    assert space["num_thread"] == [8]
    assert space["num_gpu"] == [0, 1]


def test_ollama_space_num_ctx_capped_by_show():
    show = show_fixture("qwen3-8b")
    show["model_info"]["llama.context_length"] = 4096
    assert ollama_space(show, hw_8thread())["num_ctx"] == [2048, 4096]


def test_ollama_space_no_gpu_single_zero():
    space = ollama_space(show_fixture("qwen3-8b"), hw_8thread(gpus=0))
    assert space["num_gpu"] == [0]


def test_ollama_space_sampling_keys_not_swept():
    space = ollama_space(show_fixture("qwen3-8b"), hw_8thread())
    for key in ("temperature", "top_k", "top_p", "min_p", "repeat_penalty"):
        assert key not in space  # owned by existing generation profiles


def test_ollama_capabilities_is_property():
    from lm_optimizer.backends.ollama.client import OllamaClient

    assert isinstance(inspect.getattr_static(OllamaClient(), "capabilities"), property)


def test_ollama_show_returns_raw_details():
    payload = show_fixture("qwen3-8b")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/show":
            return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"error": "not found"})

    async def run() -> None:
        from lm_optimizer.backends.ollama.client import OllamaClient

        client = OllamaClient(transport=httpx.MockTransport(handler))
        try:
            assert await client.show("qwen3:8b") == payload
        finally:
            await client.close()

    import asyncio

    asyncio.run(run())


def test_search_space_ollama_branch_matches_registry():
    from unittest.mock import MagicMock

    from lm_optimizer.services.search_space import SearchSpaceGenerator

    show = show_fixture("qwen3-8b")
    hw = hw_8thread()
    stub = MagicMock()
    stub.backend_name = "ollama"
    assert SearchSpaceGenerator(stub).generate_ollama(show, hw) == ollama_space(show, hw)


def test_ollama_space_missing_context_length_uses_defaults():
    space = ollama_space({"model_info": {}}, hw_8thread())
    assert space["num_ctx"] == [2048, 4096, 8192]
