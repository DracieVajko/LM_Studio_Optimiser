"""OllamaClient tests (Task 2) — mocked httpx transport only, no live calls."""

import json

import httpx
import pytest

from lm_optimizer.backends.base import assert_conforms
from lm_optimizer.backends.ollama.client import OllamaClient

TAGS_PAYLOAD = {
    "models": [
        {
            "name": "qwen3:8b",
            "model": "qwen3:8b",
            "size": 5_000_000_000,
            "digest": "abc123",
            "details": {"parameter_size": "8.2B", "quantization_level": "Q4_K_M"},
        }
    ]
}

# eval_count/eval_duration*1e9 = 306/5e9*1e9 = 61.2 tok/s
GENERATE_PAYLOAD = {
    "model": "qwen3:8b",
    "response": "hi there",
    "done": True,
    "done_reason": "stop",
    "total_duration": 6_000_000_000,
    "load_duration": 500_000_000,
    "prompt_eval_count": 50,
    "prompt_eval_duration": 1_000_000_000,
    "eval_count": 306,
    "eval_duration": 5_000_000_000,
}


def _patch_async_client(monkeypatch, transport):
    """Route every httpx.AsyncClient construction through MockTransport."""
    orig = httpx.AsyncClient

    def factory(*args, **kwargs):
        if kwargs.get("transport") is None:
            kwargs["transport"] = transport
        return orig(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


@pytest.fixture
def mock_httpx_tags(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/tags":
            return httpx.Response(200, json=TAGS_PAYLOAD)
        if request.method == "GET" and request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.11.0"})
        return httpx.Response(404, json={"error": "not found"})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))


@pytest.fixture
def mock_httpx_generate(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/generate":
            body = json.loads(request.content or b"{}")
            assert body.get("model") == "qwen3:8b"
            assert body.get("stream") is False
            assert body.get("options", {}).get("num_ctx") == 4096
            return httpx.Response(200, json=GENERATE_PAYLOAD)
        return httpx.Response(404, json={"error": "not found"})

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))


async def test_ollama_conforms_and_lists_models(mock_httpx_tags):
    assert_conforms(OllamaClient())
    assert [m.id for m in await OllamaClient().list_models()] == ["qwen3:8b"]


async def test_ollama_metrics_normalized(mock_httpx_generate):
    m = await OllamaClient().generate("qwen3:8b", "hi", {"num_ctx": 4096})
    assert m.gen_tok_s == pytest.approx(61.2, abs=0.1)


async def test_ollama_metrics_full_normalization(mock_httpx_generate):
    m = await OllamaClient().generate("qwen3:8b", "hi", {"num_ctx": 4096})
    assert m.prompt_tok_s == pytest.approx(50.0, abs=0.1)
    assert m.load_ms == pytest.approx(500.0, abs=1.0)
    assert m.text == "hi there"


async def test_ollama_unload_verified_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True, "done_reason": "unload"})
        if request.method == "GET" and request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(404, json={"error": "not found"})

    client = OllamaClient(transport=httpx.MockTransport(handler))
    try:
        assert await client.unload("qwen3:8b") is True
    finally:
        await client.close()


async def test_ollama_connect_sets_capabilities(mock_httpx_tags):
    client = OllamaClient()
    try:
        await client.connect()
        assert client.capabilities.version == "0.11.0"
    finally:
        await client.close()
