"""Tests for the ollama endpoint control probe (mocked transport)."""

import httpx
import pytest

from lm_optimizer.backends.ollama.client import OllamaClient
from lm_optimizer.backends.ollama.probe import probe_model


def _handler_thinking_family(request: httpx.Request) -> httpx.Response:
    if request.method == "GET" and request.url.path == "/api/tags":
        return httpx.Response(200, json={"models": [
            {"name": "m:1b", "model": "m:1b", "size": 1, "details": {}}
        ]})
    if request.method == "GET" and request.url.path == "/api/version":
        return httpx.Response(200, json={"version": "0.40.2"})
    if request.method == "GET" and request.url.path == "/api/ps":
        return httpx.Response(200, json={"models": []})
    if request.method == "POST" and request.url.path == "/api/show":
        return httpx.Response(200, json={
            "parameters": "num_ctx 4096",
            "details": {"context_length": 8192, "family": "test",
                        "quantization_level": "Q4", "parameter_size": "1B"},
        })
    if request.method == "POST" and request.url.path == "/api/generate":
        import json as _json

        body = _json.loads(request.content or b"{}")
        if body.get("think"):
            return httpx.Response(200, json={
                "response": "OK", "thinking": "t", "eval_count": 4,
                "eval_duration": 10**9, "prompt_eval_count": 2,
                "prompt_eval_duration": 10**9, "load_duration": 10**6,
                "done_reason": "stop",
            })
        return httpx.Response(200, json={
            "response": "", "thinking": "t...", "eval_count": 8,
            "eval_duration": 10**9, "prompt_eval_count": 2,
            "prompt_eval_duration": 10**9, "load_duration": 10**6,
            "done_reason": "length",
        })
    if request.method == "POST" and request.url.path == "/api/generate-unload":
        return httpx.Response(200, json={"done": True, "done_reason": "unload"})
    return httpx.Response(404, json={"error": "not found"})


def _handler_plain(request: httpx.Request) -> httpx.Response:
    if request.method == "GET" and request.url.path == "/api/tags":
        return httpx.Response(200, json={"models": [
            {"name": "m:1b", "model": "m:1b", "size": 1, "details": {}}
        ]})
    if request.method == "GET" and request.url.path == "/api/version":
        return httpx.Response(200, json={"version": "0.40.2"})
    if request.method == "GET" and request.url.path == "/api/ps":
        return httpx.Response(200, json={"models": []})
    if request.method == "POST" and request.url.path == "/api/show":
        return httpx.Response(200, json={
            "parameters": "",
            "details": {"context_length": 8192, "family": "test",
                        "quantization_level": "Q4", "parameter_size": "1B"},
        })
    if request.method == "POST" and request.url.path == "/api/generate":
        return httpx.Response(200, json={
            "response": "OK", "eval_count": 2,
            "eval_duration": 10**9, "prompt_eval_count": 2,
            "prompt_eval_duration": 10**9, "load_duration": 10**6,
            "done_reason": "stop",
        })
    return httpx.Response(404, json={"error": "not found"})


async def test_probe_retries_think_once_for_empty_answer():
    client = OllamaClient(transport=httpx.MockTransport(_handler_thinking_family))
    try:
        report = await probe_model(client, "m:1b", ctx_pair=(2048, 4096))
    finally:
        await client.close()
    assert report["steps"]["connect"]["ok"] is True
    assert report["steps"]["list"]["present"] is True
    assert report["steps"]["show"]["context_length"] == 8192
    off = report["steps"]["generate_think_off"]
    assert off["ok"] is True and off["empty"] is True
    retry = report["steps"]["generate_think_on_retry"]
    assert retry.get("skipped") is None
    assert retry["text"] == "OK"


async def test_probe_skips_think_retry_when_answered():
    client = OllamaClient(transport=httpx.MockTransport(_handler_plain))
    try:
        report = await probe_model(client, "m:1b", ctx_pair=(2048, 4096))
    finally:
        await client.close()
    assert report["steps"]["generate_think_off"]["text"] == "OK"
    assert report["steps"]["generate_think_on_retry"] == {"ok": True, "skipped": True}
    assert "4096" in report["steps"]["ctx_timings"]


async def test_probe_reports_connect_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "down"})

    client = OllamaClient(transport=httpx.MockTransport(handler))
    try:
        report = await probe_model(client, "m:1b")
    finally:
        await client.close()
    assert report["steps"]["connect"]["ok"] is False
