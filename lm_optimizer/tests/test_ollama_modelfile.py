"""Modelfile export + apply tests (Task 4) — mocked httpx only, no live calls."""

from __future__ import annotations

import json

import httpx
import pytest

from lm_optimizer.backends.ollama.modelfile import (
    default_tag_for,
    parse_modelfile,
    render_modelfile,
)


def test_roundtrip_modelfile():
    text = render_modelfile("qwen3:8b", {"num_ctx": 8192, "temperature": 0.7})
    assert parse_modelfile(text) == {"from": "qwen3:8b", "num_ctx": 8192, "temperature": 0.7}


def test_bad_line_names_number():
    with pytest.raises(ValueError, match="line 3"):
        parse_modelfile("FROM qwen3:8b\nPARAMETER nope\nPARAMETER num_ctx abc")


def test_render_exact_lines():
    assert render_modelfile("qwen3:8b", {"num_ctx": 8192, "temperature": 0.7}) == (
        "FROM qwen3:8b\nPARAMETER num_ctx 8192\nPARAMETER temperature 0.7\n"
    )


def test_render_bool_lowercase_and_roundtrip():
    text = render_modelfile("qwen3:8b", {"use_mmap": True, "num_gpu": 1})
    assert "PARAMETER use_mmap true\n" in text
    assert parse_modelfile(text) == {"from": "qwen3:8b", "use_mmap": True, "num_gpu": 1}


def test_parse_missing_from_names_line_1():
    with pytest.raises(ValueError, match="line 1"):
        parse_modelfile("PARAMETER num_ctx 8192\n")


def test_parse_ignores_blanks_and_comments():
    text = "FROM qwen3:8b\n\n# tuned 2026-10-02\nPARAMETER num_ctx 8192\n"
    assert parse_modelfile(text) == {"from": "qwen3:8b", "num_ctx": 8192}


def test_default_tag_appends_best():
    assert default_tag_for("qwen3:8b") == "qwen3:8b-best"
    assert default_tag_for("qwen3") == "qwen3-best"


def test_parse_show_parameters():
    from lm_optimizer.backends.ollama.modelfile import parse_show_parameters

    show = {"parameters": "num_ctx 8192\ntemperature 0.7\n# comment\nbadline\n"}
    assert parse_show_parameters(show) == {"num_ctx": 8192, "temperature": 0.7}
    assert parse_show_parameters({}) == {}
    assert parse_show_parameters({"parameters": 123}) == {}


TAGS_EMPTY = {"models": []}
TAGS_ONE = {
    "models": [
        {"name": "qwen3:opt", "model": "qwen3:opt", "size": 5_000_000_000, "details": {}}
    ]
}


def _client_with(handler):
    from lm_optimizer.backends.ollama.client import OllamaClient

    return OllamaClient(transport=httpx.MockTransport(handler))


async def test_create_posts_from_and_parameters():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/create":
            seen.update(json.loads(request.content or b"{}"))
            return httpx.Response(200, json={"status": "success"})
        return httpx.Response(404, json={"error": "not found"})

    client = _client_with(handler)
    try:
        result = await client.create(
            "qwen3:opt", from_="qwen3:8b", parameters={"num_ctx": 8192}
        )
    finally:
        await client.close()
    assert seen["model"] == "qwen3:opt"
    assert seen["from"] == "qwen3:8b"
    assert seen["parameters"] == {"num_ctx": 8192}
    assert result.get("status") == "success"


async def test_apply_refuses_existing_tag():
    from lm_optimizer.backends.ollama.modelfile import apply_modelfile_text

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/tags":
            return httpx.Response(200, json=TAGS_ONE)
        return httpx.Response(404, json={"error": "not found"})

    client = _client_with(handler)
    try:
        with pytest.raises(ValueError, match="already exists"):
            await apply_modelfile_text(
                client, "FROM qwen3:8b\nPARAMETER num_ctx 8192\n", "qwen3:opt"
            )
    finally:
        await client.close()


async def test_apply_new_tag_posts_create():
    from lm_optimizer.backends.ollama.modelfile import apply_modelfile_text

    created: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/tags":
            return httpx.Response(200, json=TAGS_EMPTY)
        if request.method == "POST" and request.url.path == "/api/create":
            created.update(json.loads(request.content or b"{}"))
            return httpx.Response(200, json={"status": "success"})
        return httpx.Response(404, json={"error": "not found"})

    client = _client_with(handler)
    try:
        result = await apply_modelfile_text(
            client, "FROM qwen3:8b\nPARAMETER num_ctx 8192\n", "qwen3:opt"
        )
    finally:
        await client.close()
    assert created["model"] == "qwen3:opt"
    assert created["from"] == "qwen3:8b"
    assert result.get("status") == "success"


def test_cli_has_ollama_export_and_apply():
    import typer

    from lm_optimizer.cli.main import app

    info = typer.main.get_command(app)
    assert "ollama-export" in info.commands
    assert "ollama-apply" in info.commands
    apply_params = [p.name for p in info.commands["ollama-apply"].params]
    assert "file" in apply_params
    assert "yes" in apply_params


def test_cli_export_writes_file(tmp_path):
    from typer.testing import CliRunner

    from lm_optimizer.cli.main import app

    out = tmp_path / "qwen3-best.modelfile.md"
    result = CliRunner().invoke(
        app,
        [
            "ollama-export",
            "--model",
            "qwen3:8b",
            "--param",
            "num_ctx=8192",
            "--param",
            "temperature=0.7",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert parse_modelfile(out.read_text(encoding="utf-8")) == {
        "from": "qwen3:8b",
        "num_ctx": 8192,
        "temperature": 0.7,
    }
