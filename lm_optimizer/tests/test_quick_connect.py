"""Load-free connects: status/models must never load a model (§check)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from lm_optimizer.services.lm_studio import LMStudioClient


def _client():
    c = LMStudioClient.__new__(LMStudioClient)
    LMStudioClient.__init__(c, base_url="http://127.0.0.1:1234")
    return c


class TestQuickConnect:
    def test_echo_probe_loads_smallest(self):
        c = _client()
        c._client = MagicMock()
        m = MagicMock()
        m.id = "tiny-model"
        m.size_bytes = 100
        m.context_limit = 4096
        with patch.object(LMStudioClient, "_detect_capabilities", new=AsyncMock()), \
             patch.object(LMStudioClient, "list_models",
                          new=AsyncMock(return_value=[m])), \
             patch.object(LMStudioClient, "load_model",
                          new=AsyncMock(return_value=MagicMock(
                              success=True, identifier="i",
                              loaded_config=MagicMock(to_dict=lambda: {})))) as lm, \
             patch.object(LMStudioClient, "unload_model", new=AsyncMock()):
            async def _probe():
                await c._probe_echo_phase()
            asyncio.run(_probe())
            assert lm.await_count == 1

    def test_status_and_models_use_quick_connect(self):
        import inspect

        from lm_optimizer.cli import main as _cli

        src = inspect.getsource(_cli.status)
        assert "echo_probe=not quick" in src
        src = inspect.getsource(_cli.models)
        assert "echo_probe=False" in src

    def test_lms_output_decoding_never_crashes(self, monkeypatch):
        import subprocess as _sp

        from lm_optimizer.services import lms_cli

        seen = {}

        def fake_run(*args, **kwargs):
            seen.update(kwargs)
            out = MagicMock()
            out.returncode = 0
            out.stdout = "ok"
            out.stderr = ""
            return out

        monkeypatch.setattr(_sp, "run", fake_run)
        code, _ = lms_cli.run_lms("load", "--help")
        assert code == 0
        assert seen.get("encoding") == "utf-8"
        assert seen.get("errors") == "replace"

    def test_rejected_reasoning_value_remembered(self):
        import httpx

        c = LMStudioClient.__new__(LMStudioClient)
        LMStudioClient.__init__(c, base_url="http://127.0.0.1:1234")
        bodies = []

        def _resp(payload, status=200):
            req = httpx.Request("POST", "http://127.0.0.1:1234/api/v1/chat")
            return httpx.Response(status, json=payload, request=req)

        async def fake_request(method, path, json_data=None, **kwargs):
            bodies.append(json_data)
            if json_data.get("reasoning") == "off":
                raise httpx.HTTPStatusError(
                    "bad", request=httpx.Request("POST", "x"),
                    response=_resp({"error": {"code": "invalid_value",
                                              "message": "reasoning bad value"}} ,
                                   status=400))
            return _resp({"output": [{"type": "message", "content": "hi"}],
                          "stats": {"input_tokens": 1, "total_output_tokens": 2}})

        c._request_with_retry = fake_request
        out1 = asyncio.run(c.chat_completion(model="m", input_text="hi",
                                             temperature=0.3, max_output_tokens=5,
                                             reasoning="off"))
        assert out1["choices"][0]["message"]["content"] == "hi"
        n_first = len(bodies)
        out2 = asyncio.run(c.chat_completion(model="m", input_text="hi",
                                             temperature=0.3, max_output_tokens=5,
                                             reasoning="off"))
        assert out2["choices"][0]["message"]["content"] == "hi"
        # Second call skips the doomed attempt: exactly one request.
        assert len(bodies) == n_first + 1
        assert "reasoning" not in bodies[-1]

    def test_full_ids_lists_everything(self, monkeypatch, capsys):
        from typer.testing import CliRunner

        from lm_optimizer.cli import main as _cli

        m1 = MagicMock()
        m1.id = "ternary-bonsai-2-27b-pq2-0"
        m2 = MagicMock()
        m2.id = "openai/gpt-oss-20b"
        fake = MagicMock()
        fake.connect = AsyncMock()
        fake.list_models = AsyncMock(return_value=[m1, m2])
        fake.close = AsyncMock()
        monkeypatch.setattr(_cli, "get_client", lambda **k: fake)
        result = CliRunner().invoke(_cli.app, ["models", "--full-ids"])
        assert result.exit_code == 0, result.output
        assert "1|ternary-bonsai-2-27b-pq2-0" in result.output
        assert "2|openai/gpt-oss-20b" in result.output
        fake.connect.assert_awaited_once_with(echo_probe=False)
