"""Model-vs-model sandbox duels: text / html / scene (TDD RED batch)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


def _client(text_a="Answer A.", text_b="Answer B.", fail_b=False, tok_a=60.0, tok_b=30.0):
    state = {"side": "A"}

    async def _load(model_id, cfg):
        state["side"] = "B" if model_id == "model-b" else "A"
        if model_id == "model-b" and fail_b:
            bad = MagicMock()
            bad.success = False
            bad.error = "engine abort"
            return bad
        ok = MagicMock()
        ok.success = True
        ok.identifier = "x"
        return ok

    async def _chat(**kwargs):
        a_side = state["side"] == "A"
        return {
            "choices": [{"message": {"content": text_a if a_side else text_b}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "_stats": {
                "tokens_per_second": tok_a if a_side else tok_b,
                "time_to_first_token_seconds": 0.05,
            },
        }

    c = MagicMock()
    c.load_model = AsyncMock(side_effect=_load)
    c.ensure_unloaded = AsyncMock(return_value=True)
    c.chat_completion = AsyncMock(side_effect=_chat)
    c.unload_all = AsyncMock(return_value={})
    c.get_loaded_instances = AsyncMock(return_value=[])
    return c


class TestSandboxUnit:
    def test_clean_strips_fences(self):
        from lm_optimizer.services.sandbox import clean_generated_html

        assert clean_generated_html("```html\n<b>x</b>\n```").strip() == "<b>x</b>"
        assert clean_generated_html("```\n<b>y</b>\n```").strip() == "<b>y</b>"
        assert "<b>z</b>" in clean_generated_html("<b>z</b>")

    def test_invalid_kind_rejected(self):
        from lm_optimizer.services.sandbox import run_duel

        with pytest.raises(ValueError, match="[Kk]ind"):
            asyncio.run(run_duel(_client(), "model-a", "model-b", "hi", kind="video"))

    def test_text_duel_runs_a_then_b(self):
        from lm_optimizer.services.sandbox import run_duel

        res = asyncio.run(run_duel(_client(), "model-a", "model-b", "Say hi.", kind="text"))
        assert res.side_a.ok and res.side_b.ok
        assert res.side_a.text == "Answer A."
        assert res.side_b.text == "Answer B."
        assert res.faster == "A"
        assert res.side_a.tok_s == 60.0
        # Unloaded between sides and at the end.
        assert res.model_a == "model-a" and res.model_b == "model-b"

    def test_b_load_fail_partial(self):
        from lm_optimizer.services.sandbox import run_duel

        res = asyncio.run(
            run_duel(_client(fail_b=True), "model-a", "model-b", "Say hi.", kind="text")
        )
        assert res.side_a.ok is True
        assert res.side_b.ok is False
        assert res.side_b.error

    def test_html_writes_isolated_files(self, tmp_path):
        from lm_optimizer.services.sandbox import run_duel

        c = _client(text_a="<h1>A page</h1>", text_b="<h1>B page</h1>")
        res = asyncio.run(
            run_duel(c, "model-a", "model-b", "Make a page.", kind="html", sandbox_root=tmp_path)
        )
        assert res.side_a.ok and res.side_b.ok
        pa = tmp_path / res.job_id / "A" / "index.html"
        pb = tmp_path / res.job_id / "B" / "index.html"
        assert pa.read_text(encoding="utf-8") == "<h1>A page</h1>"
        assert pb.read_text(encoding="utf-8") == "<h1>B page</h1>"
        assert res.side_a.file_path and res.side_b.file_path
        assert res.side_a.file_path != res.side_b.file_path

class TestDuelPersistence:
    def _duel(self, job_id="abc123def456", status="done"):
        return {
            "id": job_id,
            "kind": "text",
            "model_a": "model-a",
            "model_b": "model-b",
            "prompt": "Say hi.",
            "status": status,
            "faster": "A",
            "result_json": '{"faster": "A"}',
            "error": None,
        }

    def test_save_get_roundtrip(self):
        from lm_optimizer.database.repositories import duel_repo

        duel_repo.save(self._duel())
        got = duel_repo.get("abc123def456")
        assert got["kind"] == "text"
        assert got["model_a"] == "model-a"
        assert got["faster"] == "A"
        assert got["result_json"] == '{"faster": "A"}'

    def test_list_desc_and_delete(self):
        from lm_optimizer.database.repositories import duel_repo

        duel_repo.save(self._duel("job-1"))
        duel_repo.save(self._duel("job-2"))
        rows = duel_repo.list_all()
        assert [r["id"] for r in rows] == ["job-2", "job-1"]
        assert duel_repo.delete("job-1") is True
        assert duel_repo.get("job-1") is None
        assert duel_repo.delete("job-1") is False

    def test_runs_table_untouched_by_duels(self):
        from lm_optimizer.database import manager as _manager
        from lm_optimizer.database.repositories import duel_repo, run_repo

        duel_repo.save(self._duel("solo"))
        assert run_repo.list_all(10, 0) == []
        with _manager.db_manager.get_connection() as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert "duels" in tables and "runs" in tables


class TestDuelHistoryApi:
    def test_list_and_delete_duel(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        import lm_optimizer.api.routes as routes
        from lm_optimizer.api.main import app
        from lm_optimizer.database.repositories import duel_repo

        import lm_optimizer.services.sandbox as sandbox_mod

        monkeypatch.setattr(sandbox_mod, "DEFAULT_SANDBOX_ROOT", tmp_path / "sandbox")
        (tmp_path / "sandbox" / "del1" / "A").mkdir(parents=True)
        (tmp_path / "sandbox" / "del1" / "A" / "index.html").write_text("<h1>x</h1>")

        duel_repo.save({
            "id": "del1", "kind": "html", "model_a": "a", "model_b": "b",
            "prompt": "p", "status": "done", "faster": "A",
            "result_json": "{}", "error": None,
        })
        with TestClient(app, raise_server_exceptions=False) as c:
            rows = c.get("/api/duels").json()["duels"]
            assert any(r["id"] == "del1" for r in rows)
            r = c.delete("/api/duels/del1")
            assert r.status_code == 200, r.text[:200]
            assert not (tmp_path / "sandbox" / "del1").exists(), "files go with the record"
            assert c.delete("/api/duels/del1").status_code == 404

    def test_run_delete_guarded(self, monkeypatch):
        from types import SimpleNamespace

        from fastapi.testclient import TestClient

        import lm_optimizer.api.routes as routes
        from lm_optimizer.api.main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            assert c.delete("/api/runs/00000000-0000-0000-0000-000000000000").status_code == 404

            live_id = "11111111-1111-1111-1111-111111111111"
            live = SimpleNamespace(id=live_id, status=SimpleNamespace(value="running"))
            monkeypatch.setattr(routes.run_repo, "get", lambda *a, **k: live)
            monkeypatch.setattr(routes, "_is_stale_run", lambda *a, **k: False)
            monkeypatch.setattr(routes, "_current_run_id", live_id)
            r = c.delete(f"/api/runs/{live_id}")
            assert r.status_code == 409, r.status_code


class TestAbandonRun:
    def test_abandon_stale_running(self, monkeypatch):
        from types import SimpleNamespace

        from fastapi.testclient import TestClient

        import lm_optimizer.api.routes as routes
        from lm_optimizer.api.main import app

        saved: list = []
        run_id = "22222222-2222-2222-2222-222222222222"
        live = SimpleNamespace(id=run_id, status=SimpleNamespace(value="running"))
        monkeypatch.setattr(routes.run_repo, "get", lambda *a, **k: live)
        monkeypatch.setattr(routes, "_is_stale_run", lambda *a, **k: True)
        monkeypatch.setattr(routes.run_repo, "save", lambda run: saved.append(run.status.value) or run_id)
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post(f"/api/runs/{run_id}/abandon")
            assert r.status_code == 200, r.text[:200]
            assert saved == ["interrupted"]

    def test_abandon_live_refused(self, monkeypatch):
        from types import SimpleNamespace

        from fastapi.testclient import TestClient

        import lm_optimizer.api.routes as routes
        from lm_optimizer.api.main import app

        run_id = "33333333-3333-3333-3333-333333333333"
        live = SimpleNamespace(id=run_id, status=SimpleNamespace(value="running"))
        monkeypatch.setattr(routes.run_repo, "get", lambda *a, **k: live)
        monkeypatch.setattr(routes, "_is_stale_run", lambda *a, **k: False)
        monkeypatch.setattr(routes, "_current_run_id", run_id)
        with TestClient(app, raise_server_exceptions=False) as c:
            assert c.post(f"/api/runs/{run_id}/abandon").status_code == 409

    def test_abandon_terminal_refused(self, monkeypatch):
        from types import SimpleNamespace

        from fastapi.testclient import TestClient

        import lm_optimizer.api.routes as routes
        from lm_optimizer.api.main import app

        run_id = "44444444-4444-4444-4444-444444444444"
        done = SimpleNamespace(id=run_id, status=SimpleNamespace(value="success"))
        monkeypatch.setattr(routes.run_repo, "get", lambda *a, **k: done)
        with TestClient(app, raise_server_exceptions=False) as c:
            assert c.post(f"/api/runs/{run_id}/abandon").status_code == 409


class TestSandboxPresets:
    def test_preset_buttons_per_kind(self):
        from pathlib import Path

        js = (Path(__file__).parent.parent / "ui" / "static" / "js" / "sandbox.js").read_text()
        for needle in ("Hash-table explainer", "find_duplicates coding", "Coffee landing page",
                       "Todo app", "Skyblock demo", "Simple spinner", "sb-presets"):
            assert needle in js, f"missing preset: {needle}"


class TestStartOptimization:
    def test_returns_run_immediately(self, monkeypatch):
        """POST /api/optimize must return the prepared run, never 500."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        import lm_optimizer.api.routes as routes
        from lm_optimizer.api.main import app
        from lm_optimizer.domain.models import (
            ModelIdentity,
            OptimizationProfile,
            OptimizationRun,
            RunStatus,
        )
        from fastapi.testclient import TestClient

        run = OptimizationRun(
            model=ModelIdentity(id="m", name="m"), profile=OptimizationProfile.BALANCED,
            status=RunStatus.RUNNING,
        )

        class FakeClient:
            async def get_model(self, model_id):
                return ModelIdentity(id=model_id, name=model_id)

            async def close(self):
                pass

        async def fake_get_client(echo_probe=True):
            return FakeClient()

        calls: list = []

        class FakeOptimizer:
            # NOTE: no .state here on purpose: the route must use the
            # prepare_run() return value, never optimizer.state (None
            # until the background task runs -> the production 500).
            async def prepare_run(self, *args, **kwargs):
                calls.append(args)
                return run, None, {}
        monkeypatch.setattr(routes, "get_lm_client", fake_get_client)
        monkeypatch.setattr(routes, "AdaptiveOptimizer", lambda *a, **k: FakeOptimizer())
        monkeypatch.setattr(routes, "run_optimization_task", AsyncMock())
        old_opt, old_id = routes._current_optimizer, routes._current_run_id
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                r = c.post("/api/optimize", json={"model_id": "m"})
                assert r.status_code == 200, r.text[:300]
                assert r.json()["id"] == str(run.id)
                assert r.json()["status"] == "running"
        finally:
            routes._current_optimizer, routes._current_run_id = old_opt, old_id


class TestOptimizeDirtyHost:
    def test_dirty_host_rejected_409(self, monkeypatch):
        """A resident model must refuse new web optimizations (fail-closed)."""
        from unittest.mock import AsyncMock

        import lm_optimizer.api.routes as routes
        from lm_optimizer.api.main import app
        from lm_optimizer.domain.models import ModelIdentity
        from fastapi.testclient import TestClient

        class StubbornClient:
            async def get_model(self, model_id):
                return ModelIdentity(id=model_id, name=model_id)

            async def unload_all(self):
                return {}

            async def get_loaded_instances(self):
                return [{"instance_id": "stuck", "model": "other", "config": {}}]

            async def close(self):
                pass

        async def fake_get_client(echo_probe=True):
            return StubbornClient()

        monkeypatch.setattr(routes, "get_lm_client", fake_get_client)
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post("/api/optimize", json={"model_id": "m"})
            assert r.status_code == 409, r.text[:300]


class TestSandboxApi:
    def _paths(self):
        from lm_optimizer.api.main import app

        out = {}
        for r in app.routes:
            path = getattr(r, "path", None)
            if path and "sandbox" in path:
                out.setdefault(path, set()).update(getattr(r, "methods", set()) or set())
            if type(r).__name__ == "_IncludedRouter":
                prefix = getattr(getattr(r, "include_context", None), "prefix", "") or ""
                inner = getattr(r, "original_router", None)
                for sub in getattr(inner, "routes", []) or []:
                    p = getattr(sub, "path", None)
                    if p and "sandbox" in p:
                        out.setdefault(prefix + p, set()).update(
                            getattr(sub, "methods", set()) or set()
                        )
        return out

    def test_routes_exist(self):
        paths = self._paths()
        assert "/api/sandbox" in paths and "POST" in paths["/api/sandbox"]
        assert "/api/sandbox/{job_id}" in paths and "GET" in paths["/api/sandbox/{job_id}"]
        assert "/sandbox" in paths and "GET" in paths["/sandbox"]

    def test_file_serving_rejects_bad_side_and_unknown_job(self):
        from fastapi.testclient import TestClient

        from lm_optimizer.api.main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/sandbox/files/deadbeef1234/X/index.html")
            assert r.status_code in (400, 404), r.status_code
            r = c.get("/sandbox/files/deadbeef1234/A/index.html")
            assert r.status_code == 404, r.status_code

    def test_sandbox_page_renders(self):
        from fastapi.testclient import TestClient

        from lm_optimizer.api.main import app

        with TestClient(app) as c:
            r = c.get("/sandbox")
            assert r.status_code == 200
            assert "sandbox.js?v=" in r.text

    def test_start_validates_kind(self):
        from fastapi.testclient import TestClient

        from lm_optimizer.api.main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post(
                "/api/sandbox",
                json={"model_a": "a", "model_b": "b", "prompt": "hi", "kind": "video"},
            )
            assert r.status_code == 422, r.status_code

    def test_oversize_output_rejected(self, tmp_path):
        from lm_optimizer.services.sandbox import run_duel

        big = "<p>" + "x" * 3000 + "</p>"
        res = asyncio.run(
            run_duel(
                _client(text_a=big, text_b=big), "model-a", "model-b", "Make a page.",
                kind="html", sandbox_root=tmp_path, max_file_chars=100,
            )
        )
        assert res.side_a.ok is False
        assert "too large" in (res.side_a.error or "").lower()
