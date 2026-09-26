"""Dashboard/API route truth: every documented path exists (§29)."""

from fastapi.testclient import TestClient


def _paths():
    """All paths incl. FastAPI>=0.141 deferred _IncludedRouter entries."""
    from lm_optimizer.api.main import app

    out = set()
    for r in app.routes:
        if hasattr(r, "path"):
            out.add(r.path)
        if type(r).__name__ == "_IncludedRouter":
            prefix = getattr(getattr(r, "include_context", None), "prefix", "") or ""
            inner = getattr(r, "original_router", None)
            for sub in getattr(inner, "routes", []) or []:
                if hasattr(sub, "path"):
                    out.add(prefix + sub.path)
    return out


class TestRoutesExist:
    def test_pages(self):
        paths = _paths()
        assert "/" in paths
        assert "/history" in paths
        assert "/results/{run_id}" in paths
        assert "/settings" in paths

    def test_api_routes(self):
        paths = _paths()
        for p in (
            "/api/status",
            "/api/models",
            "/api/runs",
            "/api/optimize",
            "/api/checkpoints",
            "/api/settings",
            "/api/presets",
        ):
            assert p in paths, f"missing {p}"

    def test_websocket_registered(self):
        assert "/ws/optimize/{run_id}" in _paths()


class TestDashboardRenders:
    def test_index_and_history_render_without_server(self):
        from lm_optimizer.api.main import app

        with TestClient(app) as client:
            for path in ("/", "/history", "/settings"):
                resp = client.get(path)
                assert resp.status_code == 200, path
                assert "LM Studio" in resp.text or "lm-status" in resp.text

    def test_results_page_renders(self):
        from lm_optimizer.api.main import app

        with TestClient(app) as client:
            resp = client.get("/results/some-run-id")
            assert resp.status_code == 200

    def test_static_version_consistent(self):
        import re
        from pathlib import Path

        base = Path("lm_optimizer/ui")
        versions = set()
        for html in (base / "templates").glob("*.html"):
            versions.update(
                re.findall(r"/static/js/[\w-]+\.js\?v=([\w.]+)",
                            html.read_text(encoding="utf-8")))
        for js in ("app", "history", "results", "settings"):
            text = (base / "static" / "js" / f"{js}.js").read_text(encoding="utf-8")
            versions.update(re.findall(r"from '\./[\w-]+\.js\?v=([\w.]+)", text))
        assert versions, "no versioned static references found"
        assert len(versions) == 1, f"static versions diverged: {versions}"

    def test_ready_state_safe_boot(self):
        from pathlib import Path

        base = Path("lm_optimizer/ui/static/js")
        for js in ("app", "history", "results", "settings"):
            text = (base / f"{js}.js").read_text(encoding="utf-8")
            assert "document.readyState" in text, f"{js}.js boot is DOMContentLoaded-only"

    def test_pages_have_boot_reporter_and_cache_buster(self):
        from lm_optimizer.api.main import app

        with TestClient(app) as client:
            for path, module in (("/", "app.js"), ("/history", "history.js"),
                                 ("/results/x", "results.js"),
                                 ("/settings", "settings.js")):
                html = client.get(path).text
                assert "main-content" in html, path
                # Boot reporter turns silent blank pages into visible errors.
                assert "addEventListener('error'" in html, path
                # Cache-busted module script (stale JS can never blank pages).
                assert f"/static/js/{module}?v=" in html, path

    def test_invalid_run_id_is_json_not_html(self):
        from lm_optimizer.api.main import app

        with TestClient(app) as client:
            resp = client.get("/api/runs/not-a-uuid",
                              headers={"Accept": "application/json"})
            assert resp.status_code in (404, 422)
            assert resp.headers["content-type"].startswith("application/json")

    def test_missing_run_is_json_404(self):
        from lm_optimizer.api.main import app

        with TestClient(app) as client:
            resp = client.get("/api/runs/00000000-0000-0000-0000-000000000000",
                              headers={"Accept": "application/json"})
            assert resp.status_code == 404
            assert "detail" in resp.json()

    def test_missing_configurations_is_json_404(self):
        from lm_optimizer.api.main import app

        with TestClient(app) as client:
            resp = client.get(
                "/api/runs/00000000-0000-0000-0000-000000000000/configurations",
                headers={"Accept": "application/json"})
            assert resp.status_code == 404
            assert "detail" in resp.json()

    def test_api_error_shape(self):
        from lm_optimizer.api.main import app

        with TestClient(app) as client:
            resp = client.get("/api/models/does-not-exist",
                              headers={"Accept": "application/json"})
            assert resp.status_code == 404
            assert "detail" in resp.json()


class TestLauncher:
    def test_start_optimizer_menu(self):
        from pathlib import Path

        bat = Path("start_optimizer.bat")
        assert bat.exists(), "primary Windows launcher missing"
        text = bat.read_text(encoding="utf-8", errors="replace")
        # Staged startup with visible progress (never silently closes).
        for item in ("[1/4]", "[2/4]", "[3/4]", "[4/4]", "startup.log",
                     "/api/status", "127.0.0.1:8080", ".venv",
                     "Press any key", "FAILED",
                     "Optimize one model", "Check LM Studio",
                     "List models", "Select 1-5"):
            assert item in text, f"launcher missing: {item}"
        low = text.lower()
        # Browser opens only after readiness; no bulk auto-runs.
        assert "opening browser" in low
        assert "call run-all" not in low and "start \"\" run-all" not in low
        assert "for %%m in (" not in low
        # Model picker: numbered full IDs, confirm before optimizing.
        assert "full-ids" in low
        assert "model number or full id" in low
        assert "status --quick" in low

    def test_optimization_uses_same_services(self):
        import inspect

        from lm_optimizer.api import routes as _routes
        from lm_optimizer.cli import main as _cli

        assert "AdaptiveOptimizer" in inspect.getsource(_routes.run_optimization_task)
        assert "BenchmarkService" in inspect.getsource(_cli._run_optimization)
