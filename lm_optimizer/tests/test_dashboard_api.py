"""Dashboard/API route truth: every documented path exists (§29)."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Plain TestClient (no lifespan side effects; requests work regardless)."""
    from lm_optimizer.api.main import app

    return TestClient(app)


def _seed_best_run(model_id):
    from lm_optimizer.database import repositories as _repo
    from lm_optimizer.domain.models import (
        BenchmarkMetrics,
        ConfigurationResult,
        ConfigurationStatus,
        HardwareInfo,
        LoadConfiguration,
        ModelIdentity,
        OptimizationProfile,
        OptimizationRun,
        QualityScore,
    )

    hw = HardwareInfo(os="T", cpu_name="C", cpu_cores_physical=1,
                      cpu_cores_logical=2, total_ram_gb=8, gpu_count=0)
    model = ModelIdentity(id=model_id, name=model_id, context_limit=131072)
    _repo.hardware_repo.save(hw)
    _repo.model_repo.save(model)
    run = OptimizationRun(model=model, hardware=hw,
                          profile=OptimizationProfile.BALANCED)
    _repo.run_repo.save(run)
    best = ConfigurationResult(
        config=LoadConfiguration(context_length=4096, flash_attention=True,
                                 offload_kv_cache_to_gpu=True,
                                 eval_batch_size=256),
        context_length=4096,
        status=ConfigurationStatus.PASSED,
        metrics=[BenchmarkMetrics(test_name="seed", category="instruction",
                                  success=True, generation_tok_s=30.0,
                                  prompt_tok_s=500.0, estimated_ttft_ms=100.0,
                                  prompt_tokens=10, completion_tokens=50,
                                  total_tokens=60, output_text="seeded best")],
        quality_score=QualityScore(overall=1.0, task_completion=1.0,
                                   factual_consistency=1.0,
                                   format_compliance=1.0,
                                   coding_correctness=1.0, no_truncation=1.0,
                                   no_malformed=1.0, checks_passed=6,
                                   checks_total=6),
        score=9.1,
    )
    best.run_id = run.id
    _repo.config_repo.save(best)
    run.configurations = [best]
    run.best_config_id = best.id
    _repo.run_repo.save(run)


def _deep_mock_client():
    from unittest.mock import AsyncMock, MagicMock

    c = MagicMock()
    ok = MagicMock()
    ok.success = True
    ok.identifier = "mock-id"
    ok.loaded_config = None
    ok.error = None
    c.load_model = AsyncMock(return_value=ok)
    c.ensure_unloaded = AsyncMock(return_value=True)
    c.unload_all = AsyncMock(return_value={})
    c.get_loaded_instances = AsyncMock(return_value=[])
    c.model_known_no_reasoning = lambda model_id: False

    async def _chat(**kwargs):
        return {
            "choices": [
                {"message": {"content": "Line 1\nLine 2 compass\nLine 3\nLine 4."}}
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20,
                      "total_tokens": 70},
            "_stats": {"tokens_per_second": 25.0,
                       "time_to_first_token_seconds": 0.05},
            "thinking_text": "mock reasoning trace",
        }

    c.chat_completion = AsyncMock(side_effect=_chat)
    return c


@pytest.fixture
def seeded_deep():
    """One real completed deep batch on disk; returns its batch stamp id."""
    from lm_optimizer.services.deep import run_deep_batch

    _seed_best_run("deep-ui-m")
    out = run_deep_batch(_deep_mock_client(), ["deep-ui-m"],
                         stamp="20261005-000000")
    assert out["models"]["deep-ui-m"]["status"] == "completed"
    return out["batch_stamp"]


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
        from unittest.mock import AsyncMock

        from lm_optimizer.api import routes as _routes
        from lm_optimizer.api.main import app

        async def _no_client(*args, **kwargs):
            fake = AsyncMock()
            fake.get_model.return_value = None
            fake.close.return_value = None
            return fake

        import lm_optimizer.api.routes as routes_mod
        _orig = routes_mod.get_lm_client
        routes_mod.get_lm_client = _no_client
        try:
            with TestClient(app) as client:
                resp = client.get("/api/models/does-not-exist",
                                  headers={"Accept": "application/json"})
                assert resp.status_code == 404
                assert "detail" in resp.json()
        finally:
            routes_mod.get_lm_client = _orig

    def test_model_route_unreachable_is_json_not_raise(self):
        """CI has no LM Studio: unreachable server must be JSON 503, never raise."""
        from fastapi.testclient import TestClient

        from lm_optimizer.api.main import app
        import lm_optimizer.api.routes as routes_mod

        async def _raise(*args, **kwargs):
            raise ConnectionError("LM Studio unreachable")

        _orig = routes_mod.get_lm_client
        routes_mod.get_lm_client = _raise
        try:
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.get("/api/models/does-not-exist",
                                  headers={"Accept": "application/json"})
                assert resp.status_code in (404, 503)
                assert "detail" in resp.json()
        finally:
            routes_mod.get_lm_client = _orig


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
                     "Optimize one model", "Optimize all models one by one",
                     "Check LM Studio", "List models", "Select 1-6"):
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


class TestAllConfigsWebTable:
    """Task 3: web results page lists every tried config (no backend change)."""

    def test_configurations_endpoint_lists_all_statuses(self, monkeypatch):
        """Existing endpoint already returns passed AND failed rows."""
        from types import SimpleNamespace
        from uuid import uuid4

        from fastapi.testclient import TestClient

        from lm_optimizer.api import routes as routes_mod
        from lm_optimizer.api.main import app
        from lm_optimizer.domain.models import (
            BenchmarkMetrics,
            ConfigurationResult,
            ConfigurationStatus,
            LoadConfiguration,
        )

        run_id = uuid4()
        ok = ConfigurationResult(
            run_id=run_id,
            config=LoadConfiguration(context_length=16384),
            context_length=16384,
            status=ConfigurationStatus.PASSED,
            metrics=[BenchmarkMetrics(test_name="t", category="i",
                                      success=True, generation_tok_s=30.0)],
            score=9.1,
        )
        bad = ConfigurationResult(
            run_id=run_id,
            config=LoadConfiguration(context_length=8192),
            context_length=8192,
            status=ConfigurationStatus.LOAD_FAILED,
            metrics=[],
            score=None,
            error="load refused",
        )
        monkeypatch.setattr(
            routes_mod.run_repo, "get",
            lambda *a, **k: SimpleNamespace(configurations=[ok, bad]))

        with TestClient(app) as client:
            resp = client.get(f"/api/runs/{run_id}/configurations")
            assert resp.status_code == 200
            data = resp.json()
            assert {c["status"] for c in data["configurations"]} >= {
                "passed", "load_failed"}

    def test_results_js_has_render_all_configs(self):
        """results.js renders the md-identical all-configs table (TDD anchor)."""
        from pathlib import Path

        text = Path(
            "lm_optimizer/ui/static/js/results.js").read_text(encoding="utf-8")
        assert "renderAllConfigs" in text
        assert "no configurations recorded" in text
        assert "showing " in text and " of " in text  # 200-row cap note
        assert "/configs/${" in text or "/configs/" in text  # JSON detail link


class TestDeepLeaderboard:
    """Task 4: deep leaderboard + preview UI and API (TDD: must FAIL before impl)."""

    def test_deep_runs_endpoint_lists_completed_batches(self, client, seeded_deep):
        data = client.get("/api/deep/runs").json()
        assert seeded_deep in [r["id"] for r in data["runs"]]

    def test_deep_run_detail_has_leaderboard_preview_and_export(self, client, seeded_deep):
        data = client.get(f"/api/deep/runs/{seeded_deep}").json()
        assert data["id"] == seeded_deep
        lb = data["leaderboard"]
        assert lb, "expected at least one leaderboard row"
        row = lb[0]
        for col in ("model", "gen_tok_s", "quality", "thinking_chars",
                    "elapsed_s", "verdict"):
            assert col in row, f"leaderboard row missing column: {col}"
        assert "export_url" in data and data["export_url"].endswith("/export")
        model_id = row["model"]
        assert model_id in data["models"]
        preview_tasks = data["models"][model_id]["preview"]
        assert preview_tasks, "expected per-task preview"
        for tab in ("prompt", "output", "thinking"):
            assert tab in preview_tasks[0], f"preview task missing tab: {tab}"

    def test_deep_leaderboard_tiebreak_is_deterministic(self):
        from lm_optimizer.api.routes import _sort_deep_leaderboard

        rows = [
            {"model": "b-model", "score": 1.0, "gen_tok_s": 10.0,
             "elapsed_s": 5.0, "quality": 1.0, "thinking_chars": 3,
             "status": "completed"},
            {"model": "a-model", "score": 1.0, "gen_tok_s": 10.0,
             "elapsed_s": 5.0, "quality": 1.0, "thinking_chars": 3,
             "status": "completed"},
            {"model": "fast-model", "score": 1.0, "gen_tok_s": 99.0,
             "elapsed_s": 50.0, "quality": 1.0, "thinking_chars": 3,
             "status": "completed"},
        ]
        ordered = [r["model"] for r in _sort_deep_leaderboard(rows)]
        assert ordered[0] == "fast-model"  # gen tok/s breaks the score tie
        assert ordered[1:] == ["a-model", "b-model"]  # model id breaks full tie

    def test_deep_preview_flags_outputs_over_50k_chars(self, client):
        from lm_optimizer.api import routes as routes_mod

        big = "x" * 60000
        entry = routes_mod._preview_entry("t", "p", "th", big)
        assert entry["paginated"] is True
        assert entry["chars"] == 60000
        assert "50000" in (entry["note"] or "")

    def test_deep_unknown_batch_is_json_404(self, client):
        resp = client.get("/api/deep/runs/no-such-batch")
        assert resp.status_code == 404
        assert "detail" in resp.json()

    def test_deep_page_renders_with_boot_reporter(self, client):
        html = client.get("/deep").text
        assert 'src="/static/js/deep.js?v=' in html
        assert "addEventListener('error'" in html
        assert "main-content" in html

    def test_deep_js_paginates_and_links_history(self):
        from pathlib import Path

        text = Path("lm_optimizer/ui/static/js/deep.js").read_text(encoding="utf-8")
        assert "50000" in text  # 50k-char preview pagination
        assert "Back to History" in text or 'href="/history"' in text
        assert "spinner" in text  # results-page loading pattern
        assert "document.readyState" in text  # readyState-safe boot
