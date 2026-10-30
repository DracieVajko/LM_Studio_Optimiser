"""Dashboard quick-stats + history winner metrics (reliability fixes).

Covers: embedding exclusion, inventory error vs empty, optimized-model count,
local-day "runs today", and server-computed best metrics on /api/runs.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

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


def _model(mid, mtype=None):
    m = MagicMock()
    m.id = mid
    m.model_type = mtype
    return m


def _fake_client(models):
    fake = MagicMock()
    fake.list_models = AsyncMock(return_value=models)
    fake.close = AsyncMock()
    return fake


def _patch_client(monkeypatch, models=None, raises=False):
    import lm_optimizer.api.routes as routes_mod

    if raises:
        async def _raise(*a, **k):
            raise ConnectionError("LM Studio unreachable")

        monkeypatch.setattr(routes_mod, "get_lm_client", _raise)
    else:
        async def _ok(*a, **k):
            return _fake_client(models or [])

        monkeypatch.setattr(routes_mod, "get_lm_client", _ok)


def _hw():
    return HardwareInfo(
        os="T", cpu_name="C", cpu_cores_physical=1,
        cpu_cores_logical=2, total_ram_gb=8, gpu_count=0,
    )


def _save_model(mid):
    from lm_optimizer.database.repositories import model_repo

    model_repo.save(ModelIdentity(id=mid, name=mid))


def _qs(overall=0.99):
    return QualityScore(
        overall=overall, task_completion=1.0, factual_consistency=1.0,
        format_compliance=1.0, coding_correctness=1.0, no_truncation=1.0,
        no_malformed=1.0, checks_passed=6, checks_total=6,
    )


def _metrics(gen=42.5):
    return [
        BenchmarkMetrics(
            test_name="t", category="instruction", success=True,
            generation_tok_s=gen, prompt_tok_s=500, estimated_ttft_ms=100,
            prompt_tokens=10, completion_tokens=50, total_tokens=60,
            output_text="measured",
        )
    ]


class TestStatsEndpoint:
    def test_embedding_models_excluded_from_available(self, monkeypatch):
        models = [
            _model("qwen3-8b", "llm"),
            _model("llama-3-8b", "llm"),
            _model("nomic-embed-text", "embeddings"),
        ]
        _patch_client(monkeypatch, models)
        from lm_optimizer.api.main import app

        with TestClient(app) as client:
            r = client.get("/api/stats")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["models_total"] == 3
        assert body["models_available"] == 2
        assert body["models_embedding_only"] == 1
        assert body["inventory_ok"] is True

    def test_inventory_error_is_not_silent_zero(self, monkeypatch):
        _patch_client(monkeypatch, raises=True)
        from lm_optimizer.api.main import app

        with TestClient(app) as client:
            r = client.get("/api/stats")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["inventory_ok"] is False
        assert body["inventory_error"]
        assert body["models_available"] == 0

    def test_optimized_and_total_counts(self, monkeypatch):
        from lm_optimizer.database.repositories import run_repo

        _patch_client(monkeypatch, [])
        # Two distinct models with a validated winner, one without.
        for mid, best in (("m-a", True), ("m-b", True), ("m-c", False)):
            _save_model(mid)
            run = OptimizationRun(
                model=ModelIdentity(id=mid, name=mid), hardware=_hw(),
                profile=OptimizationProfile.BALANCED,
            )
            if best:
                run.best_config_id = run.id  # any non-null id (no FK)
            run_repo.save(run)
        from lm_optimizer.api.main import app

        with TestClient(app) as client:
            r = client.get("/api/stats")
        body = r.json()
        assert body["total_runs"] == 3
        assert body["optimized_models"] == 2
        assert body["runs_today"] == 3  # all just created (local day)

    def test_runs_today_uses_local_day(self, monkeypatch):
        from lm_optimizer.database.repositories import run_repo

        _patch_client(monkeypatch, [])
        _save_model("m")
        run = OptimizationRun(
            model=ModelIdentity(id="m", name="m"), hardware=_hw(),
            profile=OptimizationProfile.BALANCED,
        )
        run.created_at = datetime(2001, 1, 1, 12, 0, 0)  # clearly not today
        run_repo.save(run)
        from lm_optimizer.api.main import app

        with TestClient(app) as client:
            r = client.get("/api/stats")
        body = r.json()
        assert body["total_runs"] == 1
        assert body["runs_today"] == 0


class TestHistoryWinnerMetrics:
    def test_list_run_reports_winner_metrics(self, monkeypatch):
        from lm_optimizer.database.repositories import config_repo, run_repo

        _patch_client(monkeypatch, [])
        _save_model("m")
        run = OptimizationRun(
            model=ModelIdentity(id="m", name="m"), hardware=_hw(),
            profile=OptimizationProfile.BALANCED,
        )
        run_repo.save(run)
        cfg = ConfigurationResult(
            run_id=run.id,
            config=LoadConfiguration(context_length=8192),
            context_length=8192,
            status=ConfigurationStatus.PASSED,
            metrics=_metrics(gen=33.3),
            quality_score=_qs(0.981),
            score=0.8,
            tested_at=datetime.now(),
        )
        config_repo.save(cfg)
        run.best_config_id = cfg.id
        run_repo.save(run)

        from lm_optimizer.api.main import app

        with TestClient(app) as client:
            r = client.get("/api/runs")
        assert r.status_code == 200, r.text
        rows = r.json()["runs"]
        assert rows, "run missing from list"
        row = rows[0]
        assert row["best_gen_tok_s"] == 33.3
        assert round(row["best_quality"], 3) == 0.981
        assert row["best_context_length"] == 8192
        assert row["has_validated_winner"] is True

    def test_run_without_winner_has_null_metrics(self, monkeypatch):
        from lm_optimizer.database.repositories import config_repo, run_repo

        _patch_client(monkeypatch, [])
        _save_model("m2")
        run = OptimizationRun(
            model=ModelIdentity(id="m2", name="m2"), hardware=_hw(),
            profile=OptimizationProfile.BALANCED,
        )
        run_repo.save(run)
        cfg = ConfigurationResult(
            run_id=run.id,
            config=LoadConfiguration(context_length=4096),
            context_length=4096,
            status=ConfigurationStatus.LOAD_FAILED,
            metrics=[],
            score=None,
            tested_at=datetime.now(),
        )
        config_repo.save(cfg)
        from lm_optimizer.api.main import app

        with TestClient(app) as client:
            r = client.get("/api/runs")
        row = r.json()["runs"][0]
        assert row["best_gen_tok_s"] is None
        assert row["best_quality"] is None
        assert row["has_validated_winner"] is False
