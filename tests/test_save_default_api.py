"""API endpoint that persists a run winner as LM Studio per-model defaults."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from lm_optimizer.api.main import app
from lm_optimizer.domain.models import (
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _run(*, model_id: str = "prism-ml/bonsai-27b", best: bool = True) -> OptimizationRun:
    run = OptimizationRun(
        model=ModelIdentity(id=model_id, name="Model"),
        profile=OptimizationProfile.BALANCED,
    )
    if best:
        result = ConfigurationResult(
            config=LoadConfiguration(context_length=8192, flash_attention=True),
            context_length=8192,
            status=ConfigurationStatus.PASSED,
            score=0.9,
        )
        run.configurations = [result]
        run.best_config_id = result.id
    return run


def _home(tmp_path: Path, *, model_id: str = "prism-ml/bonsai-27b") -> Path:
    home = tmp_path / "lmstudio"
    internal = home / ".internal"
    internal.mkdir(parents=True)
    (internal / "model-data.json").write_text(
        json.dumps({"json": [[model_id, {"uiIdentifier": model_id}]], "meta": {}}),
        encoding="utf-8",
    )
    path = internal / "user-concrete-model-default-config" / f"{model_id}.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {"preset": "", "operation": {"fields": []}, "load": {"fields": []}}
        ),
        encoding="utf-8",
    )
    return home


def test_save_default_unknown_run_is_404(client, monkeypatch):
    from lm_optimizer.api import routes

    monkeypatch.setattr(routes.run_repo, "get", lambda _run_id: None)
    response = client.post(f"/api/runs/{uuid4()}/save-default")
    assert response.status_code == 404
    assert "Run not found" in response.json()["detail"]


def test_save_default_without_winner_is_404(client, monkeypatch):
    from lm_optimizer.api import routes

    monkeypatch.setattr(routes.run_repo, "get", lambda _run_id: _run(best=False))
    response = client.post(f"/api/runs/{uuid4()}/save-default")
    assert response.status_code == 404
    assert "winner" in response.json()["detail"]


def test_save_default_unknown_model_is_404(client, monkeypatch, tmp_path):
    from lm_optimizer.api import routes

    monkeypatch.setattr(routes.run_repo, "get", lambda _run_id: _run(model_id="vendor/missing"))
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(tmp_path / "empty"))
    response = client.post(f"/api/runs/{uuid4()}/save-default")
    assert response.status_code == 404
    assert "model-data.json" in response.json()["detail"]


def test_save_default_writes_file_and_reports_fields(client, monkeypatch, tmp_path):
    from lm_optimizer.api import routes

    home = _home(tmp_path)
    monkeypatch.setattr(routes.run_repo, "get", lambda _run_id: _run())
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(home))

    response = client.post(f"/api/runs/{uuid4()}/save-default")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model_id"] == "prism-ml/bonsai-27b"
    assert payload["fields"]["llm.load.contextLength"] == 8192
    saved = json.loads(Path(payload["path"]).read_text(encoding="utf-8"))
    values = {f["key"]: f["value"] for f in saved["load"]["fields"]}
    assert values["llm.load.llama.flashAttention"] is True


def test_save_default_dry_run_keeps_file_unchanged(client, monkeypatch, tmp_path):
    from lm_optimizer.api import routes

    home = _home(tmp_path)
    defaults = (
        home
        / ".internal"
        / "user-concrete-model-default-config"
        / "prism-ml"
        / "bonsai-27b.json"
    )
    original = defaults.read_text(encoding="utf-8")
    monkeypatch.setattr(routes.run_repo, "get", lambda _run_id: _run())
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(home))

    response = client.post(f"/api/runs/{uuid4()}/save-default?dry_run=true")

    assert response.status_code == 200, response.text
    assert response.json()["dry_run"] is True
    assert defaults.read_text(encoding="utf-8") == original


def test_save_default_refuses_unreadable_file(client, monkeypatch, tmp_path):
    from lm_optimizer.api import routes

    home = _home(tmp_path)
    defaults = (
        home
        / ".internal"
        / "user-concrete-model-default-config"
        / "prism-ml"
        / "bonsai-27b.json"
    )
    defaults.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(routes.run_repo, "get", lambda _run_id: _run())
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(home))

    response = client.post(f"/api/runs/{uuid4()}/save-default")

    assert response.status_code == 409
    assert "unreadable" in response.json()["detail"]
