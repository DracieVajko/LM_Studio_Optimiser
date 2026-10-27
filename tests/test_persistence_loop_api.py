"""API endpoints for defaults diff, restore and verify."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from lm_optimizer.api import routes
from lm_optimizer.api.client import LoadConfig, LoadModelResponse, UnloadModelResponse
from lm_optimizer.api.main import app
from lm_optimizer.domain.models import (
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
)
from lm_optimizer.services.lmstudio_defaults import create_backup, resolve_defaults_path


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


class FakeClient:
    def __init__(self, *, loaded_config=None, success=True, identifier="inst-1", unload_ok=True):
        self.loaded_config = loaded_config
        self.success = success
        self.identifier = identifier
        self.unload_ok = unload_ok
        self.unloaded = []
        self.loaded = []

    async def load_model(self, model_id, load_config):
        self.loaded.append((model_id, load_config.to_api_params() if hasattr(load_config, "to_api_params") else load_config))
        return LoadModelResponse(success=self.success, identifier=self.identifier, loaded_config=self.loaded_config)

    async def unload_model(self, *, identifier=None, model_id=None):
        self.unloaded.append(identifier)
        return UnloadModelResponse(success=self.unload_ok)

    async def close(self):
        pass


async def _fake_get_client(echo_probe=True):
    return None


def test_model_default_returns_exists_fields_backups(client, monkeypatch, tmp_path):
    home = _home(tmp_path)
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(home))

    response = client.get("/api/model-default?model_id=prism-ml/bonsai-27b")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["exists"] is True
    assert isinstance(payload["fields"], dict)
    assert isinstance(payload["backups"], list)


def test_model_default_lists_backups(client, monkeypatch, tmp_path):
    home = _home(tmp_path)
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(home))
    path = resolve_defaults_path("prism-ml/bonsai-27b", home=home)
    backup = create_backup(path)

    response = client.get("/api/model-default?model_id=prism-ml/bonsai-27b")

    assert response.status_code == 200, response.text
    assert backup.name in response.json()["backups"]


def test_default_diff_returns_changes(client, monkeypatch, tmp_path):
    home = _home(tmp_path)
    monkeypatch.setattr(routes.run_repo, "get", lambda _run_id: _run())
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(home))

    response = client.get(f"/api/runs/{uuid4()}/default-diff")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload["changes"], list)
    assert len(payload["changes"]) > 0


def test_restore_latest_returns_restored_from(client, monkeypatch, tmp_path):
    home = _home(tmp_path)
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(home))
    path = resolve_defaults_path("prism-ml/bonsai-27b", home=home)
    backup = create_backup(path)
    # modify current file so restore is not a no-op
    path.write_text(
        json.dumps({"preset": "", "operation": {"fields": []}, "load": {"fields": [{"key": "llm.load.contextLength", "value": 999}]}}),
        encoding="utf-8",
    )

    response = client.post(
        "/api/model-default/restore",
        json={"model_id": "prism-ml/bonsai-27b", "backup": "latest", "dry_run": False},
    )

    assert response.status_code == 200, response.text
    assert response.json()["restored_from"] == backup.name


def test_restore_unknown_backup_returns_400(client, monkeypatch, tmp_path):
    home = _home(tmp_path)
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(home))

    response = client.post(
        "/api/model-default/restore",
        json={"model_id": "prism-ml/bonsai-27b", "backup": "nope", "dry_run": False},
    )

    assert response.status_code == 400, response.text


def test_verify_default_calls_fake_client_and_returns_verified(client, monkeypatch, tmp_path):
    home = _home(tmp_path)
    monkeypatch.setattr(routes.run_repo, "get", lambda _run_id: _run())
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(home))

    fake_client = FakeClient(loaded_config=LoadConfig(context_length=8192, flash_attention=True))

    async def fake_get_lm_client(echo_probe=True):
        return fake_client

    monkeypatch.setattr(routes, "get_lm_client", fake_get_lm_client)

    response = client.post(f"/api/runs/{uuid4()}/verify-default")

    assert response.status_code == 200, response.text
    assert response.json()["verified"] is True
    assert fake_client.loaded  # client was called


def test_save_default_verify_true_includes_verification(client, monkeypatch, tmp_path):
    home = _home(tmp_path)
    monkeypatch.setattr(routes.run_repo, "get", lambda _run_id: _run())
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(home))

    fake_client = FakeClient(loaded_config=LoadConfig(context_length=8192, flash_attention=True))

    async def fake_get_lm_client(echo_probe=True):
        return fake_client

    monkeypatch.setattr(routes, "get_lm_client", fake_get_lm_client)

    response = client.post(f"/api/runs/{uuid4()}/save-default?verify=true")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "verification" in payload
    assert payload["verification"]["verified"] is True


def test_save_default_verify_dry_run_skips(client, monkeypatch, tmp_path):
    home = _home(tmp_path)
    monkeypatch.setattr(routes.run_repo, "get", lambda _run_id: _run())
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(home))

    response = client.post(f"/api/runs/{uuid4()}/save-default?dry_run=true&verify=true")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["verification"] == {"skipped": "dry_run"}