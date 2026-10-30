"""`lm-optimizer lab` persistence loop CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from typer.testing import CliRunner

from lm_optimizer.cli import main as cli_main
from lm_optimizer.domain.models import (
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
)


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


def _home(tmp_path: Path) -> Path:
    home = tmp_path / "lmstudio"
    internal = home / ".internal"
    internal.mkdir(parents=True)
    (internal / "model-data.json").write_text(
        json.dumps(
            {
                "json": [["prism-ml/bonsai-27b", {"uiIdentifier": "prism-ml/bonsai-27b"}]],
                "meta": {},
            }
        ),
        encoding="utf-8",
    )
    path = internal / "user-concrete-model-default-config" / "prism-ml" / "bonsai-27b.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"preset": "", "operation": {"fields": []}, "load": {"fields": []}}),
        encoding="utf-8",
    )
    return home


def test_diff_default_prints_changes(monkeypatch, tmp_path):
    from lm_optimizer.database.repositories import run_repo
    home = _home(tmp_path)
    monkeypatch.setattr(run_repo, "get", lambda _run_id: _run())
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(home))
    result = CliRunner().invoke(cli_main.app, ["lab", "diff-default", str(uuid4())])
    assert result.exit_code == 0, result.output
    assert "change(s)" in result.output


def test_restore_default_dry_run(monkeypatch, tmp_path):
    from lm_optimizer.database.repositories import run_repo
    from lm_optimizer.services.lmstudio_defaults import create_backup, resolve_defaults_path
    home = _home(tmp_path)
    monkeypatch.setattr(run_repo, "get", lambda _run_id: _run())
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(home))
    path = resolve_defaults_path("prism-ml/bonsai-27b", home=home)
    create_backup(path)
    result = CliRunner().invoke(
        cli_main.app, ["lab", "restore-default", "prism-ml/bonsai-27b", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output


def test_restore_default_yes(monkeypatch, tmp_path):
    from lm_optimizer.database.repositories import run_repo
    from lm_optimizer.services.lmstudio_defaults import create_backup
    home = _home(tmp_path)
    monkeypatch.setattr(run_repo, "get", lambda _run_id: _run())
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(home))
    path = (
        home / ".internal" / "user-concrete-model-default-config" / "prism-ml" / "bonsai-27b.json"
    )
    create_backup(path)
    result = CliRunner().invoke(
        cli_main.app,
        ["lab", "restore-default", "prism-ml/bonsai-27b", "--yes", "--backup", "latest"],
    )
    assert result.exit_code == 0, result.output
    assert "Restored" in result.output or "noop" in result.output


def test_restore_default_bad_backup_exits_1(monkeypatch, tmp_path):
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(tmp_path / "empty"))
    result = CliRunner().invoke(
        cli_main.app,
        ["lab", "restore-default", "prism-ml/bonsai-27b", "--backup", "nope", "--yes"],
    )
    assert result.exit_code == 1
    assert "nope" in result.output or "not found" in result.output.lower()


def test_verify_default_verified(monkeypatch, tmp_path):
    from lm_optimizer.api import routes
    from lm_optimizer.database.repositories import run_repo
    from lm_optimizer.database.repositories import run_repo
    from lm_optimizer.api.client import LoadConfig, LoadModelResponse, UnloadModelResponse
    home = _home(tmp_path)
    monkeypatch.setattr(run_repo, "get", lambda _run_id: _run())
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(home))

    class FakeClient:
        def __init__(self):
            self.unloaded = []
        async def load_model(self, model_id, load_config):
            return LoadModelResponse(success=True, identifier="i", loaded_config=LoadConfig(context_length=8192, flash_attention=True))
        async def unload_model(self, *, identifier=None, model_id=None):
            self.unloaded.append(identifier)
            return UnloadModelResponse(success=True)
        async def close(self):
            pass

    async def fake_get_client(echo_probe=True):
        return FakeClient()

    monkeypatch.setattr(routes, "get_lm_client", fake_get_client)
    result = CliRunner().invoke(cli_main.app, ["lab", "verify-default", str(uuid4())])
    assert result.exit_code == 0, result.output
    assert "verified" in result.output

def test_test_manager_command_runs(monkeypatch):
    result = CliRunner().invoke(cli_main.app, ["lab", "test-manager"])
    assert result.exit_code == 0, result.output
    assert "Test manager started" in result.output

def test_recommend_command_runs(monkeypatch):
    result = CliRunner().invoke(cli_main.app, ["lab", "recommend", "prism-ml/bonsai-27b"])
    assert result.exit_code == 0, result.output
    assert "Recommend mode activated" in result.output

def test_clean_command_runs(monkeypatch):
    result = CliRunner().invoke(cli_main.app, ["lab", "clean", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Clean" in result.output


def test_assistant_not_used_without_env(monkeypatch):
    from lm_optimizer.services.test_manager import TestManager
    manager = TestManager()
    assert manager.can_use_assistant() is False
