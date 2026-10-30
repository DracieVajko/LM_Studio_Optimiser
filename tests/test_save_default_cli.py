"""`lm-optimizer lab save-default` command."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

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


def test_save_default_unknown_run_exits_nonzero(monkeypatch):
    from lm_optimizer.database.repositories import run_repo

    monkeypatch.setattr(run_repo, "get", lambda _run_id: None)
    result = CliRunner().invoke(cli_main.app, ["lab", "save-default", str(uuid4())])
    assert result.exit_code == 1
    assert "Run not found" in result.output


def test_save_default_without_winner_exits_nonzero(monkeypatch):
    from lm_optimizer.database.repositories import run_repo

    monkeypatch.setattr(run_repo, "get", lambda _run_id: _run(best=False))
    result = CliRunner().invoke(cli_main.app, ["lab", "save-default", str(uuid4())])
    assert result.exit_code == 1
    assert "No validated winner" in result.output


def test_save_default_writes_file(monkeypatch, tmp_path):
    from lm_optimizer.database.repositories import run_repo

    home = _home(tmp_path)
    monkeypatch.setattr(run_repo, "get", lambda _run_id: _run())
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(home))

    result = CliRunner().invoke(cli_main.app, ["lab", "save-default", str(uuid4())])

    assert result.exit_code == 0, result.output
    assert "Saved" in result.output
    path = (
        home
        / ".internal"
        / "user-concrete-model-default-config"
        / "prism-ml"
        / "bonsai-27b.json"
    )
    values = {
        f["key"]: f["value"]
        for f in json.loads(path.read_text(encoding="utf-8"))["load"]["fields"]
    }
    assert values["llm.load.contextLength"] == 8192


def test_save_default_dry_run_writes_nothing(monkeypatch, tmp_path):
    from lm_optimizer.database.repositories import run_repo

    home = _home(tmp_path)
    path = (
        home
        / ".internal"
        / "user-concrete-model-default-config"
        / "prism-ml"
        / "bonsai-27b.json"
    )
    original = path.read_text(encoding="utf-8")
    monkeypatch.setattr(run_repo, "get", lambda _run_id: _run())
    monkeypatch.setenv("LM_OPTIMIZER_LMSTUDIO_HOME", str(home))

    result = CliRunner().invoke(
        cli_main.app, ["lab", "save-default", str(uuid4()), "--dry-run"]
    )

    assert result.exit_code == 0, result.output
    assert "dry run" in result.output.lower()
    assert path.read_text(encoding="utf-8") == original
