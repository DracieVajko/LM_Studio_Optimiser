"""Task 3: unified pre-flight before connect (auto + optimize).

Pre-flight validation + experimental resolution + summary + confirm must run
before any backend connect. Rejection exits 2 with zero backend calls.
"""
from unittest.mock import AsyncMock, MagicMock


def _fake_client():
    fake = MagicMock()
    fake.connect = AsyncMock()
    fake.close = AsyncMock()
    fake.list_models = AsyncMock(return_value=[])
    fake.get_model = AsyncMock(return_value=None)
    return fake


def test_auto_invalid_skip_phases_exits_2_without_connect(monkeypatch):
    from typer.testing import CliRunner

    from lm_optimizer.cli import main as _cli

    fake = _fake_client()
    mk = MagicMock(return_value=fake)
    monkeypatch.setattr(_cli, "get_client", mk)
    result = CliRunner().invoke(
        _cli.app, ["auto", "--yes", "--skip-phases", "bogus-phase"]
    )
    assert result.exit_code == 2, result.output
    mk.assert_not_called()
    fake.connect.assert_not_called()


def test_auto_bad_physical_batches_exits_2_without_connect(monkeypatch):
    from typer.testing import CliRunner

    from lm_optimizer.cli import main as _cli

    fake = _fake_client()
    mk = MagicMock(return_value=fake)
    monkeypatch.setattr(_cli, "get_client", mk)
    result = CliRunner().invoke(
        _cli.app, ["auto", "--yes", "--physical-batches", "abc"]
    )
    assert result.exit_code == 2, result.output
    mk.assert_not_called()


def test_auto_bad_warmup_exits_2_without_connect(monkeypatch):
    from typer.testing import CliRunner

    from lm_optimizer.cli import main as _cli

    fake = _fake_client()
    mk = MagicMock(return_value=fake)
    monkeypatch.setattr(_cli, "get_client", mk)
    result = CliRunner().invoke(_cli.app, ["auto", "--yes", "--warmup-runs", "99"])
    assert result.exit_code == 2, result.output
    mk.assert_not_called()


def test_auto_bool_like_model_exits_2_without_connect(monkeypatch):
    from typer.testing import CliRunner

    from lm_optimizer.cli import main as _cli

    fake = _fake_client()
    mk = MagicMock(return_value=fake)
    monkeypatch.setattr(_cli, "get_client", mk)
    result = CliRunner().invoke(_cli.app, ["auto", "False", "--yes"])
    assert result.exit_code == 2, result.output
    mk.assert_not_called()
    fake.connect.assert_not_called()


def test_auto_literal_bool_flag_value_exits_2_without_connect(monkeypatch):
    from typer.testing import CliRunner

    from lm_optimizer.cli import main as _cli

    fake = _fake_client()
    mk = MagicMock(return_value=fake)
    monkeypatch.setattr(_cli, "get_client", mk)
    result = CliRunner().invoke(_cli.app, ["auto", "--yes", "--enable-rope", "False"])
    assert result.exit_code == 2, result.output
    mk.assert_not_called()
    fake.connect.assert_not_called()


def test_optimize_literal_bool_flag_value_exits_2_without_connect(monkeypatch):
    from typer.testing import CliRunner

    from lm_optimizer.cli import main as _cli

    fake = _fake_client()
    mk = MagicMock(return_value=fake)
    monkeypatch.setattr(_cli, "get_client", mk)
    result = CliRunner().invoke(
        _cli.app, ["optimize", "mymodel", "--yes", "--enable-rope", "False"]
    )
    assert result.exit_code == 2, result.output
    mk.assert_not_called()
    fake.connect.assert_not_called()


def test_auto_dry_run_summary_no_connect(monkeypatch):
    from typer.testing import CliRunner

    from lm_optimizer.cli import main as _cli

    fake = _fake_client()
    mk = MagicMock(return_value=fake)
    monkeypatch.setattr(_cli, "get_client", mk)
    result = CliRunner().invoke(_cli.app, ["auto", "--dry-run", "--yes"])
    assert result.exit_code == 0, result.output
    assert "Pre-flight Summary" in result.output
    assert "all (small first)" in result.output
    mk.assert_not_called()
    fake.connect.assert_not_called()


def test_preflight_auto_pure_no_server_call():
    from lm_optimizer.cli.main import _preflight_auto
    from pathlib import Path

    effective, explicit = _preflight_auto(
        ["m1"], "balanced", "balanced", 0.97, 2,
        False, False, None, None, None, "both", False,
        Path("out"), True, "throughput", False, 1.0, 0.0,
        1, "run", None, None, None, None, None, True, False, 60.0, 30,
    )
    assert effective["profile"] == "balanced"
    assert effective["skip_phases"] == []
    assert effective["experimental"] == {
        "enable_rope": False, "enable_cpu_moe": False, "enable_speculative": False,
    }
    assert explicit == ["m1"]


def test_optimize_invalid_warmup_exits_2_without_connect(monkeypatch):
    from typer.testing import CliRunner

    from lm_optimizer.cli import main as _cli

    fake = _fake_client()
    mk = MagicMock(return_value=fake)
    monkeypatch.setattr(_cli, "get_client", mk)
    result = CliRunner().invoke(
        _cli.app, ["optimize", "mymodel", "--yes", "--warmup-runs", "99"]
    )
    assert result.exit_code == 2, result.output
    mk.assert_not_called()
    fake.connect.assert_not_called()


def test_optimize_invalid_skip_phases_exits_2_without_connect(monkeypatch):
    from typer.testing import CliRunner

    from lm_optimizer.cli import main as _cli

    fake = _fake_client()
    mk = MagicMock(return_value=fake)
    monkeypatch.setattr(_cli, "get_client", mk)
    result = CliRunner().invoke(
        _cli.app, ["optimize", "mymodel", "--yes", "--skip-phases", "nope"]
    )
    assert result.exit_code == 2, result.output
    mk.assert_not_called()


def test_preflight_optimize_exists_and_pure():
    from lm_optimizer.cli.main import _preflight_optimize

    eff = _preflight_optimize(
        model="mymodel", profile="balanced", style="balanced",
        quality_threshold=0.97, repetitions=2, min_context=2048,
        max_context=8192, warmup_runs=1, report_sort="run",
        skip_phases=None, physical_batches=None,
        enable_rope=None, enable_cpu_moe=None, enable_speculative=None,
        yes=True, non_interactive=False, dry_run=True,
    )
    assert eff["model"] == "mymodel"
    assert eff["experimental"] == {
        "enable_rope": False, "enable_cpu_moe": False, "enable_speculative": False,
    }
