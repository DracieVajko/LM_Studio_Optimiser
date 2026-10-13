"""Cross-process optimizer job lock tests (single-model guard, Task 2)."""

import json
import os
import subprocess
import sys

import psutil
import pytest
from fastapi.testclient import TestClient

from lm_optimizer.services.joblock import (
    LOCK_PATH,
    JobBusyError,
    acquire_lock,
    release_lock,
)


def _dead_pid():
    """PID of a reaped child: guaranteed not alive (stale-lock fixture)."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    pid = proc.pid
    proc.wait()
    assert not psutil.pid_exists(pid)
    return pid


def test_second_acquire_refuses(tmp_path):
    acquire_lock("job-a", lock_dir=tmp_path)
    try:
        with pytest.raises(JobBusyError):
            acquire_lock("job-b", lock_dir=tmp_path)
    finally:
        release_lock(lock_dir=tmp_path)


def test_acquire_writes_pid_purpose_json(tmp_path):
    record = acquire_lock("job-a", lock_dir=tmp_path)
    try:
        assert record["pid"] == os.getpid()
        assert record["purpose"] == "job-a"
        raw = json.loads((tmp_path / ".optimizer.lock").read_text(encoding="utf-8"))
        assert raw == record
        assert set(raw) == {"pid", "started", "purpose"}
    finally:
        release_lock(lock_dir=tmp_path)


def test_busy_error_carries_holder(tmp_path):
    acquire_lock("job-a", lock_dir=tmp_path)
    try:
        with pytest.raises(JobBusyError) as exc:
            acquire_lock("job-b", lock_dir=tmp_path)
        assert exc.value.holder["pid"] == os.getpid()
        assert exc.value.holder["purpose"] == "job-a"
    finally:
        release_lock(lock_dir=tmp_path)


def test_release_is_idempotent(tmp_path):
    release_lock(lock_dir=tmp_path)  # nothing held: no-op
    acquire_lock("job-a", lock_dir=tmp_path)
    release_lock(lock_dir=tmp_path)
    release_lock(lock_dir=tmp_path)  # already released: no-op
    assert not (tmp_path / ".optimizer.lock").exists()


def test_stale_lock_taken_over(tmp_path):
    stale = {"pid": _dead_pid(), "started": 0.0, "purpose": "crashed-job"}
    (tmp_path / ".optimizer.lock").write_text(json.dumps(stale), encoding="utf-8")
    record = acquire_lock("job-b", lock_dir=tmp_path)
    try:
        assert record["pid"] == os.getpid()
        assert record["purpose"] == "job-b"
    finally:
        release_lock(lock_dir=tmp_path)


def test_release_keeps_foreign_lock(tmp_path):
    foreign = {"pid": _dead_pid(), "started": 0.0, "purpose": "other-job"}
    lock_file = tmp_path / ".optimizer.lock"
    lock_file.write_text(json.dumps(foreign), encoding="utf-8")
    release_lock(lock_dir=tmp_path)
    assert json.loads(lock_file.read_text(encoding="utf-8")) == foreign


def test_lock_path_constant_shape():
    assert LOCK_PATH.name == ".optimizer.lock"
    assert LOCK_PATH.parent.name == "reports"


def test_default_lock_uses_configured_results_dir():
    from lm_optimizer.config import config

    expected = config.storage.results_dir / ".optimizer.lock"
    record = acquire_lock("job-a")  # no lock_dir: configured results dir
    try:
        assert record["pid"] == os.getpid()
        assert expected.exists()
        with pytest.raises(JobBusyError):
            acquire_lock("job-b")
    finally:
        release_lock()
    assert not expected.exists()


def test_cli_wait_flags_default_60_30():
    """optimize/auto/fit/ctx expose --wait-interval/--max-waits defaulting to 60.0/30."""
    import typer

    from lm_optimizer.cli.main import app

    info = typer.main.get_command(app)
    for cmd in ("optimize", "auto", "fit", "ctx"):
        params = {p.name: p for p in info.commands[cmd].params}
        assert params["wait_interval"].default == 60.0, cmd
        assert params["max_waits"].default == 30, cmd


def test_cli_run_lock_refuses_second_job():
    """Second CLI job is refused with holder info (exit 1), lock kept."""
    from lm_optimizer.cli.main import _acquire_run_lock

    acquire_lock("other-job")
    try:
        with pytest.raises(SystemExit) as exc:
            _acquire_run_lock("optimize:m")
        assert exc.value.code == 1
    finally:
        release_lock()


def test_web_optimize_refuses_with_holder_409():
    """POST /optimize while locked: 409 carrying holder pid/purpose."""
    import lm_optimizer.api.routes as routes
    from lm_optimizer.api.main import app

    assert routes._current_optimizer is None
    acquire_lock("other-job")
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post("/api/optimize", json={"model_id": "m"})
        assert r.status_code == 409, r.text[:300]
        body = r.json().get("detail", "")
        assert str(os.getpid()) in body
        assert "other-job" in body
    finally:
        release_lock()


async def test_web_run_task_releases_lock():
    """Background run completion releases the job lock (lock held by test)."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    import lm_optimizer.api.routes as routes

    run = SimpleNamespace(
        id="run-1",
        status=SimpleNamespace(value="completed"),
        stage=SimpleNamespace(value="complete"),
    )
    optimizer = MagicMock()
    optimizer.execute = AsyncMock(return_value=run)
    optimizer.state = None
    acquire_lock("web-optimize:m")
    await routes.run_optimization_task(
        optimizer,
        SimpleNamespace(id="m"),
        SimpleNamespace(value="balanced"),
        run,
        None,
        {},
    )
    from lm_optimizer.services.joblock import _default_lock_path

    assert not _default_lock_path().exists()
    assert routes._current_optimizer is None
    assert routes._current_run_id is None
