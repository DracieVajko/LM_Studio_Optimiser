"""Cross-process optimizer job lock tests (single-model guard, Task 2)."""

import json
import os
import subprocess
import sys

import psutil
import pytest

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
