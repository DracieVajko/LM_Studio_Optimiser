"""Artifact retention tests (spec §13.5)."""
import time

import pytest

from lm_optimizer.services.sandbox_lab import SandboxPolicy
from lm_optimizer.services.sandbox_runner import apply_retention


def _make_dirs(base, count):
    dirs = []
    for i in range(count):
        d = base / f"run_{i}"
        d.mkdir(parents=True)
        (d / "artifact.txt").write_text(str(i), encoding="utf-8")
        # Spread mtimes so ordering is deterministic.
        ts = time.time() - (count - i) * 10
        import os
        os.utime(d, (ts, ts))
        dirs.append(d)
    return dirs


def test_retention_keeps_newest(tmp_path):
    dirs = _make_dirs(tmp_path, 5)
    policy = SandboxPolicy(workdir=tmp_path, artifact_retention="keep_last_2")
    removed = apply_retention(tmp_path, policy)
    assert len(removed) == 3
    # The two newest survive.
    remaining = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert remaining == ["run_3", "run_4"]


def test_retention_noop_when_under_limit(tmp_path):
    _make_dirs(tmp_path, 2)
    policy = SandboxPolicy(workdir=tmp_path, artifact_retention="keep_last_10")
    assert apply_retention(tmp_path, policy) == []


def test_retention_missing_dir_is_safe(tmp_path):
    policy = SandboxPolicy(workdir=tmp_path)
    assert apply_retention(tmp_path / "nope", policy) == []


def test_retention_bad_policy_falls_back(tmp_path):
    _make_dirs(tmp_path, 12)
    policy = SandboxPolicy(workdir=tmp_path, artifact_retention="garbage")
    removed = apply_retention(tmp_path, policy)
    # Default keep_last_10 -> two removed.
    assert len(removed) == 2


def test_retention_keep_zero(tmp_path):
    _make_dirs(tmp_path, 3)
    policy = SandboxPolicy(workdir=tmp_path, artifact_retention="keep_last_0")
    removed = apply_retention(tmp_path, policy)
    assert len(removed) == 3
    assert not [p for p in tmp_path.iterdir() if p.is_dir()]
