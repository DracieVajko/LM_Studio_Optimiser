"""Stall-watchdog liveness math in tools/granular_suite.py.

Regression for the 2026-10-20 bug: the first version computed
`time.monotonic() - file_mtime`, always hugely negative, so the stall
watchdog never fired. _quiet_seconds must be wall-clock based and treat
inner logs (data/logs/maxctx/*.log) as liveness for maxctx steps.
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest


def _load_suite_runner():
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "granular_suite", repo / "tools" / "granular_suite.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("granular_suite", mod)
    spec.loader.exec_module(mod)
    return mod


suite = _load_suite_runner()


def test_quiet_seconds_is_wall_clock(tmp_path):
    log = tmp_path / "step.log"
    log.write_text("banner", encoding="utf-8")
    # pretend the file was written 90 s ago
    old = time.time() - 90
    os.utime(log, (old, old))
    q = suite._quiet_seconds(log, time.time() - 3600, [])
    assert 85 <= q <= 100


def test_quiet_seconds_uses_newest_inner_log(tmp_path):
    log = tmp_path / "step.log"
    log.write_text("banner", encoding="utf-8")
    stale = time.time() - 4000
    os.utime(log, (stale, stale))
    inner = tmp_path / "maxctx"
    inner.mkdir()
    hot = inner / "model__kvgpu.log"
    hot.write_text("rung output", encoding="utf-8")
    fresh = time.time() - 10
    os.utime(hot, (fresh, fresh))
    # silent step log but alive inner log -> quiet comes from inner
    assert suite._quiet_seconds(log, stale - 10, [inner]) < 60
    # inner dir stale too -> quiet grows
    os.utime(hot, (stale, stale))
    assert suite._quiet_seconds(log, stale - 10, [inner]) > 3600


def test_quiet_seconds_never_negative(tmp_path):
    missing = tmp_path / "nope.log"
    since = time.time()
    assert suite._quiet_seconds(missing, since, [tmp_path / "ghost"]) >= 0.0
