"""Windows checkpoint correctness: locks, retry, retain-old, stale cleanup (§5)."""

import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

from lm_optimizer.domain.models import (
    ConfigurationResult,
    ConfigurationStatus,
    HardwareInfo,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
)


def _patch_results_dir(tmp_path, monkeypatch):
    from lm_optimizer.config import config as _cfg

    monkeypatch.setattr(_cfg.storage, "results_dir", tmp_path)


def _state(run_id=None):
    from lm_optimizer.services.optimizer import OptimizationState
    from lm_optimizer.services.search_space import SearchSpace

    hw = HardwareInfo(
        os="T", cpu_name="C", cpu_cores_physical=1, cpu_cores_logical=2, total_ram_gb=8, gpu_count=0
    )
    run = OptimizationRun(
        model=ModelIdentity(id="m", name="m"), hardware=hw, profile=OptimizationProfile.BALANCED
    )
    if run_id is not None:
        run.id = run_id
    space = SearchSpace(
        context_lengths=[4096],
        gpu_ratios=[1.0],
        flash_attention_options=[True],
        kv_cache_options=[True],
        batch_sizes=[256],
    )
    state = OptimizationState(run=run, search_space=space)
    state.tested_configs = [
        ConfigurationResult(
            config=LoadConfiguration(context_length=4096),
            context_length=4096,
            status=ConfigurationStatus.PASSED,
            tested_at=datetime.now(),
            score=0.5,
        )
    ]
    state.started_at = datetime.now()
    return state


class TestRetryRetain:
    def test_transient_permission_error_retries(self, tmp_path, monkeypatch):
        _patch_results_dir(tmp_path, monkeypatch)
        from lm_optimizer.storage import run_checkpoint as rc

        state = _state()
        calls = {"n": 0}
        real_replace = Path.replace

        def flaky(self, target):
            calls["n"] += 1
            if calls["n"] == 1:
                raise PermissionError(13, "Access is denied")
            return real_replace(self, target)

        monkeypatch.setattr(Path, "replace", flaky)
        out = rc.save_checkpoint(state, reason="cancelled")
        assert out is not None and out.exists()
        assert calls["n"] == 2
        assert json.loads(out.read_bytes())["run_id"] == str(state.run.id)

    def test_persistent_failure_retains_previous(self, tmp_path, monkeypatch):
        _patch_results_dir(tmp_path, monkeypatch)
        from lm_optimizer.storage import run_checkpoint as rc

        state = _state()
        first = rc.save_checkpoint(state, reason="periodic")
        assert first is not None
        before = first.read_bytes()
        monkeypatch.setattr(
            Path, "replace", MagicMock(side_effect=PermissionError(13, "Access is denied"))
        )
        out = rc.save_checkpoint(state, reason="cancelled")
        assert out is None
        assert first.read_bytes() == before  # previous valid checkpoint kept
        assert json.loads(before)["run_id"] == str(state.run.id)

    def test_unique_temp_names(self, tmp_path, monkeypatch):
        _patch_results_dir(tmp_path, monkeypatch)
        from lm_optimizer.storage import run_checkpoint as rc

        seen = set()

        real_mkstemp = __import__("tempfile").mkstemp

        def spy(**kwargs):
            fd, name = real_mkstemp(**kwargs)
            seen.add(name)
            return fd, name

        monkeypatch.setattr("tempfile.mkstemp", spy)
        rc.save_checkpoint(_state(), reason="a")
        rc.save_checkpoint(_state(), reason="b")
        assert len(seen) == 2

    def test_stale_temps_cleaned(self, tmp_path, monkeypatch):
        _patch_results_dir(tmp_path, monkeypatch)
        from lm_optimizer.storage import run_checkpoint as rc

        state = _state()
        rc.save_checkpoint(state, reason="periodic")
        d = rc.checkpoint_dir()
        stale = d / f".run_{state.run.id}.json.{1234}.old.tmp"
        stale.write_bytes(b"{}")
        old = time.time() - 25 * 3600
        import os as _os

        _os.utime(stale, (old, old))
        rc.save_checkpoint(state, reason="periodic")
        assert not stale.exists()

    def test_per_run_locks_serialize(self, tmp_path, monkeypatch):
        _patch_results_dir(tmp_path, monkeypatch)
        from lm_optimizer.storage import run_checkpoint as rc

        rid = uuid4()
        assert rc._lock_for(str(rid)) is rc._lock_for(str(rid))
        assert rc._lock_for(str(rid)) is not rc._lock_for(str(uuid4()))
        one = rc.save_checkpoint(_state(rid), reason="a")
        two = rc.save_checkpoint(_state(rid), reason="b")
        assert one == two and one.exists()
