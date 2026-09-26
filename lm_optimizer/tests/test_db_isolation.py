"""Production DB must never be touched by tests (§5, §16)."""

from datetime import datetime

from lm_optimizer.domain.models import (
    HardwareInfo,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
)


def _heavy_state():
    from lm_optimizer.services.optimizer import OptimizationState
    from lm_optimizer.services.search_space import SearchSpace

    hw = HardwareInfo(os="T", cpu_name="C", cpu_cores_physical=1,
                      cpu_cores_logical=2, total_ram_gb=8, gpu_count=0)
    run = OptimizationRun(model=ModelIdentity(id="iso-ckpt", name="x"),
                          hardware=hw, profile=OptimizationProfile.BALANCED)
    space = SearchSpace(context_lengths=[4096], gpu_ratios=[1.0],
                        flash_attention_options=[True], kv_cache_options=[True],
                        batch_sizes=[256])
    return OptimizationState(run=run, search_space=space)


def _snapshot_production():
    import sqlite3

    from lm_optimizer.config import config

    con = sqlite3.connect(config.storage.database_path)
    try:
        runs = con.execute("select count(*) from runs").fetchone()[0]
        configs = con.execute("select count(*) from configurations").fetchone()[0]
        latest = con.execute("select max(created_at) from runs").fetchone()[0]
        return runs, configs, latest
    finally:
        con.close()


class TestIsolation:
    def test_db_points_at_tmp(self, _isolated_db, tmp_path):
        from lm_optimizer.database import repositories as _repos

        assert str(_repos.db_manager.db_path).startswith(str(tmp_path))

    def test_writes_stay_in_tmp(self, _isolated_db):
        from lm_optimizer.database import repositories as _repos

        before = _snapshot_production()
        hw = HardwareInfo(os="T", cpu_name="C", cpu_cores_physical=1,
                          cpu_cores_logical=2, total_ram_gb=8, gpu_count=0)
        hw_id = _repos.hardware_repo.save(hw)
        model = ModelIdentity(id="isolation-probe", name="probe")
        _repos.model_repo.save(model)
        run = OptimizationRun(model=model, hardware=hw, profile=OptimizationProfile.BALANCED)
        run.hardware_id = hw_id
        _repos.run_repo.save(run)
        assert _repos.run_repo.get(str(run.id)) is not None
        assert _snapshot_production() == before

    def test_production_checkpoints_untouched(self, _isolated_db, tmp_path):
        from pathlib import Path

        from lm_optimizer.config import config as _cfg
        from lm_optimizer.storage import run_checkpoint as rc

        assert str(_cfg.storage.results_dir).startswith(str(tmp_path))
        real_dir = Path(_cfg.storage.database_path).parent / "reports" / "checkpoints"
        before = sorted(p.name for p in real_dir.glob("run_*.json")) if real_dir.exists() else []
        # Heavy checkpoint activity lands in tmp only.
        rc.save_checkpoint(_heavy_state(), reason="test")
        rc.save_checkpoint(_heavy_state(), reason="test")
        after = sorted(p.name for p in real_dir.glob("run_*.json")) if real_dir.exists() else []
        assert after == before
