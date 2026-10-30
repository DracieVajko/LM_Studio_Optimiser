"""Stale `running` reconciliation + abandon-preserves-checkpoint (Priority 3)."""

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from lm_optimizer.domain.models import (
    HardwareInfo,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
    RunStatus,
)


def _hw():
    return HardwareInfo(
        os="T", cpu_name="C", cpu_cores_physical=1,
        cpu_cores_logical=2, total_ram_gb=8, gpu_count=0,
    )


def _save_run(mid, status=RunStatus.RUNNING):
    from lm_optimizer.database.repositories import model_repo, run_repo

    model_repo.save(ModelIdentity(id=mid, name=mid))
    run = OptimizationRun(
        model=ModelIdentity(id=mid, name=mid), hardware=_hw(),
        profile=OptimizationProfile.BALANCED,
    )
    run.status = status
    run_repo.save(run)
    return run


class TestReconcileService:
    def test_preview_reports_without_mutating(self):
        from lm_optimizer.database.repositories import run_repo
        from lm_optimizer.services.run_reconcile import reconcile_stale_runs

        run = _save_run("m")
        out = reconcile_stale_runs(dry_run=True, is_fresh=lambda rid, now: False)
        assert out["changed"] == 1
        assert out["changes"][0]["to"] == "interrupted"
        # Nothing persisted yet.
        assert run_repo.get(str(run.id)).status == RunStatus.RUNNING

    def test_apply_marks_interrupted_and_is_idempotent(self):
        from lm_optimizer.database.repositories import run_repo
        from lm_optimizer.services.run_reconcile import reconcile_stale_runs

        run = _save_run("m")
        first = reconcile_stale_runs(dry_run=False, is_fresh=lambda rid, now: False)
        assert first["changed"] == 1
        assert run_repo.get(str(run.id)).status == RunStatus.INTERRUPTED
        # Second pass: nothing left in a live state.
        second = reconcile_stale_runs(dry_run=False, is_fresh=lambda rid, now: False)
        assert second["changed"] == 0

    def test_fresh_run_is_never_touched(self):
        from lm_optimizer.database.repositories import run_repo
        from lm_optimizer.services.run_reconcile import reconcile_stale_runs

        run = _save_run("m")
        out = reconcile_stale_runs(dry_run=False, is_fresh=lambda rid, now: True)
        assert out["changed"] == 0
        assert run_repo.get(str(run.id)).status == RunStatus.RUNNING

    def test_current_live_run_is_skipped(self):
        from lm_optimizer.services.run_reconcile import reconcile_stale_runs

        run = _save_run("m")
        out = reconcile_stale_runs(
            dry_run=False, current_run_id=str(run.id), is_fresh=lambda rid, now: False
        )
        assert out["changed"] == 0

    def test_model_id_filter_scopes_reconciliation(self):
        """Auto-reconcile on optimize X must not touch other models runs."""
        from lm_optimizer.database.repositories import run_repo
        from lm_optimizer.services.run_reconcile import reconcile_stale_runs

        a = _save_run("model-a")
        b = _save_run("model-b")
        out = reconcile_stale_runs(
            dry_run=False, model_id="model-a", is_fresh=lambda rid, now: False
        )
        assert out["changed"] == 1
        assert out["changes"][0]["id"] == str(a.id)
        # model-b untouched.
        assert run_repo.get(str(b.id)).status == RunStatus.RUNNING

    def test_terminal_runs_untouched(self):
        from lm_optimizer.services.run_reconcile import reconcile_stale_runs

        _save_run("done", status=RunStatus.SUCCESS)
        out = reconcile_stale_runs(dry_run=False, is_fresh=lambda rid, now: False)
        assert out["changed"] == 0


class TestCliReconcile:
    def test_dry_run_then_apply(self, monkeypatch):
        import lm_optimizer.services.run_reconcile as rr
        from lm_optimizer.cli.main import app
        from lm_optimizer.database.repositories import run_repo

        run = _save_run("m")
        monkeypatch.setattr(rr, "_default_is_fresh", lambda rid, now: False)
        r = CliRunner().invoke(app, ["reconcile-runs"])
        assert r.exit_code == 0, r.output
        assert "Dry run" in r.output
        assert run_repo.get(str(run.id)).status == RunStatus.RUNNING
        r2 = CliRunner().invoke(app, ["reconcile-runs", "--apply"])
        assert r2.exit_code == 0, r2.output
        assert run_repo.get(str(run.id)).status == RunStatus.INTERRUPTED


class TestApiReconcileAndAbandon:
    def test_reconcile_endpoint_preview_then_apply(self, monkeypatch):
        import lm_optimizer.services.run_reconcile as rr
        from lm_optimizer.api.main import app
        from lm_optimizer.database.repositories import run_repo

        run = _save_run("m")
        monkeypatch.setattr(rr, "_default_is_fresh", lambda rid, now: False)
        with TestClient(app) as client:
            prev = client.post("/api/runs/reconcile")
            assert prev.status_code == 200, prev.text
            assert prev.json()["changed"] == 1
            assert run_repo.get(str(run.id)).status == RunStatus.RUNNING
            applied = client.post("/api/runs/reconcile?apply=true")
            assert applied.json()["changed"] == 1
            assert run_repo.get(str(run.id)).status == RunStatus.INTERRUPTED

    def test_abandon_preserves_checkpoint(self, monkeypatch):
        from lm_optimizer.api.main import app
        from lm_optimizer.database.repositories import run_repo

        run = _save_run("m")
        called = {"rm": 0}
        monkeypatch.setattr(
            "lm_optimizer.storage.run_checkpoint.remove_checkpoint",
            lambda rid: called.__setitem__("rm", called["rm"] + 1),
        )
        with TestClient(app) as client:
            r = client.post(f"/api/runs/{run.id}/abandon?force=true")
        assert r.status_code == 200, r.text
        assert run_repo.get(str(run.id)).status == RunStatus.INTERRUPTED
        assert called["rm"] == 0  # checkpoint never deleted by abandon
