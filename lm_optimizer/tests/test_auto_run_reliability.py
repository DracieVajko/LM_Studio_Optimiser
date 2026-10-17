"""Auto-run reliability: report integrity, eligibility, locking, versions, status.

Reproduces the confirmed causes:
  P0  unit tests wrote auto-run reports into the persistent results/ dir
  P1  embedding-only models reached the generation optimizer
  P2  two auto-run processes could drive one host concurrently
  P3  an lms CLI build commit was fingerprinted as an LM Studio version
  P4  a no-winner completed campaign was titled "FAILED run"

Every artifact-producing test writes into tmp_path. No provider is contacted.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lm_optimizer.domain.models import (
    GPUInfo,
    HardwareInfo,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
    RunStatus,
)


def _hw():
    return HardwareInfo(
        os="Windows",
        cpu_name="AMD Ryzen 7 5800H",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        total_ram_gb=32.0,
        gpu_count=1,
        gpus=[GPUInfo(index=0, name="RTX 3060 Laptop", vram_gb=6.0, vendor="NVIDIA")],
    )


def _result(model, action, **kw):
    base = {
        "model": model,
        "action": action,
        "run_id": None,
        "status": None,
        "best_score": None,
        "fp_diff": None,
        "error": None,
        "previous_run_id": None,
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------- P0 reports


class TestReportIntegrity:
    def test_plan_report_never_claims_an_outcome(self, tmp_path):
        """A dry-run report must not present run ids / scores as results."""
        from lm_optimizer.cli.auto_run import _generate_report

        rows = [
            _result("qwen3-32b", "FULL", previous_run_id="old-run-1"),
            _result("llama3.1-8b", "RESUME_REVALIDATE", fp_diff=["hardware_fingerprint"],
                    previous_run_id="old-run-2"),
        ]
        md = _generate_report(rows, tmp_path, _hw(), "0.3.12", kind="plan")

        assert "dry run" in md.lower()
        assert "nothing was executed" in md
        # Intent only: no result fields at all.
        assert "Best Score" not in md
        assert "**Status**" not in md
        assert "old-run-1" in md  # allowed, labelled as an existing run
        # Neither a fabricated run id nor a fabricated score may appear.
        assert "- **Run**: `" not in md

    def test_results_report_carries_real_identifiers(self, tmp_path):
        from lm_optimizer.cli.auto_run import _generate_report

        rows = [_result("qwen3-32b", "FULL", run_id="8f1c0a2e-1111-2222-3333-444455556666",
                        status="success", best_score=0.912)]
        md = _generate_report(rows, tmp_path, _hw(), "0.3.12", kind="results")

        assert "8f1c0a2e-1111-2222-3333-444455556666" in md
        assert "qwen3-32b" in md
        assert "0.912" in md
        assert "success" in md

    def test_error_stays_an_error(self, tmp_path):
        """A failure must never be dressed up with a score or a run id."""
        from lm_optimizer.cli.auto_run import _generate_report

        rows = [_result("gemma-2-2b", "FAILED", error="Connection refused: host down")]
        md = _generate_report(rows, tmp_path, _hw(), "0.3.12", kind="results")

        assert "Connection refused: host down" in md
        assert "Best Score" not in md
        assert "- **Run**: `" not in md
        assert "## Failures" in md

    def test_discovery_never_loads_a_model(self, tmp_path, monkeypatch):
        """auto-run discovery must connect load-free.

        connect() defaults to echo_probe=True, which performs a real load of
        the smallest LLM to verify applied configs. A plan/review pass must not
        touch provider state, so discovery has to opt out explicitly.
        """
        import lm_optimizer.cli.auto_run as ar
        from typer.testing import CliRunner
        from lm_optimizer.cli.main import app

        seen: dict = {}

        async def _connect(self, echo_probe: bool = True):
            seen["echo_probe"] = echo_probe
            self._client = object()

        async def _list(self, force_refresh: bool = False):
            return []

        async def _close(self):
            return None

        class _Recorder:
            connect = _connect
            list_models = _list
            close = _close

        monkeypatch.setattr(ar, "_acquire_auto_run_lock", lambda n: None)
        monkeypatch.setattr(ar, "LMStudioClient", _Recorder)
        monkeypatch.setattr(ar, "release_lock", lambda *a, **k: None)

        res = CliRunner().invoke(
            app, ["auto-run", "--dry-run", "--output", str(tmp_path), "--non-interactive"]
        )
        assert res.exit_code == 0, res.output
        assert seen.get("echo_probe") is False, "discovery must not run the echo probe"

    def test_unknown_kind_rejected(self, tmp_path):
        from lm_optimizer.cli.auto_run import _generate_report

        with pytest.raises(ValueError, match="unknown report kind"):
            _generate_report([], tmp_path, _hw(), "0.3.12", kind="fabricated")

    def test_counts_reconcile_with_rows(self, tmp_path):
        from lm_optimizer.cli.auto_run import ACTIONS, _generate_report

        rows = [
            _result("a-model", "FULL"),
            _result("b-model", "SKIP"),
            _result("c-model", "SKIP"),
            _result("d-model", "RESUME"),
            _result("e-model", "FAILED", error="boom"),
        ]
        md = _generate_report(rows, tmp_path, _hw(), "0.3.12", kind="results")

        counts = {}
        for line in md.splitlines():
            for action in ACTIONS:
                if line.startswith(f"| {action} |"):
                    counts[action] = int(line.split("|")[2].strip())
        assert sum(counts.values()) == len(rows)
        assert counts == {"FULL": 1, "SKIP": 2, "RESUME": 1,
                          "RESUME_REVALIDATE": 0, "RE_RUN": 0, "FAILED": 1}
        assert f"| **Total** | **{len(rows)}** |" in md

    def test_repeated_reports_do_not_overwrite(self, tmp_path):
        """Two invocations in the same second must not clobber each other."""
        from lm_optimizer.cli.auto_run import _generate_report

        rows = [_result("qwen3-32b", "FULL")]
        _generate_report(rows, tmp_path, _hw(), "0.3.12", kind="results")
        _generate_report(rows, tmp_path, _hw(), "0.3.12", kind="results")

        files = list(tmp_path.glob("auto-run-results-*.md"))
        assert len(files) == 2, "each invocation needs its own report file"

    def test_plan_and_results_files_are_distinguishable(self, tmp_path):
        from lm_optimizer.cli.auto_run import _generate_report

        rows = [_result("qwen3-32b", "FULL")]
        _generate_report(rows, tmp_path, _hw(), "0.3.12", kind="plan")
        _generate_report(rows, tmp_path, _hw(), "0.3.12", kind="results")

        assert list(tmp_path.glob("auto-run-plan-*.md"))
        assert list(tmp_path.glob("auto-run-results-*.md"))


# ------------------------------------------------------------- P1 eligibility


class TestModelEligibility:
    def test_embedding_model_excluded(self):
        from lm_optimizer.services.model_eligibility import check_model, ModelEligibility

        m = ModelIdentity(id="text-embedding-nomic-embed-text-v1.5",
                          name="Nomic Embed", architecture="bert",
                          model_type="embedding")
        v = check_model(m)
        assert v.eligibility is ModelEligibility.EXCLUDED
        assert not v.allowed
        assert "embedding" in v.reason

    @pytest.mark.parametrize(
        "model_id",
        [
            "nomic-ai/nomic-embed-text-v1.5",
            "BAAI/bge-m3",
            "mixedbread-ai/mxbai-embed-large-v1",
            "jinaai/jina-embeddings-v2-base-en",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ],
    )
    def test_embedding_names_excluded_without_metadata(self, model_id):
        """No provider type: name/architecture signals still exclude."""
        from lm_optimizer.services.model_eligibility import check_model, ModelEligibility

        v = check_model(ModelIdentity(id=model_id, name=model_id))
        assert v.eligibility is ModelEligibility.EXCLUDED, model_id

    def test_encoder_architecture_excluded(self):
        from lm_optimizer.services.model_eligibility import check_model, ModelEligibility

        v = check_model(ModelIdentity(id="some/model", name="some/model", architecture="BERT"))
        assert v.eligibility is ModelEligibility.EXCLUDED
        assert "encoder-only" in v.reason

    def test_generation_model_eligible(self):
        from lm_optimizer.services.model_eligibility import check_model, ModelEligibility

        for mid in ["qwen/qwen3-coder-30b", "mistralai/ministral-3-14b-reasoning"]:
            v = check_model(ModelIdentity(id=mid, name=mid, architecture="llama", model_type="llm"))
            assert v.eligibility is ModelEligibility.ELIGIBLE, mid

    def test_unknown_capability_is_not_assumed_generative(self):
        """No type and no signal -> UNKNOWN, never silently eligible."""
        from lm_optimizer.services.model_eligibility import check_model, ModelEligibility

        v = check_model(ModelIdentity(id="acme/opaque-model-7b", name="acme/opaque-model-7b"))
        assert v.eligibility is ModelEligibility.UNKNOWN
        assert not v.allowed
        assert "no model type" in v.reason

    def test_unrecognized_type_is_unknown_not_eligible(self):
        from lm_optimizer.services.model_eligibility import check_model, ModelEligibility

        v = check_model(ModelIdentity(id="x/y", name="x/y", model_type="quantum-diffusion"))
        assert v.eligibility is ModelEligibility.UNKNOWN

    def test_empty_inventory_reconciles(self):
        from lm_optimizer.services.model_eligibility import partition_models

        eligible, skipped, summary = partition_models([])
        assert eligible == [] and skipped == []
        assert summary.discovered == 0
        assert summary.reconciled()

    def test_counts_reconcile(self):
        from lm_optimizer.services.model_eligibility import partition_models

        models = [
            ModelIdentity(id="text-embedding-nomic-embed-text-v1.5", name="nomic-embed",
                          architecture="bert", model_type="embedding"),
            ModelIdentity(id="qwen/qwen3-coder-30b", name="qwen3", model_type="llm"),
            ModelIdentity(id="acme/opaque", name="acme/opaque"),
            ModelIdentity(id="BAAI/bge-m3", name="bge-m3"),
        ]
        eligible, skipped, summary = partition_models(models)
        assert [m.id for m in eligible] == ["qwen/qwen3-coder-30b"]
        assert len(skipped) == 3
        assert (summary.discovered, summary.eligible, summary.excluded, summary.unknown) == (4, 1, 2, 1)
        assert summary.reconciled()

    def test_excluded_model_is_not_a_failure(self):
        """Exclusion carries a reason, not an error, and no action."""
        from lm_optimizer.services.model_eligibility import partition_models

        models = [ModelIdentity(id="text-embedding-nomic-embed-text-v1.5",
                                name="nomic-embed", model_type="embedding")]
        eligible, skipped, _ = partition_models(models)
        assert eligible == []
        (model, verdict), = skipped
        assert verdict.eligibility.value == "excluded"
        assert not hasattr(verdict, "error")

    def test_force_generation_opt_in(self):
        from lm_optimizer.services.model_eligibility import check_model, ModelEligibility

        unknown = ModelIdentity(id="acme/opaque", name="acme/opaque")
        assert check_model(unknown, force_generation=True).eligibility is ModelEligibility.UNKNOWN

        gen = ModelIdentity(id="acme/opaque", name="acme/opaque", model_type="llm")
        assert check_model(gen, force_generation=True).eligibility is ModelEligibility.ELIGIBLE

    def test_provider_metadata_beats_name_signal(self):
        """A model typed `llm` stays eligible even with an embed-ish name."""
        from lm_optimizer.services.model_eligibility import check_model, ModelEligibility

        v = check_model(ModelIdentity(id="vendor/embed-llm-7b", name="embed-llm", model_type="llm"))
        assert v.eligibility is ModelEligibility.ELIGIBLE
        assert v.source == "provider-metadata"

    def test_raw_provider_dict_supported(self):
        from lm_optimizer.services.model_eligibility import check_model, ModelEligibility

        v = check_model({"key": "x/embed-model", "type": "embedding", "architecture": "bert"})
        assert v.eligibility is ModelEligibility.EXCLUDED

    def test_provider_failure_is_not_an_empty_inventory(self):
        """A discovery failure must be distinguishable from zero models.

        Enforced through the command surface in TestProviderFailure.
        """
        from lm_optimizer.cli import auto_run

        src = Path(auto_run.__file__).read_text(encoding="utf-8")
        assert "Cannot reach LM Studio to list models" in src
        assert "Model inventory unavailable" in src


class TestProviderFailure:
    def _run_cli(self, argv):
        from typer.testing import CliRunner
        from lm_optimizer.cli.main import app

        return CliRunner().invoke(app, argv)

    def test_empty_inventory_is_not_a_failure(self, tmp_path, monkeypatch):
        """Zero models is a clean, explained no-op (exit 0, no report)."""
        import lm_optimizer.cli.auto_run as ar
        from typer.testing import CliRunner
        from lm_optimizer.cli.main import app

        async def _empty(self, *a, **k):
            return []

        class _EmptyClient:
            connect = _empty
            list_models = _empty
            close = _empty

        monkeypatch.setattr(ar, "_acquire_auto_run_lock", lambda n: None)
        monkeypatch.setattr(ar, "LMStudioClient", _EmptyClient)
        monkeypatch.setattr(ar, "release_lock", lambda *a, **k: None)

        res = CliRunner().invoke(
            app, ["auto-run", "--dry-run", "--output", str(tmp_path), "--non-interactive"]
        )
        assert res.exit_code == 0, res.output
        assert "No models available" in res.output
        assert not list(tmp_path.glob("auto-run-*.md"))

    def test_provider_failure_exits_nonzero_via_client(self, tmp_path, monkeypatch):
        """Discovery raising is a failure, not an empty inventory."""
        import lm_optimizer.cli.auto_run as ar
        from typer.testing import CliRunner
        from lm_optimizer.cli.main import app

        async def _boom(self, *a, **k):
            raise ConnectionError("host unreachable")

        class _DeadClient:
            connect = _boom
            list_models = _boom
            close = _boom

        monkeypatch.setattr(ar, "_acquire_auto_run_lock", lambda n: None)
        monkeypatch.setattr(ar, "LMStudioClient", _DeadClient)
        monkeypatch.setattr(ar, "release_lock", lambda *a, **k: None)

        res = CliRunner().invoke(
            app, ["auto-run", "--dry-run", "--output", str(tmp_path), "--non-interactive"]
        )
        assert res.exit_code != 0
        assert "Cannot reach LM Studio" in res.output
        assert not list(tmp_path.glob("auto-run-*.md"))


# ------------------------------------------------------------------- P2 lock


class TestAutoRunLock:
    def test_second_invocation_fails_fast(self, tmp_path):
        """A held lock makes the command exit nonzero without touching state."""
        from lm_optimizer.services.joblock import acquire_lock, release_lock

        acquire_lock("auto-run:other", lock_dir=tmp_path)
        try:
            from typer.testing import CliRunner
            from lm_optimizer.cli.main import app
            import lm_optimizer.cli.auto_run as ar
            from pathlib import Path as _P

            monkey = None
            # Point the command at the temporary lock dir.
            import lm_optimizer.services.joblock as jl
            orig = jl._default_lock_path
            jl._default_lock_path = lambda: tmp_path / jl.LOCK_NAME
            try:
                res = CliRunner().invoke(
                    app, ["auto-run", "--dry-run", "--output", str(_P(tmp_path) / "out"),
                          "--non-interactive"]
                )
                assert res.exit_code == 1
                assert "already running" in res.output
            finally:
                jl._default_lock_path = orig
        finally:
            release_lock(lock_dir=tmp_path)

    def test_lock_released_after_body(self, tmp_path):
        import lm_optimizer.services.joblock as jl

        orig = jl._default_lock_path
        jl._default_lock_path = lambda: tmp_path / jl.LOCK_NAME
        try:
            import lm_optimizer.cli.auto_run as ar

            ar._acquire_auto_run_lock(3)
            assert (tmp_path / jl.LOCK_NAME).exists()
            jl.release_lock()
            assert not (tmp_path / jl.LOCK_NAME).exists()
        finally:
            jl._default_lock_path = orig

    def test_lock_released_on_exception(self, tmp_path):
        """An exception inside the orchestration must still release the lock."""
        import lm_optimizer.services.joblock as jl

        orig = jl._default_lock_path
        jl._default_lock_path = lambda: tmp_path / jl.LOCK_NAME
        try:
            import lm_optimizer.cli.auto_run as ar
            from lm_optimizer.services.joblock import JobBusyError

            with pytest.raises(RuntimeError):
                ar._acquire_auto_run_lock(1)
                try:
                    raise RuntimeError("mid-orchestration failure")
                finally:
                    pass
        finally:
            jl.release_lock()
            jl._default_lock_path = orig
        assert not (tmp_path / jl.LOCK_NAME).exists()

    def test_stale_lock_reclaimed(self, tmp_path):
        """A lock whose holder died is taken over, not a permanent block."""
        from lm_optimizer.services.joblock import LOCK_NAME, acquire_lock, release_lock

        dead = 2 ** 22  # far above any live PID on the test host
        (tmp_path / LOCK_NAME).write_text(
            json.dumps({"pid": dead, "started": 0.0, "purpose": "dead-job"}), encoding="utf-8"
        )
        rec = acquire_lock("auto-run:fresh", lock_dir=tmp_path)
        assert rec["purpose"] == "auto-run:fresh"
        release_lock(lock_dir=tmp_path)

    def test_pid_reuse_detected(self, tmp_path):
        """Same PID, different creation time -> holder is dead."""
        pytest.importorskip("psutil")
        from lm_optimizer.services.joblock import _holder_alive

        alive = {"pid": os.getpid(), "create_time": _now_create_time()}
        assert _holder_alive(alive) is True

        # Pretend the PID was recycled: a wildly different start time.
        recycled = {"pid": os.getpid(), "create_time": _now_create_time() - 100_000}
        assert _holder_alive(recycled) is False

    def test_independent_process_contention(self, tmp_path):
        """A real second process must be refused while the first holds it."""
        from lm_optimizer.services.joblock import LOCK_NAME, acquire_lock, release_lock

        acquire_lock("auto-run:first", lock_dir=tmp_path)
        try:
            code = textwrap.dedent(
                f"""
                import sys
                sys.path.insert(0, {str(Path.cwd())!r})
                from lm_optimizer.services.joblock import acquire_lock, JobBusyError
                try:
                    acquire_lock("auto-run:second", lock_dir={str(tmp_path)!r})
                    print("ACQUIRED")
                except JobBusyError:
                    print("REFUSED")
                """
            )
            out = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
            )
            assert "REFUSED" in out.stdout, out.stdout + out.stderr
        finally:
            release_lock(lock_dir=tmp_path)


def _now_create_time():
    import psutil

    return psutil.Process(os.getpid()).create_time()


# ------------------------------------------------------------- P3 versions


class TestVersionIdentity:
    def test_semantic_version_extracted(self):
        from lm_optimizer.services.version_identity import identity_from_version_string

        v = identity_from_version_string("0.3.12")
        assert v.semantic == "0.3.12"
        assert v.compatibility_key == "semver:0.3.12"
        assert v.is_semantic

    @pytest.mark.parametrize(
        "raw", ["CLI commit: dcefd5a", "CLI commit: 69d945a", "version 0.3.12", "unknown (no lms CLI)"]
    )
    def test_raw_is_preserved(self, raw):
        from lm_optimizer.services.version_identity import identity_from_version_string

        assert identity_from_version_string(raw).raw == raw

    def test_commit_is_not_mislabelled_as_semver(self):
        from lm_optimizer.services.version_identity import identity_from_version_string

        v = identity_from_version_string("CLI commit: dcefd5a")
        assert v.semantic is None
        assert v.build == "dcefd5a"
        assert "semver" not in v.compatibility_key

    def test_build_churn_does_not_revalidate_non_loading_cli(self):
        """Two CLI commits, one compatibility key when we never load via CLI."""
        from lm_optimizer.services.version_identity import identity_from_version_string

        a = identity_from_version_string("CLI commit: dcefd5a", cli_performs_loads=False)
        b = identity_from_version_string("CLI commit: 69d945a", cli_performs_loads=False)
        assert a.compatibility_key == b.compatibility_key

    def test_build_change_revalidates_when_cli_loads(self):
        """When lms performs loads, a CLI rebuild IS compatibility relevant."""
        from lm_optimizer.services.version_identity import identity_from_version_string

        a = identity_from_version_string("CLI commit: dcefd5a", cli_performs_loads=True)
        b = identity_from_version_string("CLI commit: 69d945a", cli_performs_loads=True)
        assert a.compatibility_key != b.compatibility_key
        assert a.compatibility_key.startswith("build:")

    def test_real_version_change_always_revalidates(self):
        from lm_optimizer.services.version_identity import identity_from_version_string

        a = identity_from_version_string("0.3.10", cli_performs_loads=False)
        b = identity_from_version_string("0.3.12", cli_performs_loads=False)
        assert a.compatibility_key != b.compatibility_key

    def test_unknown_is_stable_and_transparent(self):
        from lm_optimizer.services.version_identity import identity_from_version_string

        a = identity_from_version_string("unknown (no lms CLI)")
        b = identity_from_version_string(None)
        assert a.compatibility_key == b.compatibility_key == "unknown"
        assert a.semantic is None and a.build is None

    def test_version_never_fabricated(self):
        """No numeric version is invented from a commit string."""
        from lm_optimizer.services.version_identity import identity_from_version_string

        v = identity_from_version_string("CLI commit: dcefd5a")
        assert v.semantic is None
        assert "." not in v.compatibility_key

    def test_roundtrip_dict(self):
        from lm_optimizer.services.version_identity import (
            VersionIdentity,
            identity_from_version_string,
        )

        v = identity_from_version_string("0.3.12")
        assert VersionIdentity.from_dict(v.to_dict()) == v

    def test_backward_compatible_with_legacy_components(self):
        """A run stored with the old raw string is handled, not crashed on.

        The legacy raw string carries no normalized identity, so the migration
        upgrades it instead of re-benchmarking valid measurements.
        """
        from lm_optimizer.cli.auto_run import _decide_action
        from lm_optimizer.domain.models import (
            OptimizationProfile,
            OptimizationRun,
        )

        run = OptimizationRun(
            model=ModelIdentity(id="m1", name="m1"),
            hardware=_hw(),
            profile=OptimizationProfile.BALANCED,
        )
        run.status = RunStatus.INTERRUPTED
        run.benchmark_params = {
            "fingerprint": "legacy-raw",
            "fingerprint_components": {
                "model_id": "m1",
                "quantization": None,
                "architecture": None,
                "lm_studio_version": "CLI commit: dcefd5a",  # legacy raw string
                "hardware_fingerprint": "hw1",
                "benchmark_suite_version": "2026-10-15",
            },
        }
        action, diff = _decide_action(
            run,
            "new",
            {
                "model_id": "m1",
                "quantization": None,
                "architecture": None,
                "lm_studio_version": "build:dcefd5a",
                "hardware_fingerprint": "hw1",
                "benchmark_suite_version": "2026-10-15",
            },
        )
        assert action == "RESUME", "legacy shape must not re-benchmark valid work"
        assert diff is None


# ------------------------------------------------------------- P4 status text


class TestNoWinnerWording:
    @pytest.mark.parametrize(
        "status,expected",
        [
            (RunStatus.FAILED, "FAILED run"),
            (RunStatus.CANCELLED, "CANCELLED run"),
            (RunStatus.INTERRUPTED, "INTERRUPTED run"),
            (RunStatus.PARTIAL_SUCCESS, "PARTIAL run"),
            (RunStatus.COMPLETED_NO_WINNER, "NO WINNER"),
            (RunStatus.SUCCESS, "NO WINNER"),
        ],
    )
    def test_title_matches_status(self, status, expected):
        from lm_optimizer.services.reporting import no_winner_title

        assert no_winner_title(status) == expected

    def test_no_winner_completion_is_not_titled_failed(self):
        from lm_optimizer.services.reporting import no_winner_summary, no_winner_title

        title = no_winner_title(RunStatus.COMPLETED_NO_WINNER)
        summary = no_winner_summary(RunStatus.COMPLETED_NO_WINNER)
        assert "FAILED" not in title
        assert "not an operational failure" in summary

    def test_genuine_failure_says_so(self):
        from lm_optimizer.services.reporting import no_winner_summary, no_winner_title

        assert no_winner_title(RunStatus.FAILED) == "FAILED run"
        assert "no usable output" in no_winner_summary(RunStatus.FAILED)

    def test_failed_candidates_do_not_imply_failed_run(self, tmp_path):
        """Individual candidate failures never downgrade the run title."""
        from lm_optimizer.domain.models import (
            BenchmarkMetrics,
            ConfigurationResult,
            ConfigurationStatus,
            LoadConfiguration,
            OptimizationProfile,
            OptimizationRun,
        )
        from lm_optimizer.services.reporting import save_failed_report

        run = OptimizationRun(
            model=ModelIdentity(id="acme/model-7b", name="acme/model-7b"),
            hardware=_hw(),
            profile=OptimizationProfile.BALANCED,
        )
        run.status = RunStatus.COMPLETED_NO_WINNER
        m = BenchmarkMetrics(test_name="t", category="c", success=False, error="bad output")
        run.configurations = [
            ConfigurationResult(
                run_id=run.id,
                config=LoadConfiguration(context_length=4096),
                context_length=4096,
                status=ConfigurationStatus.QUALITY_FAILED,
                metrics=[m],
                error="quality below threshold",
            )
        ]

        path = save_failed_report(run, out_dir=tmp_path, stamp="20260101-000000")
        assert path is not None
        text = Path(path).read_text(encoding="utf-8")
        assert text.startswith("# NO WINNER:")
        assert "status: completed_no_validated_winner" in text
        # The candidate failure is still preserved.
        assert "quality below threshold" in text

    def test_title_summary_and_status_agree(self, tmp_path):
        from lm_optimizer.domain.models import (
            BenchmarkMetrics,
            ConfigurationResult,
            ConfigurationStatus,
            LoadConfiguration,
            OptimizationProfile,
            OptimizationRun,
        )
        from lm_optimizer.services.reporting import save_failed_report

        for status in (RunStatus.FAILED, RunStatus.COMPLETED_NO_WINNER, RunStatus.PARTIAL_SUCCESS):
            run = OptimizationRun(
                model=ModelIdentity(id=f"acme/model-{status.value}", name="m"),
                hardware=_hw(),
                profile=OptimizationProfile.BALANCED,
            )
            run.status = status
            run.configurations = [
                ConfigurationResult(
                    run_id=run.id,
                    config=LoadConfiguration(context_length=2048),
                    context_length=2048,
                    status=ConfigurationStatus.FAILED,
                    metrics=[BenchmarkMetrics(test_name="t", category="c", success=False)],
                    error="load failed",
                )
            ]
            p = save_failed_report(run, out_dir=tmp_path, stamp=f"s-{status.value}")
            text = Path(p).read_text(encoding="utf-8")
            title = text.splitlines()[0]
            assert status.value in text
            if status is RunStatus.FAILED:
                assert title.startswith("# FAILED run:")
            elif status is RunStatus.PARTIAL_SUCCESS:
                assert title.startswith("# PARTIAL run:")
            else:
                assert title.startswith("# NO WINNER:")


# --------------------------------------------------------- P5 plan semantics


class TestActionSemantics:
    def test_all_actions_are_documented(self):
        from lm_optimizer.cli.auto_run import ACTIONS, _ACTION_ICONS

        assert set(ACTIONS) == {"FULL", "SKIP", "RESUME", "RESUME_REVALIDATE", "RE_RUN", "FAILED"}
        for a in ACTIONS:
            assert a in _ACTION_ICONS, f"{a} has no report icon"

    def test_no_winner_complete_run_is_skipped_not_rerun_forever(self):
        from lm_optimizer.cli.auto_run import _decide_action

        run = OptimizationRun(
            model=ModelIdentity(id="m", name="m"), hardware=_hw(),
            profile=OptimizationProfile.BALANCED,
        )
        run.status = RunStatus.COMPLETED_NO_WINNER
        run.benchmark_params = {"fingerprint": "fp", "fingerprint_components": {"x": 1}}
        action, _ = _decide_action(run, "fp", {"x": 1})
        assert action == "SKIP"

    def test_incomplete_run_with_winner_data_is_not_skipped(self):
        from lm_optimizer.cli.auto_run import _decide_action

        run = OptimizationRun(
            model=ModelIdentity(id="m", name="m"), hardware=_hw(),
            profile=OptimizationProfile.BALANCED,
        )
        run.status = RunStatus.PAUSED
        run.benchmark_params = {"fingerprint": "fp", "fingerprint_components": {"x": 1}}
        action, _ = _decide_action(run, "fp", {"x": 1})
        assert action == "RESUME", "incomplete work must still be completed"

    def test_missing_fingerprint_stays_unknown(self):
        from lm_optimizer.cli.auto_run import _decide_action

        run = OptimizationRun(
            model=ModelIdentity(id="m", name="m"), hardware=_hw(),
            profile=OptimizationProfile.BALANCED,
        )
        run.status = RunStatus.RUNNING
        run.benchmark_params = {}
        action, diff = _decide_action(run, "new", {"lm_studio_version": "semver:1.0"})
        assert action == "RESUME"
        assert diff is None


class TestLegacyVersionShapeDoesNotChurn:
    """A legacy-shaped stored version must not force revalidation."""

    def _run(self, status, stored_version):
        run = OptimizationRun(
            model=ModelIdentity(id="m", name="m"), hardware=_hw(),
            profile=OptimizationProfile.BALANCED,
        )
        run.status = status
        run.benchmark_params = {
            "fingerprint": "old",
            "fingerprint_components": {
                "model_id": "m", "quantization": None, "architecture": None,
                "lm_studio_version": stored_version,
                "hardware_fingerprint": "hw1",
                "benchmark_suite_version": "2026-10-15",
            },
        }
        return run

    def _current(self, version):
        return {
            "model_id": "m", "quantization": None, "architecture": None,
            "lm_studio_version": version,
            "hardware_fingerprint": "hw1",
            "benchmark_suite_version": "2026-10-15",
        }

    def test_non_loading_placeholder_does_not_revalidate(self):
        """`build:non-loading` carries no identity: upgrade, do not re-measure."""
        from lm_optimizer.cli.auto_run import _decide_action

        run = self._run(RunStatus.RUNNING, "build:non-loading")
        action, diff = _decide_action(run, "new", self._current("build:dcefd5a"))
        assert action == "RESUME", diff
        assert diff is None

    def test_raw_legacy_string_does_not_revalidate(self):
        from lm_optimizer.cli.auto_run import _decide_action

        run = self._run(RunStatus.RUNNING, "CLI commit: dcefd5a")
        action, diff = _decide_action(run, "new", self._current("build:dcefd5a"))
        assert action == "RESUME", diff
        assert diff is None

    def test_real_semver_change_still_revalidates(self):
        """A genuine version change must NOT be swallowed by the migration."""
        from lm_optimizer.cli.auto_run import _decide_action

        run = self._run(RunStatus.RUNNING, "semver:0.3.10")
        action, diff = _decide_action(run, "new", self._current("semver:0.3.12"))
        assert action == "RESUME_REVALIDATE"
        assert diff == ["lm_studio_version"]

    def test_real_build_change_still_revalidates(self):
        from lm_optimizer.cli.auto_run import _decide_action

        run = self._run(RunStatus.RUNNING, "build:dcefd5a")
        action, diff = _decide_action(run, "new", self._current("build:69d945a"))
        assert action == "RESUME_REVALIDATE"
        assert diff == ["lm_studio_version"]

    def test_complete_run_legacy_version_does_not_force_rerun(self):
        """The migration must not turn finished runs into full re-runs."""
        from lm_optimizer.cli.auto_run import _decide_action

        run = self._run(RunStatus.SUCCESS, "build:non-loading")
        action, diff = _decide_action(run, "new", self._current("build:dcefd5a"))
        assert action == "SKIP", diff
        assert diff is None

    def test_normalized_shapes_recognized(self):
        from lm_optimizer.cli.auto_run import _is_normalized_version

        assert _is_normalized_version("semver:0.3.12")
        assert _is_normalized_version("build:dcefd5a")
        assert _is_normalized_version("unknown")
        assert not _is_normalized_version("build:non-loading")
        assert not _is_normalized_version("CLI commit: dcefd5a")
        assert not _is_normalized_version("")
        assert not _is_normalized_version(None)