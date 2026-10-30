"""Resume compatibility, frozen settings and partial-run honesty (spec §9, §10)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from lm_optimizer.domain.models import (
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    HardwareInfo,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
    RunStatus,
)
from lm_optimizer.services.optimizer import (
    PERSISTED_RUN_SETTINGS,
    apply_persisted_run_settings,
    build_resolved_settings,
    run_skip_phases,
)
from lm_optimizer.services.run_summary import (
    classify_run,
    final_recommendation,
    final_status_for_run,
)


def _run(**kw):
    run = OptimizationRun(
        model=ModelIdentity(id="m", name="m", context_limit=32768),
        hardware=HardwareInfo(
            os="w", cpu_name="c", cpu_cores_physical=1,
            cpu_cores_logical=1, total_ram_gb=32, gpu_count=0,
        ),
        profile=OptimizationProfile.BALANCED,
    )
    for key, value in kw.items():
        setattr(run, key, value)
    return run


# --- frozen settings --------------------------------------------------------

def test_resolved_settings_snapshot_covers_every_new_field():
    run = _run(
        min_generation_tok_s=3.5,
        enforce_min_generation_speed=False,
        run_preset="custom",
        selected_stages=["speed", "quality"],
        max_speed_probes=9,
        max_finalists=2,
        benchmark_repetitions_override=2,
        validation_repetitions_min=2,
        validation_repetitions_max=4,
        recovery_budget=0,
        context_strategy_modes=["winner"],
        final_sandbox_mode="all",
        final_sandbox_preset_ids=["web_dashboard"],
        final_sandbox_custom_prompts=[{"prompt": "x"}],
        sandbox_compare_baseline=True,
    )
    snap = build_resolved_settings(run)
    assert set(snap) == set(PERSISTED_RUN_SETTINGS)
    assert snap["recovery_budget"] == 0  # zero is preserved, not dropped
    assert snap["min_generation_tok_s"] == 3.5
    assert snap["selected_stages"] == ["speed", "quality"]


def test_resume_restores_frozen_settings_from_benchmark_params():
    saved = {
        "min_generation_tok_s": 3.5,
        "enforce_min_generation_speed": False,
        "run_preset": "custom",
        "selected_stages": ["speed", "quality"],
        "recovery_budget": 0,
        "context_strategy_modes": ["winner"],
        "final_sandbox_mode": "custom",
    }
    run = _run(benchmark_params={"resolved_settings": saved})
    # DB rows do not carry the new columns: defaults were loaded first.
    assert run.min_generation_tok_s == 2.0

    apply_persisted_run_settings(run)

    assert run.min_generation_tok_s == 3.5
    assert run.enforce_min_generation_speed is False
    assert run.run_preset == "custom"
    assert run.selected_stages == ["speed", "quality"]
    assert run.recovery_budget == 0
    assert run.context_strategy_modes == ["winner"]
    assert run.final_sandbox_mode == "custom"


def test_legacy_run_without_snapshot_keeps_safe_defaults():
    run = _run(benchmark_params={"benchmark_repetitions": 3})
    apply_persisted_run_settings(run)
    assert run.min_generation_tok_s == 2.0
    assert run.enforce_min_generation_speed is True
    assert run.run_preset == "standard"
    assert run.final_sandbox_mode == "off"


def test_corrupt_snapshot_never_crashes_resume():
    run = _run(benchmark_params={"resolved_settings": "not-a-dict"})
    apply_persisted_run_settings(run)
    assert run.min_generation_tok_s == 2.0


def test_skip_phases_prefers_memory_then_persisted():
    run = _run(
        advanced_settings={"skip_phases": ["recovery"]},
        benchmark_params={"skip_phases": ["quality"]},
    )
    assert run_skip_phases(run) == {"recovery"}

    run.advanced_settings = {}
    assert run_skip_phases(run) == {"quality"}


# --- partial runs are not "validated" --------------------------------------

def _passed_scoreless():
    return ConfigurationResult(
        config=LoadConfiguration(context_length=4096),
        context_length=4096,
        status=ConfigurationStatus.PASSED,
        score=None,
        metrics=[
            BenchmarkMetrics(
                test_name="short_instruction",
                category="speed",
                success=True,
                generation_tok_s=9.0,
                completion_tokens=32,
            )
        ],
    )


def test_speed_only_run_cannot_be_a_validated_winner():
    run = _run(benchmark_params={"skip_phases": ["quality", "recovery", "validation", "context"]})
    probe = _passed_scoreless()
    run.configurations = [probe]
    run.best_config_id = probe.id

    assert final_recommendation([probe], run) is None
    state, _info = classify_run([probe], run)
    assert state == "NO_WINNER"
    assert final_status_for_run(run) == RunStatus.COMPLETED_NO_WINNER


def test_partial_run_status_is_not_success():
    run = _run(benchmark_params={"skip_phases": ["validation"]})
    probe = _passed_scoreless()
    run.configurations = [probe]
    state, _ = classify_run([probe], run)
    assert state != "SUCCESS"


def test_full_run_with_validated_winner_is_success():
    from lm_optimizer.domain.models import QualityScore

    winner = _passed_scoreless()
    winner.score = 0.9
    winner.quality_score = QualityScore(
        overall=0.98,
        task_completion=1.0,
        factual_consistency=1.0,
        format_compliance=1.0,
        coding_correctness=1.0,
        no_truncation=1.0,
        no_malformed=1.0,
    )
    run = _run(benchmark_params={"skip_phases": []})
    run.configurations = [winner]
    run.best_config_id = winner.id
    assert final_recommendation([winner], run) is winner
    assert final_status_for_run(run) == RunStatus.SUCCESS


# --- prepare_run persists the snapshot -------------------------------------

def test_prepare_run_persists_resolved_settings(monkeypatch):
    from lm_optimizer.services import optimizer as _opt

    client = MagicMock()
    client.capabilities = None
    benchmark = MagicMock()
    benchmark.gpu_via_cli = False
    search = MagicMock()
    search.generate.return_value.to_dict.return_value = {}
    search.generate.return_value.estimate_size.return_value = 1

    opt = _opt.AdaptiveOptimizer(client, benchmark, MagicMock(), search)
    opt._resolve_baseline = AsyncMock(return_value=None)
    opt._apply_persisted_flash_advice = lambda: None
    monkeypatch.setattr(_opt.hardware_repo, "save", lambda h: "hw")
    monkeypatch.setattr(_opt.model_repo, "save", lambda m: "m")
    monkeypatch.setattr(_opt.run_repo, "save", lambda r: str(r.id))
    monkeypatch.setattr(_opt, "compute_fingerprint", lambda *a, **k: "fp")
    monkeypatch.setattr(_opt, "hardware_fingerprint", lambda h: "hwfp")
    monkeypatch.setattr("lm_optimizer.services.lms_cli.lms_version", lambda: "test")

    hw = HardwareInfo(
        os="w", cpu_name="c", cpu_cores_physical=1,
        cpu_cores_logical=1, total_ram_gb=32, gpu_count=0,
    )
    model = ModelIdentity(id="m", name="m", context_limit=32768)
    run, _baseline, _adv = asyncio.run(
        opt.prepare_run(
            model,
            hw,
            OptimizationProfile.BALANCED,
            advanced_settings={
                "run_preset": "custom",
                "selected_stages": ["speed"],
                "min_generation_tok_s": 4.25,
                "enforce_min_generation_speed": False,
                "recovery_budget": 0,
            },
        )
    )

    snap = run.benchmark_params["resolved_settings"]
    assert snap["min_generation_tok_s"] == 4.25
    assert snap["enforce_min_generation_speed"] is False
    assert snap["selected_stages"] == ["speed"]
    assert snap["recovery_budget"] == 0

    # Round-trip: a fresh run object restores the same meaning.
    restored = _run(benchmark_params={"resolved_settings": snap})
    apply_persisted_run_settings(restored)
    assert restored.min_generation_tok_s == 4.25
    assert restored.recovery_budget == 0
