"""Presets, stage selection and dependency rules (spec §2, §4.2, §7, §10)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from lm_optimizer.services.optimizer import (
    PRESET_DEFINITIONS,
    VALID_CONTEXT_STRATEGIES,
    VALID_PRESETS,
    VALID_SANDBOX_MODES,
    VALID_STAGES,
    resolve_preset,
    validate_stage_dependencies,
)


def test_documented_presets_exist():
    assert VALID_PRESETS == {"quick", "standard", "thorough", "custom"}
    for name in ("quick", "standard", "thorough"):
        p = PRESET_DEFINITIONS[name]
        assert p["max_speed_probes"] > 0
        assert p["max_finalists"] > 0
        assert p["selected_stages"]
        assert set(p["selected_stages"]) <= VALID_STAGES
        assert set(p["context_strategy_modes"]) <= VALID_CONTEXT_STRATEGIES
        assert p["final_sandbox_mode"] in VALID_SANDBOX_MODES


def test_quick_preset_budgets():
    resolved, warnings = resolve_preset("quick")
    assert resolved["max_speed_probes"] == 12
    assert resolved["max_finalists"] == 2
    assert resolved["benchmark_repetitions"] == 1
    assert resolved["validation_repetitions_min"] >= 2
    assert resolved["final_sandbox_mode"] == "off"
    assert not [w for w in warnings if "quality" in w]


def test_standard_preset_keeps_bounded_probe_cap():
    resolved, _ = resolve_preset("standard")
    assert resolved["max_speed_probes"] <= 25
    assert "quality" in resolved["selected_stages"]
    assert "recovery" in resolved["selected_stages"]


def test_thorough_preset_widens_but_stays_bounded():
    resolved, _ = resolve_preset("thorough")
    assert resolved["max_speed_probes"] <= 40
    assert resolved["max_finalists"] <= 6
    assert "context" in resolved["selected_stages"]
    assert len(resolved["context_strategy_modes"]) > len(
        PRESET_DEFINITIONS["standard"]["context_strategy_modes"]
    )


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        resolve_preset("nope")


def test_unknown_override_warns():
    _resolved, warnings = resolve_preset("quick", {"nonsense": 1})
    assert any("Unknown preset override" in w for w in warnings)


def test_custom_preset_honours_documented_overrides():
    resolved, warnings = resolve_preset(
        "custom",
        {
            "max_speed_probes": 7,
            "max_finalists": 2,
            "benchmark_repetitions": 2,
            "selected_stages": ["speed", "quality"],
            "recovery_budget": 0,
        },
    )
    assert resolved["max_speed_probes"] == 7
    assert resolved["max_finalists"] == 2
    assert resolved["benchmark_repetitions"] == 2
    assert resolved["recovery_budget"] == 0  # zero is a real budget
    assert not [w for w in warnings if "Unknown preset override" in w]


# --- dependency resolver ----------------------------------------------------

def test_quality_without_speed_is_explained():
    warnings = validate_stage_dependencies(["quality"])
    assert any("quality" in w and "speed" in w for w in warnings)


def test_preflight_is_mandatory():
    warnings = validate_stage_dependencies(["speed"])
    assert any("preflight" in w for w in warnings)


def test_recovery_requires_quality():
    warnings = validate_stage_dependencies(["speed", "recovery"])
    assert any("recovery" in w for w in warnings)


def test_validation_without_quality_is_flagged_as_unverified():
    warnings = validate_stage_dependencies(["speed", "validation"])
    assert any("quality not verified" in w for w in warnings)


def test_context_requires_a_source_config():
    warnings = validate_stage_dependencies(["context"])
    assert any("context" in w for w in warnings)


def test_experimental_requires_opt_in_note():
    warnings = validate_stage_dependencies(["speed", "experimental"])
    assert any("experimental" in w for w in warnings)


# --- CLI validation ---------------------------------------------------------

def _preflight(**kw):
    from lm_optimizer.cli import main as _cli

    base = dict(
        model="m",
        profile="balanced",
        style="balanced",
        quality_threshold=0.97,
        repetitions=3,
        min_context=2048,
        max_context=32768,
        warmup_runs=1,
        report_sort="run",
        skip_phases=None,
        physical_batches=None,
        enable_rope=None,
        enable_cpu_moe=None,
        enable_speculative=None,
        yes=True,
        non_interactive=True,
        dry_run=False,
    )
    base.update(kw)
    return _cli._preflight_optimize(**base)


def test_preflight_accepts_preset_and_stage_list():
    eff = _preflight(run_preset="custom", selected_stages=["speed", "quality"])
    assert eff["run_preset"] == "custom"
    assert eff["selected_stages"] == ["speed", "quality"]


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"run_preset": "turbo"}, "Invalid --preset"),
        ({"selected_stages": ["speed", "made_up"]}, "Invalid --stage"),
        ({"context_strategy_modes": ["magic"]}, "Invalid --context-strategy"),
        ({"final_sandbox_mode": "sometimes"}, "Invalid --final-sandbox"),
    ],
)
def test_preflight_rejects_invalid_choices(kwargs, needle):
    from typer.testing import CliRunner

    from lm_optimizer.cli import main as _cli

    args = ["optimize", "m", "--yes"]
    if "run_preset" in kwargs:
        args += ["--preset", kwargs["run_preset"]]
    if "selected_stages" in kwargs:
        for s in kwargs["selected_stages"]:
            args += ["--stage", s]
    if "context_strategy_modes" in kwargs:
        for s in kwargs["context_strategy_modes"]:
            args += ["--context-strategy", s]
    if "final_sandbox_mode" in kwargs:
        args += ["--final-sandbox", kwargs["final_sandbox_mode"]]
    result = CliRunner().invoke(_cli.app, args)
    assert result.exit_code == 2, result.output
    assert needle in result.output


# --- prepare_run honours the resolved plan ---------------------------------

def _prepared_run(monkeypatch, advanced):
    from lm_optimizer.services import optimizer as _opt

    client = MagicMock()
    client.capabilities = None
    benchmark = MagicMock()
    benchmark.gpu_via_cli = False
    search = MagicMock()
    search.generate.return_value = MagicMock()
    search.generate.return_value.to_dict.return_value = {"context_lengths": [2048]}
    search.generate.return_value.estimate_size.return_value = 1

    opt = _opt.AdaptiveOptimizer(client, benchmark, MagicMock(), search)
    opt._resolve_baseline = AsyncMock(return_value=None)
    opt._apply_persisted_flash_advice = lambda: None
    monkeypatch.setattr(_opt.hardware_repo, "save", lambda h: "hw")
    monkeypatch.setattr(_opt.model_repo, "save", lambda m: "m")
    monkeypatch.setattr(_opt.run_repo, "save", lambda r: str(r.id))
    monkeypatch.setattr(_opt, "compute_fingerprint", lambda *a, **k: "fp")
    monkeypatch.setattr(_opt, "hardware_fingerprint", lambda h: "hwfp")
    monkeypatch.setattr(
        "lm_optimizer.services.lms_cli.lms_version", lambda: "test"
    )

    from lm_optimizer.domain.models import (
        HardwareInfo,
        ModelIdentity,
        OptimizationProfile,
    )

    hw = HardwareInfo(os="w", cpu_name="c", cpu_cores_physical=1,
                      cpu_cores_logical=1, total_ram_gb=32, gpu_count=0)
    model = ModelIdentity(id="m", name="m", context_limit=32768)
    run, _baseline, _adv = asyncio.run(
        opt.prepare_run(
            model,
            hw,
            OptimizationProfile.BALANCED,
            advanced_settings=advanced,
        )
    )
    return run


def test_prepare_run_quick_preset_freezes_budgets(monkeypatch):
    run = _prepared_run(monkeypatch, {"run_preset": "quick"})
    assert run.run_preset == "quick"
    assert run.max_speed_probes == 12
    assert run.max_finalists == 2
    assert run.validation_repetitions_min >= 2
    assert run.resolved_preset["max_speed_probes"] == 12
    assert run.benchmark_params["run_preset"] == "quick"
    # quick = speed + quality + validation -> recovery/context skipped
    assert "recovery" in run.benchmark_params["skip_phases"]
    assert "context" in run.benchmark_params["skip_phases"]
    assert "speed" not in run.benchmark_params["skip_phases"]
    assert "quality" not in run.benchmark_params["skip_phases"]


def test_empty_cli_lists_do_not_override_preset_stages(monkeypatch):
    # The CLI always sends selected_stages=[], context_strategy_modes=[] etc.
    # Those must mean "unset", not "no stages / no strategies".
    run = _prepared_run(
        monkeypatch,
        {
            "run_preset": "quick",
            "selected_stages": [],
            "context_strategy_modes": [],
            "final_sandbox_preset_ids": [],
            "final_sandbox_custom_prompts": [],
        },
    )
    assert run.selected_stages == ["speed", "quality", "validation"]
    assert run.context_strategy_modes == ["winner"]
    assert set(run.benchmark_params["skip_phases"]) == {"recovery", "context"}


def test_standard_preset_keeps_its_context_strategies(monkeypatch):
    run = _prepared_run(
        monkeypatch,
        {"run_preset": "standard", "context_strategy_modes": []},
    )
    assert run.context_strategy_modes == ["winner", "model_first_kv_cpu"]
    assert run.selected_stages == ["speed", "quality", "recovery", "validation"]


def test_prepare_run_custom_stage_list_is_honoured(monkeypatch):
    run = _prepared_run(
        monkeypatch,
        {
            "run_preset": "custom",
            "selected_stages": ["speed", "quality"],
            "max_speed_probes": 9,
            "benchmark_repetitions": 2,
        },
    )
    skips = set(run.benchmark_params["skip_phases"])
    assert skips == {"recovery", "validation", "context"}
    assert run.selected_stages == ["speed", "quality"]
    assert run.max_speed_probes == 9
    assert run.benchmark_repetitions_override == 2


def test_prepare_run_floor_and_sandbox_settings_persist(monkeypatch):
    run = _prepared_run(
        monkeypatch,
        {
            "run_preset": "quick",
            "min_generation_tok_s": 3.5,
            "enforce_min_generation_speed": False,
            "final_sandbox_mode": "custom",
            "final_sandbox_custom_prompts": [{"prompt": "hi"}],
        },
    )
    assert run.min_generation_tok_s == 3.5
    assert run.enforce_min_generation_speed is False
    assert run.final_sandbox_mode == "custom"
    assert run.final_sandbox_custom_prompts == [{"prompt": "hi"}]
