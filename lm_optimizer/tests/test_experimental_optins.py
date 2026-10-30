"""Experimental tail opt-ins (Task 3).

RoPE / CPU-MoE / speculative probes are opt-in, default OFF, and run only
as a tail after the FINAL winner — never inside the main sweep. Any opt-in
marks the run `is_experimental = True` with a reason string.
"""

from lm_optimizer.domain.models import ModelIdentity, OptimizationRun
from lm_optimizer.services.optimizer import (
    apply_experimental_mark,
    parse_experimental_flags,
    resolve_experimental_optins,
)


def run_fixture() -> OptimizationRun:
    return OptimizationRun(model=ModelIdentity(id="m", name="m"))


def test_experimental_defaults_off():
    assert parse_experimental_flags(False, False, False) == {
        "enable_rope": False,
        "enable_cpu_moe": False,
        "enable_speculative": False,
    }


def test_experimental_marks_run():
    run = apply_experimental_mark(run_fixture(), {"enable_rope": True})
    assert run.is_experimental is True


def test_no_arg_defaults_off():
    assert parse_experimental_flags() == {
        "enable_rope": False,
        "enable_cpu_moe": False,
        "enable_speculative": False,
    }


def test_all_false_leaves_run_clean():
    run = apply_experimental_mark(
        run_fixture(), {"enable_rope": False, "enable_cpu_moe": False, "enable_speculative": False}
    )
    assert run.is_experimental is False
    assert run.experimental_reason is None


def test_each_flag_marks_run_with_reason():
    for flag in ("enable_rope", "enable_cpu_moe", "enable_speculative"):
        run = apply_experimental_mark(run_fixture(), {flag: True})
        assert run.is_experimental is True
        assert isinstance(run.experimental_reason, str) and run.experimental_reason


def test_resolve_all_unspecified_noninteractive_off():
    calls: list[str] = []
    out = resolve_experimental_optins(
        {"enable_rope": None, "enable_cpu_moe": None, "enable_speculative": None},
        ask=lambda q: calls.append(q) or True,
    )
    # ask=None is tested separately below; here ask must still default OFF
    # when the caller passes no answers (non-interactive wrapper passes None).
    out_none = resolve_experimental_optins(
        {"enable_rope": None, "enable_cpu_moe": None, "enable_speculative": None},
        ask=None,
    )
    assert out_none == {"enable_rope": False, "enable_cpu_moe": False, "enable_speculative": False}
    assert out == {"enable_rope": True, "enable_cpu_moe": True, "enable_speculative": True}
    assert len(calls) == 3


def test_resolve_each_positive_individually():
    for flag in ("enable_rope", "enable_cpu_moe", "enable_speculative"):
        req = {"enable_rope": None, "enable_cpu_moe": None, "enable_speculative": None}
        req[flag] = True
        out = resolve_experimental_optins(req, ask=None)
        assert out[flag] is True
        assert sum(1 for v in out.values() if v) == 1


def test_resolve_each_negative_individually():
    for flag in ("enable_rope", "enable_cpu_moe", "enable_speculative"):
        req = {"enable_rope": True, "enable_cpu_moe": True, "enable_speculative": True}
        req[flag] = False
        out = resolve_experimental_optins(req, ask=None)
        assert out[flag] is False


def test_resolve_explicit_off_with_other_on():
    out = resolve_experimental_optins(
        {"enable_rope": True, "enable_cpu_moe": False, "enable_speculative": None},
        ask=None,
    )
    assert out == {"enable_rope": True, "enable_cpu_moe": False, "enable_speculative": False}


def test_resolve_partial_interactive_prompts_only_unspecified():
    asked: list[str] = []
    out = resolve_experimental_optins(
        {"enable_rope": True, "enable_cpu_moe": None, "enable_speculative": None},
        ask=lambda q: asked.append(q) or False,
    )
    assert out["enable_rope"] is True
    assert out["enable_cpu_moe"] is False
    assert out["enable_speculative"] is False
    assert len(asked) == 2
    assert any("CPU-MoE" in q for q in asked)
    assert any("Speculative" in q for q in asked)
    assert not any("RoPE" in q for q in asked)


def test_resolve_explicit_false_counts_as_decision():
    calls: list[str] = []
    out = resolve_experimental_optins(
        {"enable_rope": False, "enable_cpu_moe": None, "enable_speculative": None},
        ask=lambda q: calls.append(q) or True,
    )
    assert out["enable_rope"] is False
    assert not any("RoPE" in q for q in calls)


class TestBoolLikeModelGuard:
    def test_true_false_rejected(self):
        import pytest

        from lm_optimizer.cli.main import _reject_bool_like_model_args

        for bad in ("False", "false", "FALSE", "True", " true "):
            with pytest.raises(SystemExit) as exc:
                _reject_bool_like_model_args(["some-model", bad])
            assert exc.value.code == 2

    def test_valid_names_pass(self):
        from lm_optimizer.cli.main import _reject_bool_like_model_args

        assert _reject_bool_like_model_args(None) is None
        assert _reject_bool_like_model_args([]) is None
        assert _reject_bool_like_model_args(["qwen3.8-9b", "Falsehood"]) is None


class TestYesNonInteractiveFlags:
    def test_auto_and_optimize_expose_flags(self):
        import typer

        from lm_optimizer.cli.main import app

        info = typer.main.get_command(app)
        for cmd in ("auto", "optimize"):
            params = [p.name for p in info.commands[cmd].params]
            assert "yes" in params, cmd
            assert "non_interactive" in params, cmd

    def test_wrapper_non_interactive_never_prompts(self):
        import sys
        from unittest.mock import patch

        import typer

        from lm_optimizer.cli import main as cli_main

        with (
            patch.object(sys.stdin, "isatty", return_value=True),
            patch.object(typer, "confirm") as confirm,
        ):
            out = cli_main._resolve_experimental_flags(None, None, None, non_interactive=True)
        assert out == {"enable_rope": False, "enable_cpu_moe": False, "enable_speculative": False}
        confirm.assert_not_called()

    def test_wrapper_yes_never_enables(self):
        from unittest.mock import patch

        import typer

        from lm_optimizer.cli import main as cli_main

        with patch.object(typer, "confirm") as confirm:
            out = cli_main._resolve_experimental_flags(None, None, None, non_interactive=True)
        assert out == {"enable_rope": False, "enable_cpu_moe": False, "enable_speculative": False}
        confirm.assert_not_called()


class TestExperimentalDetail:
    def test_detail_shape_requested_resolved_applied(self):
        from lm_optimizer.services.optimizer import build_experimental_detail

        detail = build_experimental_detail(
            {"enable_rope": True, "enable_cpu_moe": None, "enable_speculative": False},
            {"enable_rope": True, "enable_cpu_moe": False, "enable_speculative": False},
            {},
        )
        assert detail["requested"] == {
            "enable_rope": True, "enable_cpu_moe": None, "enable_speculative": False,
        }
        assert detail["resolved"] == {
            "enable_rope": True, "enable_cpu_moe": False, "enable_speculative": False,
        }
        assert detail["applied"]["enable_rope"] is True
        assert detail["exercised"]["enable_rope"] is False  # tail not yet run
        assert detail["skipped"] == {}

    def test_speculative_without_draft_skipped_never_exercised(self):
        from lm_optimizer.services.optimizer import build_experimental_detail

        detail = build_experimental_detail(
            {"enable_rope": False, "enable_cpu_moe": False, "enable_speculative": True},
            {"enable_rope": False, "enable_cpu_moe": False, "enable_speculative": True},
            {},
        )
        assert detail["applied"]["enable_speculative"] is False
        assert detail["exercised"]["enable_speculative"] is False
        assert detail["skipped"]["enable_speculative"] == "no usable draft model"

    def test_speculative_with_draft_stays_applied(self):
        from lm_optimizer.services.optimizer import build_experimental_detail

        detail = build_experimental_detail(
            {"enable_speculative": True},
            {"enable_rope": False, "enable_cpu_moe": False, "enable_speculative": True},
            {"speculative_draft_model": "tiny-draft"},
        )
        assert detail["applied"]["enable_speculative"] is True
        assert "enable_speculative" not in detail["skipped"]

    def test_detail_present_after_prepare_run(self, monkeypatch):
        import asyncio
        from unittest.mock import MagicMock

        from lm_optimizer.domain.models import HardwareInfo, ModelIdentity, OptimizationProfile
        from lm_optimizer.services import optimizer as _opt

        monkeypatch.setattr(_opt.hardware_repo, "save", lambda h: "hw1")
        monkeypatch.setattr(_opt.model_repo, "save", lambda m: "m1")
        saved = {}
        monkeypatch.setattr(_opt.run_repo, "save", lambda r: saved.setdefault("run", r) or "r1")

        opt = _opt.AdaptiveOptimizer.__new__(_opt.AdaptiveOptimizer)
        opt.benchmark = MagicMock()
        opt.benchmark.style = "balanced"
        space = MagicMock()
        space.to_dict.return_value = {}
        gen = MagicMock()
        gen.generate.return_value = space
        opt.search_generator = gen
        opt._apply_persisted_flash_advice = lambda: None

        async def _go():
            # _resolve_baseline must be awaitable returning None
            async def _base(model, baseline=None):
                return None

            opt._resolve_baseline = _base
            run, _, _ = await opt.prepare_run(
                ModelIdentity(id="m", name="m"),
                HardwareInfo(
                    os="T", cpu_name="C", cpu_cores_physical=1,
                    cpu_cores_logical=2, total_ram_gb=8, gpu_count=0,
                ),
                OptimizationProfile.BALANCED,
                advanced_settings={"enable_rope": True},
            )
            return run

        run = asyncio.run(_go())
        detail = (run.benchmark_params or {}).get("experimental_detail")
        assert detail is not None
        assert detail["resolved"]["enable_rope"] is True
        assert run.is_experimental is True

    def test_legacy_run_without_detail_loads(self):
        from lm_optimizer.services.reporting import _experimental_detail_rows
        from lm_optimizer.domain.models import (
            HardwareInfo, ModelIdentity, OptimizationProfile, OptimizationRun,
        )

        run = OptimizationRun(
            model=ModelIdentity(id="m", name="m"),
            hardware=HardwareInfo(
                os="T", cpu_name="C", cpu_cores_physical=1,
                cpu_cores_logical=2, total_ram_gb=8, gpu_count=0,
            ),
            profile=OptimizationProfile.BALANCED,
        )
        rows = _experimental_detail_rows(run)
        assert any("legacy" in r.lower() or "no experimental detail" in r.lower() for r in rows)
