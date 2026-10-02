"""Experimental tail opt-ins (Task 3).

RoPE / CPU-MoE / speculative probes are opt-in, default OFF, and run only
as a tail after the FINAL winner — never inside the main sweep. Any opt-in
marks the run `is_experimental = True` with a reason string.
"""

from lm_optimizer.domain.models import ModelIdentity, OptimizationRun
from lm_optimizer.services.optimizer import (
    apply_experimental_mark,
    parse_experimental_flags,
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
