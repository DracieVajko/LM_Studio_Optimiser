"""--test / --test-only wiring tests (these were dead code until 2026-10-22)."""
import pytest
from unittest.mock import MagicMock

from lm_optimizer.services.benchmark import BenchmarkService


def _service(**kw):
    client = MagicMock()
    return BenchmarkService(client, **kw)


def test_test_names_filter_the_suite():
    svc = _service(test_names=["short_instruction", "coding_task"])
    cases = svc.create_cases_for_context(4096)
    names = {c.name for c in cases}
    assert names == {"short_instruction", "coding_task"}


def test_no_test_names_returns_full_suite():
    svc = _service()
    cases = svc.create_cases_for_context(4096)
    assert len(cases) > 2


def test_unknown_test_names_raise_instead_of_silently_running_all():
    svc = _service(test_names=["definitely-not-a-test"])
    with pytest.raises(ValueError):
        svc.create_cases_for_context(4096)


def test_mixed_known_and_unknown_keeps_known():
    svc = _service(test_names=["short_instruction", "nope"])
    cases = svc.create_cases_for_context(4096)
    assert {c.name for c in cases} == {"short_instruction"}


def test_test_names_are_case_insensitive():
    svc = _service(test_names=["SHORT_INSTRUCTION"])
    cases = svc.create_cases_for_context(4096)
    assert {c.name for c in cases} == {"short_instruction"}


def test_blank_names_are_ignored():
    svc = _service(test_names=["  ", ""])
    cases = svc.create_cases_for_context(4096)
    assert len(cases) > 2  # no filter applied


# --- preflight composition ---

def _pf(**kw):
    from lm_optimizer.cli import main as _cli

    base = dict(
        model="m",
        profile="balanced",
        style="chat",
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


def test_preflight_test_only_composes_skips():
    eff = _pf(test_only=True, test=["short_instruction"])
    assert {"speed", "recovery", "validation", "context"} <= set(eff["skip_phases"])
    assert eff["test_names"] == ["short_instruction"]


def test_preflight_test_names_without_test_only_keep_speed():
    eff = _pf(test=["short_instruction"])
    assert "speed" not in eff["skip_phases"]
    assert eff["test_names"] == ["short_instruction"]


def test_preflight_both_flags_coexist_with_quality_only():
    eff = _pf(test_only=True, test=["coding_task"], quality_only=True)
    assert {"speed", "recovery", "validation", "context"} <= set(eff["skip_phases"])
