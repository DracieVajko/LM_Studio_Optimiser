"""Context capacity proof, floor separation and sweep failure handling (spec §5, §10)."""

from __future__ import annotations

import asyncio

import pytest

from lm_optimizer.services.context_explorer import (
    ContextPoint,
    MIN_DECODE_TPS,
    PointVerdict,
    build_report,
    classify_point,
    fill_ratios,
)
from lm_optimizer.services.context_sweep import pick_quality_rungs, sweep


# --- verdict classification -------------------------------------------------

def test_floor_scales_with_user_minimum():
    # Same 3 tok/s: fine at a 2 tok/s floor, too slow at 10.
    assert classify_point(3.0, min_tps=2.0) is PointVerdict.PASS
    assert classify_point(3.0, min_tps=10.0) is PointVerdict.FAIL_TOO_SLOW


def test_borderline_band_tracks_the_floor():
    assert classify_point(2.0, min_tps=2.0) is PointVerdict.BORDERLINE
    assert classify_point(0.2, min_tps=0.2) is PointVerdict.BORDERLINE


def test_missing_speed_telemetry_is_not_a_pass():
    assert classify_point(None, ok=True) is PointVerdict.FAIL_LOAD


@pytest.mark.parametrize(
    "error, verdict",
    [
        ("CUDA out of memory", PointVerdict.FAIL_OOM),
        ("timed out after 300s", PointVerdict.FAIL_TIMEOUT),
        ("unsupported context size", PointVerdict.FAIL_UNSUPPORTED),
        ("process crashed", PointVerdict.FAIL_CRASH),
        ("connection refused", PointVerdict.FAIL_LOAD),
    ],
)
def test_load_failures_stay_distinguishable(error, verdict):
    assert classify_point(None, ok=False, error=error) is verdict


def test_oom_is_never_reported_as_too_slow():
    v = classify_point(0.0, ok=False, error="CUDA out of memory")
    assert v is PointVerdict.FAIL_OOM
    assert v is not PointVerdict.FAIL_TOO_SLOW


# --- capacity vs floor ------------------------------------------------------

def test_max_stable_and_max_usable_are_separate_at_a_strict_floor():
    points = [
        ContextPoint(ctx=4096, verdict=PointVerdict.PASS, median_tok_s=20.0),
        ContextPoint(ctx=8192, verdict=PointVerdict.PASS, median_tok_s=10.0),
    ]
    report = build_report(points, min_tps=15.0)
    assert report.max_stable_context == 8192  # both pass the 1 tok/s default
    assert report.max_usable_context == 4096  # only 4096 meets 15 tok/s
    assert report.recommended_interactive_context == 4096


def test_loadable_capacity_survives_a_too_slow_point():
    points = [
        ContextPoint(ctx=4096, verdict=PointVerdict.PASS, median_tok_s=4.0),
        ContextPoint(ctx=16384, verdict=PointVerdict.FAIL_TOO_SLOW, median_tok_s=0.4),
        ContextPoint(ctx=32768, verdict=PointVerdict.FAIL_OOM, error="oom"),
    ]
    report = build_report(points)
    assert report.max_loadable_context == 16384  # it loaded
    assert report.max_stable_context == 4096
    assert report.max_usable_context == 4096


def test_retrieval_is_required_before_calling_a_context_usable():
    points = [
        ContextPoint(ctx=8192, verdict=PointVerdict.PASS, median_tok_s=5.0,
                     retrieval_passed=True, fill_ratio=0.9),
        ContextPoint(ctx=16384, verdict=PointVerdict.PASS, median_tok_s=4.0,
                     retrieval_passed=False, fill_ratio=0.9),
    ]
    report = build_report(points)
    assert report.max_usable_context == 16384
    assert report.max_retrieval_context == 8192
    assert report.max_quality_context == 8192


def test_allocation_only_probe_is_not_retrieval_proof():
    # A 0% fill load test passes speed but has no retrieval evidence.
    assert 0.0 in fill_ratios(detailed=False)
    assert 0.9 in fill_ratios(detailed=False)

    points = [ContextPoint(ctx=32768, verdict=PointVerdict.PASS, median_tok_s=6.0)]
    report = build_report(points)
    assert report.max_usable_context == 32768
    assert report.max_retrieval_context is None  # never invented


def test_default_floor_is_still_one_tok_s():
    assert MIN_DECODE_TPS == 1.0


# --- sweep behaviour --------------------------------------------------------

def _probe_until(limit_ctx: int, fail_error: str = "CUDA out of memory"):
    calls = []

    async def probe(ctx: int) -> dict:
        calls.append(ctx)
        if ctx >= limit_ctx:
            return {"ok": False, "tok_s": 0.0, "error": fail_error}
        return {"ok": True, "tok_s": 5.0, "error": ""}

    return probe, calls


def test_sweep_refines_the_pass_fail_boundary_with_real_probes():
    probe, calls = _probe_until(8192)
    report = asyncio.run(
        sweep(probe, min_ctx=4096, max_ctx=16384, step=512, min_tps=1.0)
    )
    assert report["maximum_stable_context"] is not None
    assert 4096 <= report["maximum_stable_context"] < 8192
    assert report["failed_boundary"] == 8192
    assert report["refinement"]  # boundary was refined, not just bracketed
    assert len(calls) == len(set(calls))  # no duplicate probes


def test_sweep_reports_oom_boundary_without_mislabelling_speed():
    probe, _ = _probe_until(8192, fail_error="CUDA out of memory")
    report = asyncio.run(
        sweep(probe, min_ctx=4096, max_ctx=8192, step=512, min_tps=1.0)
    )
    first_fail = report["geometric"][-1]
    assert first_fail["ok"] is False
    assert "memory" in first_fail["error"].lower()


def test_sweep_stops_escalating_on_precheck_estimate():
    calls = []

    async def probe(ctx: int) -> dict:
        calls.append(ctx)
        if ctx > 8192:
            return {"ok": False, "tok_s": 0.0, "error": "CUDA out of memory"}
        return {"ok": True, "tok_s": 5.0, "error": ""}

    def precheck(ctx: int) -> str | None:
        # Refuse anything above 8192 before loading it.
        return "estimate exceeds budget" if ctx > 8192 else None

    report = asyncio.run(
        sweep(
            probe,
            min_ctx=4096,
            max_ctx=32768,
            step=512,
            precheck=precheck,
            min_tps=1.0,
        )
    )
    estimated = [p for p in report["geometric"] if p.get("estimated")]
    assert estimated, "precheck refusal must be recorded as an estimate"
    # Estimates are never used as capacity proof or quality rungs.
    assert report["maximum_stable_context"] <= 8192
    assert all(ctx <= 8192 for ctx in pick_quality_rungs(report, 5))


def test_pick_quality_rungs_returns_only_real_passes():
    report = {
        "geometric": [
            {"ctx": 4096, "ok": True, "tok_s": 5.0},
            {"ctx": 16384, "ok": True, "tok_s": 2.0},
            {"ctx": 32768, "ok": True, "tok_s": 0.0, "estimated": True},
        ],
        "refinement": [],
    }
    assert pick_quality_rungs(report, 5) == [16384, 4096]
    assert pick_quality_rungs(report, 0) == []
