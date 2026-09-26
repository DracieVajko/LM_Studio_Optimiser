"""Standalone context-capacity sweep (geometric probe + bisect refinement).

Used by `lm-optimizer ctx`. Separates three concepts:

- capacity: maximum stable (largest PASS after refinement)
- performance-optimal: fastest tok/s among PASS points
- balanced recommended: largest PASS context retaining >=80% of peak speed
  (falls back to performance-optimal when nothing else qualifies).

The sweep never returns the last geometric checkpoint blindly: a
PASS/FAIL pair always triggers intermediate probes inside the interval.
Step 500/1000 controls refinement granularity.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

ProbeFn = Callable[[int], Awaitable[dict[str, Any]]]


def geometric_contexts(min_ctx: int, max_ctx: int) -> list[int]:
    """Doubling ladder clamped to [min_ctx, max_ctx], always incl. endpoints."""
    if min_ctx <= 0:
        min_ctx = 512
    if max_ctx < min_ctx:
        max_ctx = min_ctx
    out = []
    v = min_ctx
    while v < max_ctx:
        out.append(v)
        nxt = v * 2
        if nxt >= max_ctx:
            break
        v = nxt
    if not out or out[-1] != max_ctx:
        out.append(max_ctx)
    # Guarantee the first geometric step exists even for narrow ranges.
    if len(out) == 1 and min_ctx != max_ctx:
        out = [min_ctx, max_ctx]
    return sorted(set(out))


def _mid(a: int, b: int, step: int) -> int:
    mid = (a + b) // 2
    # Snap down to step granularity, keep strictly inside (a, b).
    snapped = (mid // step) * step
    if snapped <= a:
        snapped = a + step
    if snapped >= b:
        snapped = b - step
        snapped = (snapped // step) * step
    return snapped


async def refine_boundary(
    last_pass: int,
    first_fail: int,
    probe: ProbeFn,
    step: int = 1000,
    max_probes: int = 8,
) -> dict:
    """Bisect (last_pass, first_fail); returns probes + max stable + failed boundary.

    probes: list of {ctx, ok, tok_s, error} in probe order.
    """
    probes: list[dict] = []
    lo, hi = last_pass, first_fail
    # Bisection until the interval is at step granularity.
    while hi - lo > step and len(probes) < max_probes:
        cand = _mid(lo, hi, step)
        # Guard against non-progress (tiny intervals).
        if cand <= lo or cand >= hi:
            break
        res = await probe(cand)
        entry = {"ctx": cand, **res}
        probes.append(entry)
        if res.get("ok"):
            lo = cand
        else:
            hi = cand
    return {"probes": probes, "max_stable": lo, "failed_boundary": hi}


async def sweep(
    probe: ProbeFn,
    min_ctx: int = 2048,
    max_ctx: int = 65536,
    step: int = 1000,
    precheck=None,
) -> dict:
    """Full sweep: geometric phase then refinement. Returns report dict.

    `precheck(ctx)` optionally refuses escalation without loading (e.g. an
    `lms --estimate-only` refusal). A precheck refusal stops escalation but
    the interval is still refined with REAL empirical loads; precheck-only
    points are marked {"estimated": True} and never count as proof.
    """
    ladder = geometric_contexts(min_ctx, max_ctx)
    phase1: list[dict] = []
    last_pass: int | None = None
    first_fail: int | None = None
    for ctx in ladder:
        if precheck is not None and last_pass is not None:
            try:
                refusal = precheck(ctx)
            except Exception:
                refusal = None
            if refusal:
                phase1.append({"ctx": ctx, "ok": False, "tok_s": 0.0,
                               "error": str(refusal)[:160], "estimated": True})
                first_fail = ctx
                break
        res = await probe(ctx)
        phase1.append({"ctx": ctx, **res})
        if res.get("ok"):
            last_pass = ctx
        else:
            first_fail = ctx
            break
    refinement: list[dict] = []
    if last_pass is not None and first_fail is not None:
        r = await refine_boundary(last_pass, first_fail, probe, step=step)
        refinement = r["probes"]
        max_stable = r["max_stable"]
        failed_boundary = r["failed_boundary"]
    elif last_pass is not None:
        max_stable, failed_boundary = last_pass, None
    else:
        max_stable, failed_boundary = None, first_fail

    # Capacity vs recommendation split over ALL passing points.
    passing = [p for p in (phase1 + refinement) if p.get("ok")]
    perf_optimal = None
    balanced = None
    peak = 0.0
    if passing:
        perf_optimal = max(passing, key=lambda p: p.get("tok_s", 0) or 0)
        peak = perf_optimal.get("tok_s", 0) or 0.0
        floor = 0.8 * peak
        eligible = [p for p in passing if (p.get("tok_s", 0) or 0) >= floor]
        balanced = max(eligible, key=lambda p: p["ctx"]) if eligible else perf_optimal
    return {
        "geometric": phase1,
        "refinement": refinement,
        "maximum_stable_context": max_stable,
        "failed_boundary": failed_boundary,
        "performance_optimal": perf_optimal,
        "balanced_recommended": balanced,
        "peak_tok_s": peak,
    }


def format_report(report: dict, model: str) -> str:
    """Human-readable CLI report lines."""
    lines = [f"Context sweep: {model}", ""]
    for p in report["geometric"]:
        if p.get("ok"):
            mark, detail = "PASS", f"{p.get('tok_s', 0):.1f} tok/s"
        elif p.get("estimated"):
            mark, detail = "FAIL*", (p.get("error", "FAIL")[:60] + " [estimate, unloaded]")
        else:
            mark, detail = "FAIL", (p.get("error", "FAIL")[:60])
        lines.append(f"{p['ctx']:<7} {mark:<5} {detail}")
    if report["refinement"]:
        lines += ["", "Refinement:"]
        for p in report["refinement"]:
            mark = "PASS" if p.get("ok") else "FAIL"
            detail = (
                f"{p.get('tok_s', 0):.1f} tok/s" if p.get("ok") else (p.get("error", "FAIL")[:60])
            )
            lines.append(f"{p['ctx']:<7} {mark:<5} {detail}")
    lines += ["", f"Maximum stable context: {report['maximum_stable_context']}"]
    if report["failed_boundary"]:
        lines.append(f"Failed boundary: {report['failed_boundary']}")
    perf = report["performance_optimal"]
    bal = report["balanced_recommended"]
    if perf:
        lines.append(
            f"Performance-optimal context: {perf['ctx']} ({perf.get('tok_s', 0):.1f} tok/s)"
        )
    if bal:
        lines.append(f"Balanced recommended: {bal['ctx']}")
    return "\n".join(lines)
