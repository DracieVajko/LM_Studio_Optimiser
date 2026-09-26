"""Final run classification + recommendation helpers (P1 UX).

Single place deciding SUCCESS / PARTIAL_SUCCESS / FAILED from tested configs,
so CLI, API and Web UI agree. Only PASSED configs may enter the winner set.
"""

from __future__ import annotations

from typing import Any

from lm_optimizer.domain.models import OptimizationRun, RunStatus


def _status_of(c: Any) -> str:
    st = getattr(c, "status", "")
    return getattr(st, "value", str(st))


def failure_breakdown(configs: list) -> dict:
    """Count failure classes for zero-pass / partial UX."""
    out = {
        "load_failures": 0,
        "oom": 0,
        "timeouts": 0,
        "quality_rejected": 0,
        "unsupported_parameters": 0,
        "other": 0,
    }
    for c in configs:
        st = _status_of(c)
        if st == "passed":
            continue
        err = (getattr(c, "error", "") or "").lower()
        if st == "oom" or "oom" in err or "out of memory" in err:
            out["oom"] += 1
        elif st == "timeout" or "timeout" in err:
            out["timeouts"] += 1
        elif (
            st == "quality_failed" or "quality" in err or "correctness" in err or "threshold" in err
        ):
            out["quality_rejected"] += 1
        elif (
            st == "incompatible"
            or "unsupported" in err
            or "unrecognized" in err
            or "not supported" in err
        ):
            out["unsupported_parameters"] += 1
        elif (
            st in ("load_failed", "verify_failed", "preheat_failed", "benchmark_failed")
            or "load" in err
            or "refus" in err
        ):
            out["load_failures"] += 1
        else:
            out["other"] += 1
    # Anything failed without a classified error counts as load failure.
    if out["other"] and not out["load_failures"]:
        out["load_failures"] = out.pop("other")
    else:
        out.pop("other", None)
    return out


def classify_run(configs: list) -> tuple[str, dict]:
    """Return (state, info) where state in SUCCESS/PARTIAL_SUCCESS/FAILED.

    - SUCCESS: at least one PASSED and zero non-passed.
    - PARTIAL_SUCCESS: at least one PASSED and at least one failed.
    - FAILED: zero PASSED.
    """
    passed = [c for c in configs if _status_of(c) == "passed"]
    failed = [c for c in configs if _status_of(c) != "passed"]
    fb = failure_breakdown(configs)
    if not passed:
        return "FAILED", {
            "successful": 0,
            "failed": len(failed),
            "failure_classes": fb,
        }
    if failed:
        return "PARTIAL_SUCCESS", {
            "successful": len(passed),
            "failed": len(failed),
            "failure_classes": fb,
        }
    return "SUCCESS", {
        "successful": len(passed),
        "failed": 0,
        "failure_classes": fb,
    }


def final_status_for_run(run: OptimizationRun) -> RunStatus:
    """Map classification to persisted RunStatus."""
    state, _ = classify_run(list(run.configurations))
    if state == "SUCCESS":
        return RunStatus.SUCCESS
    if state == "PARTIAL_SUCCESS":
        return RunStatus.PARTIAL_SUCCESS
    return RunStatus.FAILED


def best_passed(configs: list) -> Any | None:
    """Highest-score eligible config; never a failed or unscored one."""
    eligible = [
        c for c in configs if _status_of(c) == "passed" and getattr(c, "score", None) is not None
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda c: getattr(c, "score", 0.0))


def final_recommendation(configs: list, run: Any | None) -> Any | None:
    """Architectural final winner (Phase A/B semantics).

    The run's validated best (best_config_id: QUALITY-safe, usually FINAL
    VALIDATION median) wins whenever it is still passed. Fallback is the
    highest-score PASSED config WITH quality evidence — a scoreless speed
    probe can never become FINAL RECOMMENDATION. Failed/unvalidated raws
    stay report-only (raw-fastest transparency sections).
    """
    if run is not None:
        best = run.get_best_config() if hasattr(run, "get_best_config") else None
        if best is not None and _status_of(best) == "passed":
            return best
    evidenced = [
        c
        for c in configs
        if _status_of(c) == "passed"
        and getattr(c, "score", None) is not None
        and getattr(c, "quality_score", None) is not None
    ]
    if not evidenced:
        return None
    return max(evidenced, key=lambda c: c.score)


def speed_range(configs: list) -> tuple[Any, Any] | None:
    """(fastest, slowest) eligible configs; None when nothing eligible."""
    eligible = [
        c for c in configs if _status_of(c) == "passed" and getattr(c, "score", None) is not None
    ]
    if not eligible:
        return None
    speeds = [(c.get_avg_generation_tok_s(), c) for c in eligible]
    return max(speeds, key=lambda t: t[0])[1], min(speeds, key=lambda t: t[0])[1]


def alternatives(configs: list, best: Any | None) -> dict:
    """Best Speed / Best Context / Best Quality / Best Memory among eligible."""
    passed = [
        c for c in configs if _status_of(c) == "passed" and getattr(c, "score", None) is not None
    ]
    if not passed:
        return {}
    out: dict = {}

    def quality_of(c: Any) -> float:
        q = getattr(c, "quality_score", None)
        return float(getattr(q, "overall", 0.0) or 0.0) if q else 0.0

    out["best_speed"] = max(passed, key=lambda c: c.get_avg_generation_tok_s())
    out["best_context"] = max(passed, key=lambda c: getattr(c, "context_length", 0))
    with_q = [c for c in passed if getattr(c, "quality_score", None) is not None]
    out["best_quality"] = max(with_q, key=quality_of) if with_q else out["best_speed"]
    # Best memory = lowest VRAM among configs within 90% of top speed (else lowest VRAM).
    top_speed = out["best_speed"].get_avg_generation_tok_s() or 0
    near_top = [c for c in passed if c.get_avg_generation_tok_s() >= 0.9 * top_speed]
    pool = near_top or passed
    out["best_memory"] = min(pool, key=lambda c: getattr(c, "peak_vram_gb", 0) or 0)
    # Avoid duplicating best under every crown when run is tiny; keep dict as-is.
    _ = best
    return out


def why_text(best: Any, baseline_metrics: dict | None, profile: str) -> str:
    """Data-driven one-paragraph explanation for the winner."""
    parts = [f"profile={profile}"]
    try:
        parts.append(f"gen={best.get_avg_generation_tok_s():.1f} tok/s")
    except Exception:
        pass
    try:
        parts.append(f"ctx={best.context_length}")
    except Exception:
        pass
    bd = getattr(best, "score_breakdown", None)
    if bd:
        top = sorted(bd.items(), key=lambda kv: kv[1], reverse=True)[:2]
        parts.append("strengths=" + ",".join(f"{k}:{v:.2f}" for k, v in top))
    if baseline_metrics and baseline_metrics.get("generation_tok_s"):
        try:
            base = baseline_metrics["generation_tok_s"]
            cur = best.get_avg_generation_tok_s()
            pct = (cur - base) / base * 100 if base else 0
            parts.append(f"vs-baseline={pct:+.1f}%")
        except Exception:
            pass
    q = getattr(best, "quality_score", None)
    if q is not None:
        try:
            parts.append(f"correctness={q.overall:.3f}")
        except Exception:
            pass
    return "; ".join(parts)


def failure_reason_text(run: OptimizationRun) -> str:
    """Human-readable reason for FAILED runs."""
    _, info = classify_run(list(run.configurations))
    fb = info.get("failure_classes", {})
    bits = [f"{k}={v}" for k, v in fb.items() if v]
    base = "; ".join(bits) if bits else "no configurations completed"
    if run.error:
        base += f"; error={run.error[:200]}"
    return base
