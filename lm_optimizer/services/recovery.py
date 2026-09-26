"""Bounded causal quality recovery (mission §16).

Pure planning: which suspect to roll back, in which order, within budget.
The measured rollback loop lives in the optimizer (needs the live server).
One parameter per probe; pairwise only for tiny deltas; never exhaustive.
"""

from dataclasses import dataclass

RECOVERY_BUDGET_DEFAULT = 3
PAIRWISE_MAX_KEYS = 3


@dataclass
class RecoveryAttempt:
    """One rollback probe and its measured outcome."""

    suspect: str
    old_value: object
    new_value: object
    recheck_passed: bool | None = None
    quality_after: float | None = None


def plan_recovery(ranked_suspects: list[str], budget: int = RECOVERY_BUDGET_DEFAULT) -> list[str]:
    """Top suspects in rank order, capped by the bounded budget."""
    return list(ranked_suspects[: max(0, budget)])


def pairwise_allowed(delta: dict, max_keys: int = PAIRWISE_MAX_KEYS) -> bool:
    """Pairwise rollback only when the delta is tiny (never a subset explosion)."""
    return len(delta) <= max_keys


def _severity(name: str, entry: dict) -> tuple[int, float]:
    """Explicit failed-test ordering (never compares None numerically).

    Severity 0: invalid output or unavailable evidence (overall 0/None,
    zero or null checks) — recheck first. Severity 1: low quality
    (overall < 0.5). Severity 2: partial failures. Ties break by overall
    ascending (worst first); None overall sorts as -1.0 (unknown = severe).
    """
    overall = entry.get("overall")
    passed = entry.get("checks_passed")
    total = entry.get("checks_total")
    num = overall if isinstance(overall, (int, float)) else -1.0
    if overall is None or overall == 0 or passed is None or total is None or passed == 0:
        return (0, num)
    if num < 0.5:
        return (1, num)
    return (2, num)


def rank_failed_tests(quality_by_test: dict) -> list[str]:
    """Failed tests in recheck order; passing tests excluded, None-safe.

    Failed = fewer passed than total, or null checks (unavailable evidence
    must be rechecked, never compared). overall=None counts as failed.
    """
    def _is_failed(d: dict) -> bool:
        passed, total = d.get("checks_passed"), d.get("checks_total")
        if passed is None or total is None or d.get("overall") is None:
            return True  # unavailable evidence is rechecked, never compared
        if isinstance(passed, (int, float)) and isinstance(total, (int, float)):
            return passed < total
        return True

    failed = {n: (e or {}) for n, e in (quality_by_test or {}).items() if _is_failed(e or {})}
    return sorted(failed, key=lambda n: (_severity(n, failed[n]), n))
