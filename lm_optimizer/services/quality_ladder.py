"""Quality ladder: context fallback for quality-failed runs.

When a run finishes with no quality-passing config (e.g. truncation at
max context), this workflow re-tests the winner-candidate runtime at
progressively LOWER contexts until quality passes at the standard
threshold. Ladder steps are fine-grained (not power-of-2 only) so the
kept context stays as close to max as possible.

Exactly ONE lowered-threshold attempt is allowed, on the air, on the
highest-context result only: if it passes, it is reported as an option;
if not, no further lowering happens. The USER then picks:
higher context + lower quality, or lower context + quality OK.
"""

from dataclasses import dataclass, field

DEFAULT_LADDER_TOP = 30720
DEFAULT_LADDER_MIN = 8192
DEFAULT_LADDER_STEP = 2048


def build_quality_ladder(
    max_ctx: int,
    min_ctx: int = DEFAULT_LADDER_MIN,
    step: int = DEFAULT_LADDER_STEP,
    model_limit: int | None = None,
    explicit: list[int] | None = None,
) -> list[int]:
    """Descending context rungs below max_ctx (max itself is already measured).

    Fine-grained steps (default 2048) deliberately include non-power-of-2
    values so the kept context lands as close to max as honesty allows.
    """
    if explicit is not None:
        rungs = [c for c in explicit if c > 0 and c < max_ctx]
        if model_limit:
            rungs = [c for c in rungs if c <= model_limit]
        return sorted(set(rungs), reverse=True)
    if step <= 0:
        raise ValueError("step must be positive")
    if min_ctx <= 0:
        raise ValueError("min_ctx must be positive")
    top = min(max_ctx - 1, model_limit) if model_limit else max_ctx - 1
    # Align down to the step grid, then walk down (fine, non-power-of-2).
    rung = (top // step) * step
    out: list[int] = []
    while rung >= min_ctx:
        out.append(rung)
        rung -= step
    return out


@dataclass
class RungOutcome:
    """Measured result of one ladder rung."""

    context: int
    status: str  # passed | quality_failed | failed | oom | ...
    quality_overall: float | None = None
    checks: str = ""
    gen_tok_s: float | None = None
    config_id: str = ""


@dataclass
class LadderVerdict:
    """Two-way choice for the user."""

    max_ctx_option: RungOutcome | None = None
    quality_ok_option: RungOutcome | None = None
    fallback_passed: bool = False
    fallback_message: str = ""
    rungs: list[RungOutcome] = field(default_factory=list)


def fallback_verdict(quality_overall: float | None, fallback_threshold: float) -> tuple[bool, str]:
    """Single lowered-threshold attempt on the highest-context result.

    Exactly one attempt, no re-runs, no further lowering afterwards.
    """
    if quality_overall is None:
        return False, "no quality evidence for fallback attempt"
    if quality_overall >= fallback_threshold:
        return True, (
            f"fallback PASSES at lowered threshold {fallback_threshold}: "
            f"quality {quality_overall:.3f}"
        )
    return False, (
        f"fallback FAILS at lowered threshold {fallback_threshold}: "
        f"quality {quality_overall:.3f} — no further lowering, stop"
    )


def pick_options(
    rungs: list[RungOutcome],
    source_max_ctx: RungOutcome | None = None,
    fallback_threshold: float = 0.90,
) -> LadderVerdict:
    """Build the user-facing two-way choice from measured rungs.

    Walk rungs high->low; the first rung with status 'passed' is the
    quality-OK option. The max-context option is the source result
    (already measured at max, below the standard threshold).
    """
    verdict = LadderVerdict(rungs=list(rungs), max_ctx_option=source_max_ctx)
    for rung in rungs:
        if rung.status == "passed":
            verdict.quality_ok_option = rung
            break
    if source_max_ctx is not None:
        passed, msg = fallback_verdict(source_max_ctx.quality_overall, fallback_threshold)
        verdict.fallback_passed = passed
        verdict.fallback_message = msg
    return verdict
