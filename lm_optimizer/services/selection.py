"""Authoritative 5% non-inferiority selection.

Rule: let A be the primary-metric leader. Any candidate B with
primary(B) >= (1 - threshold) * primary(A) is non-inferior: the secondary
profile preference may select B. Outside the band the primary dominates,
unless profile weights explicitly justify another result (i.e. the weighted
score itself is the primary — see `select_by_score`).

Deterministic: ties broken by (score, context, config id). Every decision
returns an explanation dict with the score breakdown.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from lm_optimizer.domain.models import ConfigurationResult

DEFAULT_PREFERENCE_THRESHOLD = 0.05


class InvalidThresholdError(ValueError):
    """Raised when a preference threshold is outside [0, 1)."""

    def __init__(self, value: float) -> None:
        super().__init__(f"preference_threshold must be in [0, 1), got {value!r}")
        self.value = value


@dataclass(frozen=True)
class SelectionConfig:
    """Configurable preference threshold."""

    preference_threshold: float = DEFAULT_PREFERENCE_THRESHOLD

    def __post_init__(self) -> None:
        if not 0.0 <= self.preference_threshold < 1.0:
            raise InvalidThresholdError(self.preference_threshold)


def is_non_inferior(
    candidate_primary: float,
    best_primary: float,
    threshold: float = DEFAULT_PREFERENCE_THRESHOLD,
) -> bool:
    """True when candidate is within `threshold` of the leader."""
    if best_primary <= 0:
        return True
    if candidate_primary <= 0:
        return False
    return candidate_primary >= (1.0 - threshold) * best_primary


def _quality_of(c: ConfigurationResult) -> float:
    q = c.quality_score
    if q is None:
        return 0.0
    try:
        return float(q.overall or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _gen_of(c: ConfigurationResult) -> float:
    try:
        return float(c.get_avg_generation_tok_s())
    except (TypeError, ValueError):
        return 0.0


def _ctx_of(c: ConfigurationResult) -> int:
    try:
        return int(c.context_length or 0)
    except (TypeError, ValueError):
        return 0


def _score_of(c: ConfigurationResult) -> float:
    try:
        return float(c.score or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _cid(c: ConfigurationResult) -> str:
    return str(c.id)


SecondaryKey = Callable[[ConfigurationResult], tuple[float, float, float]]


def secondary_key(profile: str) -> SecondaryKey:
    """Profile-specific secondary preference (deterministic, documented).

    Speed: generation speed. Balanced: weighted score, then speed.
    Context: context length, then score. Quality: quality, then stability.
    Custom/unknown: weighted score.
    """
    p = (profile or "balanced").lower()

    def _stability(c: ConfigurationResult) -> float:
        try:
            return float(c.stability_score or 0.0)
        except (TypeError, ValueError):
            return 0.0

    if p == "speed":
        return lambda c: (_gen_of(c), _score_of(c), float(_ctx_of(c)))
    if p == "context":
        return lambda c: (float(_ctx_of(c)), _score_of(c), _gen_of(c))
    if p == "quality":
        return lambda c: (_quality_of(c), _stability(c), _score_of(c))
    # balanced / custom / unknown
    return lambda c: (_score_of(c), _gen_of(c), float(_ctx_of(c)))


def select_best(
    candidates: list[ConfigurationResult],
    primary: Callable[[ConfigurationResult], float] | None = None,
    profile: str = "balanced",
    config: SelectionConfig | None = None,
) -> tuple[ConfigurationResult | None, dict[str, object]]:
    """Select winner with the non-inferiority band.

    Returns (winner, explanation). Only candidates passed in are considered;
    callers must pre-filter to PASSED. Explanation contains primary_best,
    band_floor, band members, winner, reason and the winner's score_breakdown.
    """
    cfg = config or SelectionConfig()
    primary_fn = primary or _gen_of
    passed = list(candidates)
    if not passed:
        return None, {"reason": "no candidates", "profile": profile}

    # Deterministic order before any comparison (ids are unique).
    ordered = sorted(passed, key=lambda c: (-primary_fn(c), -_score_of(c), _cid(c)))
    leader = ordered[0]
    best_primary = primary_fn(leader)
    floor = (1.0 - cfg.preference_threshold) * best_primary if best_primary > 0 else 0.0
    band = [c for c in ordered if primary_fn(c) >= floor]
    key = secondary_key(profile)
    # Tuples (key, id) are unique, so max() == sorted()[-1].
    winner = max(band, key=lambda c: (key(c), _cid(c)))
    explanation: dict[str, object] = {
        "profile": profile,
        "threshold": cfg.preference_threshold,
        "primary_best": best_primary,
        "primary_best_id": _cid(leader),
        "band_floor": floor,
        "band_size": len(band),
        "band_ids": [_cid(c) for c in band],
        "winner_id": _cid(winner),
        "winner_primary": primary_fn(winner),
        "winner_score": _score_of(winner),
        "winner_breakdown": winner.score_breakdown,
        "reason": (
            f"leader primary={best_primary:.2f}; band floor={floor:.2f} "
            f"(threshold={cfg.preference_threshold}); {len(band)} non-inferior; "
            f"profile={profile} secondary selected {_cid(winner)}"
        ),
    }
    return winner, explanation


def select_by_score(
    current_best: ConfigurationResult,
    challenger: ConfigurationResult,
    profile: str = "balanced",
    config: SelectionConfig | None = None,
) -> tuple[bool, dict[str, object]]:
    """Pairwise version used by the optimizer hot loop.

    Returns (challenger_wins, explanation). Primary = weighted score (which
    already encodes explicit profile weights, so "weights justify another
    result" is honored). Within the band, the profile secondary decides.
    """
    cfg = config or SelectionConfig()
    cur_score = _score_of(current_best)
    new_score = _score_of(challenger)
    if new_score > cur_score:
        return True, {
            "reason": f"primary dominates: {new_score:.4f} > {cur_score:.4f}",
            "profile": profile,
            "band": False,
        }
    if is_non_inferior(new_score, cur_score, cfg.preference_threshold):
        key = secondary_key(profile)
        if key(challenger) > key(current_best):
            return True, {
                "reason": (
                    f"non-inferior band ({new_score:.4f} within "
                    f"{cfg.preference_threshold:.0%} of {cur_score:.4f}); "
                    f"profile={profile} secondary prefers challenger"
                ),
                "profile": profile,
                "band": True,
            }
        return False, {
            "reason": f"within band but secondary keeps incumbent (profile={profile})",
            "profile": profile,
            "band": True,
        }
    band_floor = (1.0 - cfg.preference_threshold) * cur_score
    return False, {
        "reason": f"outside band: {new_score:.4f} < {band_floor:.4f}",
        "profile": profile,
        "band": False,
    }
