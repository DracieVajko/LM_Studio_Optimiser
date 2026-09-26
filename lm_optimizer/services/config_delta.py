"""Configuration deltas: what changed vs baseline / last-known-good (mission §14)."""

from lm_optimizer.domain.models import LoadConfiguration
from lm_optimizer.services.semantic_risk import HIGH, LOW, MEDIUM, tier_of

_TIER_ORDER = {HIGH: 0, MEDIUM: 1, LOW: 2}


def config_delta(
    base: LoadConfiguration, candidate: LoadConfiguration
) -> dict[str, tuple[object, object]]:
    """Changed load keys only: {param: (old_value, new_value)}."""
    delta: dict[str, tuple[object, object]] = {}
    for key, new_value in candidate.to_dict().items():
        old_value = getattr(base, key, None)
        if old_value != new_value:
            delta[key] = (old_value, new_value)
    return delta


def rank_delta(delta: dict[str, tuple[object, object]]) -> list[str]:
    """Suspect order: HIGH tier first, then MEDIUM, then LOW; stable by name."""
    return sorted(delta, key=lambda k: (_TIER_ORDER[tier_of(k)], k))
