"""Phase A speed probe: fixed small context, cheap workload, budget classes.

The probe is deliberately NOT the full 5-test suite: one short prompt,
~32-64 output tokens, deterministic low temperature. No quality evaluation.
Slow models are never rejected — budget classes only shrink cost.
"""

from lm_optimizer.domain.models import BenchmarkCase

# Dedicated cheap probe (mission §5): short, deterministic, reusable.
SPEED_PROBE_PROMPT = "Explain in one or two short sentences what a hash table is."
SPEED_PROBE_MAX_TOKENS = 64
SPEED_PROBE_TEMPERATURE = 0.1
SPEED_PROBE_NAME = "speed_probe"

DEFAULT_SPEED_CONTEXT = 2048
DEFAULT_QUALITY_CONTEXT = 8192


def speed_probe_case() -> BenchmarkCase:
    """The single cheap probe case (never the full suite)."""
    return BenchmarkCase(
        name=SPEED_PROBE_NAME,
        category="speed",
        prompt=SPEED_PROBE_PROMPT,
        max_tokens=SPEED_PROBE_MAX_TOKENS,
        temperature=SPEED_PROBE_TEMPERATURE,
    )


def speed_context_for(user_cap: int | None, model_limit: int | None) -> int:
    """Fixed Phase-A speed context: user cap wins, else small default.

    Clamped to the model limit so small models stay loadable.
    """
    limit = model_limit or DEFAULT_QUALITY_CONTEXT
    if user_cap:
        return min(int(user_cap), limit)
    return min(DEFAULT_SPEED_CONTEXT, limit)


def quality_context_for(
    user_cap: int | None, model_limit: int | None, speed_context: int
) -> int:
    """Frozen Phase-A quality context (NOT a sweep): user cap wins, else the
    final runtime default bounded by the model limit, never below speed ctx."""
    limit = model_limit or DEFAULT_QUALITY_CONTEXT
    if user_cap:
        return min(int(user_cap), limit)
    return max(int(speed_context), min(DEFAULT_QUALITY_CONTEXT, limit))


# Adaptive budget classes (mission §10). Budget only — NEVER rejection.
FAST = "FAST"
NORMAL = "NORMAL"
SLOW = "SLOW"
VERY_SLOW = "VERY_SLOW"


def budget_class_for(tok_s: float) -> str:
    """Classify measured speed into a cost budget (all classes continue)."""
    if tok_s > 40:
        return FAST
    if tok_s >= 10:
        return NORMAL
    if tok_s >= 3:
        return SLOW
    return VERY_SLOW


def probe_repetitions_for(budget_class: str, user_override: int | None) -> int:
    """Speed-probe repetitions: user override wins, else class default."""
    if user_override is not None:
        return max(1, int(user_override))
    return {FAST: 3, NORMAL: 2, SLOW: 1, VERY_SLOW: 1}[budget_class]


def finalist_limit_for(budget_class: str) -> int:
    """How many speed finalists reach expensive quality tests."""
    return {FAST: 5, NORMAL: 4, SLOW: 3, VERY_SLOW: 2}[budget_class]
