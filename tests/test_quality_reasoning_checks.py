"""Quality checks for reasoning sweeps + no silent full marks (2026-10-26).

Regression: reasoning variants (``medium_reasoning_off`` ... ``_xhigh``) were
unknown to the evaluator, so every level aggregated to ``overall=1.0`` with
``0/0 checks`` - a false pass. Unknown tests now score 0.0, and reasoning
levels reuse the base prompt's criteria.
"""

from __future__ import annotations

import pytest

from lm_optimizer.services.quality import QualityEvaluator

LEVELS = ("off", "low", "medium", "high", "xhigh")
GOOD = (
    "The sequence is 2, 6, 12, 20, 30, 42, 56. Look at the differences: 4, 6, "
    "8, 10, 12, 14, so each step adds two more than the previous one. The "
    "pattern continues with a difference of 16, which gives 56 + 16 = 72. "
    "Therefore the next number in the sequence is 72. This step-by-step "
    "reasoning shows the add pattern clearly and checks the arithmetic. "
    "Since the second differences are constant at two, this is a quadratic "
    "sequence and the rule is safe to extend one more time for the answer."
)
BAD = "I do not know."


def test_reasoning_sweep_variant_is_scored():
    ev = QualityEvaluator()
    q = ev.evaluate("m", "medium_reasoning_xhigh", GOOD)
    assert q.confident is True
    assert q.checks_total and q.checks_total > 0
    assert q.checks_passed == q.checks_total
    assert q.overall >= 0.97  # project quality threshold


def test_every_reasoning_level_is_known():
    ev = QualityEvaluator()
    for level in LEVELS:
        q = ev.evaluate("m", f"medium_reasoning_{level}", GOOD)
        assert q.confident is True, level
        assert (q.checks_total or 0) > 0, level


def test_bad_reasoning_answer_scores_lower_than_good():
    ev = QualityEvaluator()
    good = ev.evaluate("m", "medium_reasoning_off", GOOD)
    bad = ev.evaluate("m", "medium_reasoning_off", BAD)
    assert bad.overall < good.overall
    assert bad.checks_passed < good.checks_passed


def test_unknown_test_is_not_silently_perfect():
    ev = QualityEvaluator()
    q = ev.evaluate("m", "mystery_test", "anything")
    assert q.confident is False
    assert q.overall == 0.0
    assert q.checks_total == 0


def test_aggregate_unknown_only_is_not_a_pass():
    ev = QualityEvaluator()
    agg = ev.aggregate_quality({"mystery_test": ev.evaluate("m", "mystery_test", "x")})
    assert agg.overall == 0.0
    assert agg.checks_total == 0
    assert agg.confident is False


def test_aggregate_reasoning_levels_reports_real_checks():
    ev = QualityEvaluator()
    scores = {
        f"medium_reasoning_{level}": ev.evaluate("m", f"medium_reasoning_{level}", GOOD)
        for level in LEVELS
    }
    agg = ev.aggregate_quality(scores)
    assert agg.checks_total and agg.checks_total > 0
    assert agg.checks_passed == agg.checks_total
    assert agg.overall >= 0.97
    assert agg.confident is True


@pytest.mark.parametrize("level", LEVELS)
def test_level_suffix_does_not_break_other_lookups(level):
    # Sanity: the stripped lookup must only affect known base names.
    ev = QualityEvaluator()
    assert ev._get_prompt_def(f"medium_reasoning_{level}") is not None
    assert ev._get_prompt_def(f"totally_unknown_{level}") is None
