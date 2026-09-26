"""Bounded causal rollback planning (mission §16).

TDD: pure planning logic; the measured loop lives in the optimizer.
"""

from lm_optimizer.services.recovery import (
    RECOVERY_BUDGET_DEFAULT,
    RecoveryAttempt,
    pairwise_allowed,
    plan_recovery,
)


class TestPlan:
    def test_top_suspects_in_order_within_budget(self):
        ranked = ["speculative_draft_mtp", "flash_attention", "eval_batch_size", "parallel"]
        assert plan_recovery(ranked, budget=2) == ["speculative_draft_mtp", "flash_attention"]

    def test_default_budget(self):
        assert RECOVERY_BUDGET_DEFAULT == 3

    def test_empty_ranking(self):
        assert plan_recovery([], budget=3) == []


class TestPairwise:
    def test_small_delta_allowed(self):
        delta = {"a": (1, 2), "b": (3, 4)}
        assert pairwise_allowed(delta) is True

    def test_large_delta_refused(self):
        delta = {f"p{i}": (0, 1) for i in range(5)}
        assert pairwise_allowed(delta) is False


class TestRecord:
    def test_attempt_shape(self):
        a = RecoveryAttempt(
            suspect="speculative_draft_mtp",
            old_value=True,
            new_value=False,
            recheck_passed=True,
            quality_after=0.99,
        )
        assert a.suspect == "speculative_draft_mtp"
        assert a.recheck_passed is True
