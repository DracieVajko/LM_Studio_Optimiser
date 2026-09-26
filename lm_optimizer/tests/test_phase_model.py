"""Phase A/B model: phases, frozen contexts, adaptive budget classes.

TDD Batch 1: pure logic, no server.
"""

from lm_optimizer.domain.models import OptimizationPhase
from lm_optimizer.services.speed_probe import (
    SPEED_PROBE_MAX_TOKENS,
    SPEED_PROBE_PROMPT,
    budget_class_for,
    finalist_limit_for,
    probe_repetitions_for,
    quality_context_for,
    speed_context_for,
)


class TestPhaseEnum:
    def test_phases_exist(self):
        assert OptimizationPhase.SPEED.value == "speed"
        assert OptimizationPhase.QUALITY.value == "quality"
        assert OptimizationPhase.RECOVERY.value == "recovery"
        assert OptimizationPhase.FINAL_VALIDATION.value == "final_validation"
        assert OptimizationPhase.CONTEXT_OPTIONAL.value == "context_optional"


class TestSpeedContext:
    def test_default_small(self):
        assert speed_context_for(None, 131072) == 2048

    def test_user_cap_wins(self):
        assert speed_context_for(8192, 131072) == 8192

    def test_clamped_to_model_limit(self):
        assert speed_context_for(None, 1024) == 1024
        assert speed_context_for(100000, 32768) == 32768


class TestQualityContext:
    def test_default_final_context(self):
        assert quality_context_for(None, 131072, 2048) == 8192

    def test_user_cap_frozen(self):
        assert quality_context_for(32768, 131072, 32768) == 32768

    def test_small_model_limit(self):
        assert quality_context_for(None, 4096, 2048) == 4096

    def test_never_below_speed_context(self):
        assert quality_context_for(None, 1024, 1024) == 1024


class TestBudgetClasses:
    def test_classes(self):
        assert budget_class_for(72.1) == "FAST"
        assert budget_class_for(40.0) == "NORMAL"  # >40 is FAST
        assert budget_class_for(10.0) == "NORMAL"
        assert budget_class_for(3.0) == "SLOW"  # 3-10 SLOW
        assert budget_class_for(2.9) == "VERY_SLOW"
        assert budget_class_for(0.5) == "VERY_SLOW"

    def test_no_rejection_slow_model_continues(self):
        # 0.5 tok/s is VERY_SLOW budget, never a rejection.
        assert budget_class_for(0.5) == "VERY_SLOW"
        assert probe_repetitions_for("VERY_SLOW", None) == 1

    def test_probe_repetitions(self):
        assert probe_repetitions_for("FAST", None) == 3
        assert probe_repetitions_for("NORMAL", None) == 2
        assert probe_repetitions_for("SLOW", None) == 1
        assert probe_repetitions_for("VERY_SLOW", None) == 1

    def test_user_override_wins(self):
        assert probe_repetitions_for("VERY_SLOW", 3) == 3
        assert probe_repetitions_for("FAST", 1) == 1

    def test_finalist_limits(self):
        assert finalist_limit_for("FAST") == 5
        assert finalist_limit_for("NORMAL") == 4
        assert finalist_limit_for("SLOW") == 3
        assert finalist_limit_for("VERY_SLOW") == 2


class TestProbeShape:
    def test_probe_is_cheap(self):
        assert SPEED_PROBE_MAX_TOKENS <= 64
        assert len(SPEED_PROBE_PROMPT.split()) <= 20
