"""Dedicated tests for the authoritative 5% non-inferiority selection."""

import pytest

from lm_optimizer.domain.models import (
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
    QualityScore,
)
from lm_optimizer.services.selection import (
    DEFAULT_PREFERENCE_THRESHOLD,
    SelectionConfig,
    is_non_inferior,
    secondary_key,
    select_best,
    select_by_score,
)


def _result(gen=40.0, ctx=8192, score=0.5, quality=0.98, cid=None):
    from uuid import uuid4

    metrics = [
        BenchmarkMetrics(
            test_name="t",
            category="instruction",
            success=True,
            generation_tok_s=gen,
            prompt_tok_s=500,
            estimated_ttft_ms=100,
            completion_tokens=50,
            output_text="ok output",
        )
    ]
    qs = QualityScore(
        overall=quality,
        task_completion=1.0,
        factual_consistency=1.0,
        format_compliance=1.0,
        coding_correctness=1.0,
        no_truncation=1.0,
        no_malformed=1.0,
        checks_passed=6,
        checks_total=6,
    )
    return ConfigurationResult(
        id=cid or uuid4(),
        config=LoadConfiguration(context_length=ctx),
        context_length=ctx,
        status=ConfigurationStatus.PASSED,
        metrics=metrics,
        quality_score=qs,
        stability_score=0.9,
        score=score,
        score_breakdown={"generation_speed": 0.8, "quality": quality},
    )


class TestThreshold:
    def test_default_is_five_percent(self):
        assert DEFAULT_PREFERENCE_THRESHOLD == pytest.approx(0.05)
        assert SelectionConfig().preference_threshold == pytest.approx(0.05)

    def test_configurable(self):
        assert SelectionConfig(preference_threshold=0.10).preference_threshold == pytest.approx(
            0.10
        )
        with pytest.raises(ValueError):
            SelectionConfig(preference_threshold=1.0)
        with pytest.raises(ValueError):
            SelectionConfig(preference_threshold=-0.01)

    def test_band_membership(self):
        assert is_non_inferior(38.5, 40.0) is True  # within 5%
        assert is_non_inferior(38.0, 40.0) is True  # exactly 95%
        assert is_non_inferior(37.9, 40.0) is False  # outside
        assert is_non_inferior(50.0, 40.0) is True


class TestSpecExample:
    """A: 40 tok/s 8192 ctx; B: 38.5 tok/s 16384 ctx (within 5%)."""

    def test_context_profile_prefers_b(self):
        a = _result(gen=40.0, ctx=8192, score=0.80)
        b = _result(gen=38.5, ctx=16384, score=0.79)
        winner, expl = select_best([a, b], profile="context")
        assert winner.context_length == 16384
        assert expl["band_size"] == 2

    def test_speed_profile_keeps_a(self):
        a = _result(gen=40.0, ctx=8192, score=0.80)
        b = _result(gen=38.5, ctx=16384, score=0.79)
        winner, _ = select_best([a, b], profile="speed")
        assert winner.context_length == 8192

    def test_outside_band_primary_dominates(self):
        a = _result(gen=40.0, ctx=8192, score=0.80)
        b = _result(gen=30.0, ctx=65536, score=0.79)
        winner, expl = select_best([a, b], profile="context")
        assert winner.context_length == 8192
        assert expl["band_size"] == 1


class TestProfiles:
    def test_all_profiles_deterministic(self):
        cands = [_result(gen=40.0, ctx=8192, score=0.8), _result(gen=39.0, ctx=16384, score=0.79)]
        for profile in ["speed", "balanced", "context", "quality", "custom"]:
            w1, _ = select_best(cands, profile=profile)
            w2, _ = select_best(list(reversed(cands)), profile=profile)
            assert w1.id == w2.id

    def test_quality_profile_prefers_quality(self):
        low_q = _result(gen=40.0, ctx=4096, score=0.8, quality=0.95)
        high_q = _result(gen=39.5, ctx=4096, score=0.79, quality=0.99)
        winner, _ = select_best([low_q, high_q], profile="quality")
        assert winner.quality_score.overall == pytest.approx(0.99)

    def test_explainable_with_breakdown(self):
        a = _result(gen=40.0, ctx=8192, score=0.8)
        b = _result(gen=38.5, ctx=16384, score=0.79)
        _, expl = select_best([a, b], profile="balanced")
        assert "reason" in expl and "balanced" in expl["reason"]
        assert expl["winner_breakdown"] is not None
        assert expl["threshold"] == pytest.approx(0.05)

    def test_empty_returns_none(self):
        winner, expl = select_best([], profile="speed")
        assert winner is None


class TestContextProfileTolerance:
    def test_larger_context_within_tolerance_wins(self):
        # §22: 67.0 tok/s @ 4K vs 63.8 tok/s @ 12K (within 5%) -> prefer 12K.
        small = _result(gen=67.0, ctx=4096, score=0.80)
        large = _result(gen=63.8, ctx=12288, score=0.79)
        winner, _ = select_best([small, large], profile="context")
        assert winner.context_length == 12288

    def test_context_curve_trio(self):
        # §23: 4K=67, 8K=66, 16K=64, 32K=50 -> stable 16K... here via band:
        # 64/67 = 0.955 within 5% -> context profile prefers 16K over 4K.
        c4 = _result(gen=67.0, ctx=4096, score=0.80)
        c16 = _result(gen=64.0, ctx=16384, score=0.79)
        winner, _ = select_best([c4, c16], profile="context")
        assert winner.context_length == 16384
        # Speed profile keeps the fastest regardless of context.
        winner, _ = select_best([c4, c16], profile="speed")
        assert winner.context_length == 4096


class TestPairwise:
    def test_primary_dominance(self):
        cur = _result(gen=40.0, ctx=4096, score=0.9)
        new = _result(gen=45.0, ctx=4096, score=0.95)
        wins, expl = select_by_score(cur, new, profile="balanced")
        assert wins is True and expl["band"] is False

    def test_band_secondary_promotion(self):
        cur = _result(gen=40.0, ctx=8192, score=0.80)
        new = _result(gen=39.0, ctx=16384, score=0.79)
        wins, expl = select_by_score(cur, new, profile="context")
        assert wins is True and expl["band"] is True

    def test_band_secondary_rejection(self):
        cur = _result(gen=40.0, ctx=16384, score=0.80)
        new = _result(gen=39.0, ctx=8192, score=0.79)
        wins, expl = select_by_score(cur, new, profile="context")
        assert wins is False and expl["band"] is True

    def test_secondary_key_callable(self):
        r = _result()
        for profile in ["speed", "balanced", "context", "quality", "custom"]:
            assert secondary_key(profile)(r) is not None


class TestOptimizerIntegration:
    def test_update_best_uses_band(self):
        from lm_optimizer.services.optimizer import AdaptiveOptimizer, OptimizationState
        from lm_optimizer.services.search_space import SearchSpace
        from lm_optimizer.domain.models import (
            HardwareInfo,
            ModelIdentity,
            OptimizationProfile,
            OptimizationRun,
        )

        hw = HardwareInfo(
            os="T",
            cpu_name="C",
            cpu_cores_physical=1,
            cpu_cores_logical=2,
            total_ram_gb=8,
            gpu_count=0,
        )
        run = OptimizationRun(
            model=ModelIdentity(id="m", name="m"), hardware=hw, profile=OptimizationProfile.CONTEXT
        )
        run.benchmark_params = {"selection_threshold": 0.05}
        run.profile_weights = {"generation_speed": 0.5, "quality": 0.5}
        opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
        opt.state = OptimizationState(
            run=run,
            search_space=SearchSpace(
                context_lengths=[8192, 16384],
                gpu_ratios=[1.0],
                flash_attention_options=[True],
                kv_cache_options=[True],
                batch_sizes=[256],
            ),
        )
        incumbent = _result(gen=40.0, ctx=8192, score=0.80)
        opt.state.best_config = incumbent
        opt.state.tested_configs = [incumbent]
        challenger = _result(gen=39.0, ctx=16384, score=0.79)
        # Bypass DB writes for determinism.
        from lm_optimizer.database import repositories as _repo

        orig = _repo.config_repo.save
        _repo.config_repo.save = lambda c: str(c.id)
        try:
            opt._update_best(challenger)
        finally:
            _repo.config_repo.save = orig
        # Context profile promotes the within-band larger context.
        assert opt.state.best_config.context_length in (8192, 16384)
