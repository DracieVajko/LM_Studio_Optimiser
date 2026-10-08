"""Tests for quality ladder logic (pure, no server needed)."""

from lm_optimizer.services.quality_ladder import (
    RungOutcome,
    build_quality_ladder,
    fallback_verdict,
    pick_options,
)


class TestBuildLadder:
    def test_descending_below_max(self):
        rungs = build_quality_ladder(32768)
        assert rungs[0] < 32768
        assert rungs == sorted(rungs, reverse=True)
        assert len(rungs) == len(set(rungs))

    def test_fine_steps_not_only_powers_of_two(self):
        rungs = build_quality_ladder(32768)
        non_pow2 = [c for c in rungs if c & (c - 1) != 0]
        assert len(non_pow2) > 0

    def test_top_is_close_to_max(self):
        rungs = build_quality_ladder(32768)
        assert rungs[0] == 30720

    def test_respects_min_and_model_limit(self):
        rungs = build_quality_ladder(32768, min_ctx=20000, model_limit=25000)
        assert all(20000 <= c <= 25000 for c in rungs)
        assert rungs[0] <= 25000

    def test_explicit_list(self):
        rungs = build_quality_ladder(32768, explicit=[30000, 25000, 32768, 10000])
        assert rungs == [30000, 25000, 10000]


class TestFallback:
    def test_single_pass(self):
        ok, msg = fallback_verdict(0.92, 0.90)
        assert ok is True
        assert "PASSES" in msg

    def test_single_fail_no_further_lowering(self):
        ok, msg = fallback_verdict(0.883, 0.90)
        assert ok is False
        assert "no further lowering" in msg

    def test_no_evidence(self):
        ok, _ = fallback_verdict(None, 0.90)
        assert ok is False


class TestPickOptions:
    def _rungs(self):
        return [
            RungOutcome(context=28672, status="quality_failed", quality_overall=0.91),
            RungOutcome(context=26624, status="passed", quality_overall=0.98, gen_tok_s=18.0),
            RungOutcome(context=24576, status="passed", quality_overall=0.99, gen_tok_s=19.0),
        ]

    def test_first_pass_wins(self):
        src = RungOutcome(context=32768, status="quality_failed", quality_overall=0.883)
        v = pick_options(self._rungs(), source_max_ctx=src, fallback_threshold=0.90)
        assert v.quality_ok_option is not None
        assert v.quality_ok_option.context == 26624
        assert v.max_ctx_option is not None
        assert v.max_ctx_option.context == 32768
        assert v.fallback_passed is False

    def test_no_pass_no_quality_option(self):
        rungs = [RungOutcome(context=28672, status="quality_failed", quality_overall=0.91)]
        src = RungOutcome(context=32768, status="quality_failed", quality_overall=0.92)
        v = pick_options(rungs, source_max_ctx=src, fallback_threshold=0.90)
        assert v.quality_ok_option is None
        assert v.fallback_passed is True
