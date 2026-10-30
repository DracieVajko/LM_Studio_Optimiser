"""Task 6: winner/context/preset invariant.

Speed 4096 / quality 32768 winner must export nested==outer==preset==applied
32768 everywhere. Validation copy must use the outer context.
"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from lm_optimizer.domain.models import (
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
    QualityScore,
)


def _qs(overall=0.99):
    return QualityScore(
        overall=overall, task_completion=1.0, factual_consistency=1.0,
        format_compliance=1.0, coding_correctness=1.0, no_truncation=1.0,
        no_malformed=1.0, checks_passed=6, checks_total=6,
    )


def _best(outer_ctx=32768, nested_ctx=32768, score=0.9):
    return ConfigurationResult(
        config=LoadConfiguration(context_length=nested_ctx, flash_attention=True),
        context_length=outer_ctx,
        status=ConfigurationStatus.PASSED,
        metrics=[],
        quality_score=_qs(),
        score=score,
        tested_at=datetime.now(),
        generation={"stage": "quality"},
    )


def _optimizer_with_best(best):
    from lm_optimizer.services import optimizer as _opt
    from lm_optimizer.domain.models import (
        HardwareInfo, ModelIdentity, OptimizationProfile, OptimizationRun,
    )

    opt = _opt.AdaptiveOptimizer.__new__(_opt.AdaptiveOptimizer)
    run = OptimizationRun(
        model=ModelIdentity(id="m", name="m"),
        hardware=HardwareInfo(
            os="T", cpu_name="C", cpu_cores_physical=1,
            cpu_cores_logical=2, total_ram_gb=8, gpu_count=0,
        ),
        profile=OptimizationProfile.BALANCED,
    )
    run.validation_repetitions = 2
    run.advanced_settings = {}
    run.benchmark_params = {}
    state = MagicMock()
    state.best_config = best
    state.tested_configs = []
    state.run = run
    state.should_cancel = False
    state.planned_total = 0
    opt.state = state
    return opt, _opt


class TestValidationCopyUsesOuterContext:
    def test_divergent_pair_validates_at_outer_ctx(self, monkeypatch):
        opt, _opt = _optimizer_with_best(_best(outer_ctx=32768, nested_ctx=4096))
        seen = {}
        passed = _best(outer_ctx=32768, nested_ctx=32768)

        async def fake_test_config(vcfg, ctx, **kw):
            seen["nested"] = vcfg.context_length
            seen["outer"] = ctx
            return passed

        monkeypatch.setattr(opt, "_test_config", fake_test_config)
        monkeypatch.setattr(opt, "_pause_point", AsyncMock())
        monkeypatch.setattr(opt, "_tag_phase", lambda *a, **k: None)
        monkeypatch.setattr(opt, "_median_result", lambda vals: vals[0])
        monkeypatch.setattr(opt, "_compute_pareto_frontier", lambda cfgs: [])
        monkeypatch.setattr(opt, "_candidate_key", _opt.AdaptiveOptimizer._candidate_key)
        monkeypatch.setattr(opt, "_checkpoint", lambda *a, **k: None)
        monkeypatch.setattr(_opt.run_repo, "save", lambda r: "r1")

        asyncio.run(opt._phase_final_validation())
        assert seen["outer"] == 32768
        assert seen["nested"] == 32768

    def test_winner_preset_roundtrip_32768(self):
        best = _best(outer_ctx=32768, nested_ctx=32768)
        preset_config = dict(best.config.to_dict())
        assert preset_config["context_length"] == 32768
        applied = LoadConfiguration(**preset_config)
        assert applied.context_length == 32768
        assert best.context_length == 32768

    def test_failed_quality_never_winner(self):
        from lm_optimizer.services.run_summary import best_passed

        good = _best(outer_ctx=4096, nested_ctx=4096, score=0.4)
        bad = _best(outer_ctx=32768, nested_ctx=32768, score=None)
        bad.status = ConfigurationStatus.QUALITY_FAILED
        bad.score = None
        assert best_passed([good, bad]) is good

    def test_skip_quality_never_validated_winner(self):
        from lm_optimizer.services.optimizer import phase_skips_of
        from lm_optimizer.services.run_summary import final_recommendation

        assert "quality" in phase_skips_of({"skip_phases": ["quality"]})
        # Scoreless speed probe alone can never become the final recommendation.
        probe = _best(outer_ctx=4096, nested_ctx=4096, score=None)
        probe.quality_score = None
        assert final_recommendation([probe], None) is None
