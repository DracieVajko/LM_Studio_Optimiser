"""Staged hierarchical speed search + multi-finalist frontier (mission §7-§9).

TDD Batch 1: pure plan/frontier logic, no server.
"""

from lm_optimizer.domain.models import (
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
)
from lm_optimizer.services.speed_search import (
    SpeedCaps,
    build_interaction_probes,
    build_speed_plan,
    select_frontier,
)


def _anchor() -> LoadConfiguration:
    return LoadConfiguration(
        context_length=9999,  # must be overridden by the frozen speed context
        flash_attention=True,
        offload_kv_cache_to_gpu=True,
        eval_batch_size=512,
        physical_batch_size=512,
        parallel=4,
        context_checkpoints=32,
    )


def _caps(**kw) -> SpeedCaps:
    base = dict(
        rest_keys={
            "context_length",
            "flash_attention",
            "offload_kv_cache_to_gpu",
            "eval_batch_size",
            "physical_batch_size",
            "parallel",
            "context_checkpoints",
            "num_experts",
        },
        gpu_via_cli=True,
        is_moe=False,
        has_draft_model=False,
        flash_required=False,
    )
    base.update(kw)
    return SpeedCaps(**base)


def _probed_result(
    speed: float, status=ConfigurationStatus.PASSED, **cfg_kw
) -> ConfigurationResult:
    cfg = LoadConfiguration(context_length=2048, **cfg_kw)
    m = BenchmarkMetrics(
        test_name="speed_probe",
        category="speed",
        success=True,
        completion_tokens=40,
        generation_tok_s=speed,
    )
    return ConfigurationResult(
        config=cfg, context_length=2048, status=status, metrics=[m]
    )


class TestPlanShape:
    def test_all_probes_share_frozen_context(self):
        probes, _notes = build_speed_plan(_anchor(), 2048, _caps())
        assert probes, "expected staged probes"
        assert all(p.config.context_length == 2048 for p in probes)

    def test_no_cartesian_explosion(self):
        probes, _notes = build_speed_plan(_anchor(), 2048, _caps())
        assert len(probes) <= 25

    def test_stages_present(self):
        probes, _notes = build_speed_plan(_anchor(), 2048, _caps())
        stages = {p.stage for p in probes}
        assert {"S0", "S1", "S2", "S3"} <= stages

    def test_manual_only_never_probed(self):
        probes, notes = build_speed_plan(_anchor(), 2048, _caps())
        dicts = [p.config.to_dict() for p in probes]
        for key in ("try_mmap", "n_threads", "keep_model_in_memory"):
            vals = {d.get(key) for d in dicts}
            assert vals <= {None}, f"{key} must never be auto-probed"
        assert any("mmap" in n or "threads" in n for n in notes)

    def test_no_gpu_cli_no_ratio_probes(self):
        probes, _notes = build_speed_plan(_anchor(), 2048, _caps(gpu_via_cli=False))
        dicts = [p.config.to_dict() for p in probes]
        ratios = {d.get("gpu_ratio") for d in dicts}
        assert ratios <= {None}
        # KV placement via REST still probed.
        kv = {d.get("offload_kv_cache_to_gpu") for d in dicts}
        assert kv == {True, False}

    def test_flash_required_prunes_flash_off(self):
        probes, _notes = build_speed_plan(_anchor(), 2048, _caps(flash_required=True))
        dicts = [p.config.to_dict() for p in probes]
        assert all(d.get("flash_attention", True) is not False for d in dicts)

    def test_moe_gates_experts(self):
        probes_off, _ = build_speed_plan(_anchor(), 2048, _caps(is_moe=False))
        dicts = [p.config.to_dict() for p in probes_off]
        assert all(d.get("num_experts") is None for d in dicts)
        probes_on, _ = build_speed_plan(_anchor(), 2048, _caps(is_moe=True))
        assert any(p.config.num_experts is not None for p in probes_on)

    def test_no_draft_no_speculative(self):
        probes, _ = build_speed_plan(_anchor(), 2048, _caps(has_draft_model=False))
        dicts = [p.config.to_dict() for p in probes]
        assert all(d.get("speculative_draft_mtp") is None for d in dicts)


class TestInteractions:
    def test_combines_winners_only(self):
        a = LoadConfiguration(context_length=2048, offload_kv_cache_to_gpu=True)
        b = LoadConfiguration(context_length=2048, flash_attention=False)
        probes = build_interaction_probes([a, b], 2048)
        assert len(probes) <= 2
        assert all(p.config.context_length == 2048 for p in probes)


class TestFrontier:
    def test_band_and_contrarian(self):
        results = [
            _probed_result(70.0, offload_kv_cache_to_gpu=True),
            _probed_result(68.9, offload_kv_cache_to_gpu=True),
            _probed_result(67.8, offload_kv_cache_to_gpu=True),
            _probed_result(61.0, offload_kv_cache_to_gpu=False),
        ]
        front = select_frontier(results)
        assert front["winner"].get_avg_generation_tok_s() == 70.0
        # 5% of 70 = 66.5: A/B/C in band.
        assert len(front["finalists"]) == 3
        # D is out of band but most different -> contrarian.
        assert front["contrarian"] is not None
        assert front["contrarian"].get_avg_generation_tok_s() == 61.0

    def test_failed_never_enters(self):
        results = [
            _probed_result(99.0, status=ConfigurationStatus.LOAD_FAILED),
            _probed_result(50.0),
        ]
        front = select_frontier(results)
        assert front["winner"].get_avg_generation_tok_s() == 50.0
        assert all(
            r.status == ConfigurationStatus.PASSED
            for r in front["finalists"] + ([front["contrarian"]] if front["contrarian"] else [])
        )

    def test_single_result(self):
        front = select_frontier([_probed_result(0.5)])
        assert front["winner"].get_avg_generation_tok_s() == 0.5
        assert front["contrarian"] is None
