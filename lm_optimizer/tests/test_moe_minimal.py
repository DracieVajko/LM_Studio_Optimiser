"""MoE minimal gated probes (Task 2).

Expert probes run only for MoE models with a verified channel, capped at 3
values derived from the anchor (default, default//2, default//4, deduped,
anchor excluded). n-cpu-moe (num_cpu_expert_layers_ratio) stays manual:
no verified REST/CLI channel exists, so it is never auto-probed.
"""

from unittest.mock import MagicMock

from lm_optimizer.domain.models import LoadConfiguration, ModelIdentity
from lm_optimizer.services.lm_studio import LMStudioCapabilities
from lm_optimizer.services.search_space import SearchSpaceGenerator
from lm_optimizer.services.speed_search import SpeedCaps, build_speed_plan


def anchor(num_experts=None) -> LoadConfiguration:
    return LoadConfiguration(
        context_length=9999,  # must be overridden by the frozen speed context
        flash_attention=True,
        offload_kv_cache_to_gpu=True,
        eval_batch_size=512,
        physical_batch_size=512,
        parallel=4,
        context_checkpoints=32,
        num_experts=num_experts,
    )


def caps(**kw) -> SpeedCaps:
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


def test_non_moe_probes_no_experts():
    probes, _ = build_speed_plan(anchor(), 2048, caps(is_moe=False))
    assert all(p.config.num_experts is None for p in probes)


def _expert_variations(probes, anchor_value):
    """Only probes whose varied lever is the expert count (anchor excluded)."""
    return [
        p
        for p in probes
        if p.reason.startswith("MoE expert count")
        and p.config.num_experts not in (None, anchor_value)
    ]


def test_moe_max_three_expert_probes():
    # NOTE: scoped to the S1 expert variation set (probes whose lever is the
    # expert count). The plan's verbatim `is not None` count cannot be used:
    # _clone correctly propagates the anchor's num_experts=8 to every probe
    # (S0 must equal the anchor), so a naive count yields 20. Intent
    # preserved: at most 3 expert variations per the spec interface.
    probes, _ = build_speed_plan(anchor(num_experts=8), 2048, caps(is_moe=True))
    assert len(_expert_variations(probes, 8)) <= 3


def test_moe_expert_set_derived_from_anchor():
    """Expert values derive from the anchor, not a hardcoded (2, 4, 8)."""
    probes, _ = build_speed_plan(anchor(num_experts=16), 2048, caps(is_moe=True))
    experts = {p.config.num_experts for p in _expert_variations(probes, 16)}
    assert experts == {8, 4}  # (16, 8, 4) minus anchor


def test_moe_expert_probes_exclude_anchor():
    probes, _ = build_speed_plan(anchor(num_experts=8), 2048, caps(is_moe=True))
    experts = {p.config.num_experts for p in _expert_variations(probes, 8)}
    assert 8 not in experts
    assert experts == {4, 2}


def _search_space_expert_counts(num_experts):
    cap = LMStudioCapabilities()
    cap.supports_num_experts = True
    client = MagicMock()
    client.capabilities = cap
    gen = SearchSpaceGenerator(client)
    model = ModelIdentity(id="m", name="m", is_moe=True, num_experts=num_experts)
    return gen._generate_expert_counts(model, cap, {})


def test_search_space_expert_counts_capped_at_three():
    counts = _search_space_expert_counts(8)
    assert len(counts) <= 3
    assert sorted(counts) == [2, 4, 8]  # (default, default//2, default//4)


def test_search_space_non_moe_no_experts():
    cap = LMStudioCapabilities()
    cap.supports_num_experts = True
    client = MagicMock()
    client.capabilities = cap
    gen = SearchSpaceGenerator(client)
    model = ModelIdentity(id="m", name="m", is_moe=False, num_experts=None)
    assert gen._generate_expert_counts(model, cap, {}) == []
