"""Culprit metadata: which changed parameters can affect output semantics.

TDD Batch 1: explicit tiers, not hidden magic. Registry untouched.
"""

from lm_optimizer.domain.models import LoadConfiguration
from lm_optimizer.services.config_delta import config_delta, rank_delta
from lm_optimizer.services.semantic_risk import HIGH, LOW, MEDIUM, tier_of


class TestTiers:
    def test_execution_path_is_high(self):
        assert tier_of("speculative_draft_mtp") == HIGH
        assert tier_of("speculative_draft_simple") == HIGH
        assert tier_of("num_experts") == HIGH

    def test_memory_path_is_medium(self):
        assert tier_of("offload_kv_cache_to_gpu") == MEDIUM
        assert tier_of("gpu_ratio") == MEDIUM
        assert tier_of("flash_attention") == MEDIUM
        assert tier_of("context_checkpoints") == MEDIUM
        assert tier_of("unified_kv_cache") == MEDIUM

    def test_scheduling_is_low(self):
        assert tier_of("eval_batch_size") == LOW
        assert tier_of("physical_batch_size") == LOW
        assert tier_of("parallel") == LOW
        assert tier_of("try_mmap") == LOW
        assert tier_of("keep_model_in_memory") == LOW

    def test_unknown_is_low_not_hidden(self):
        assert tier_of("something_new") == LOW


class TestDelta:
    def test_delta_lists_changed_only(self):
        base = LoadConfiguration(context_length=2048, eval_batch_size=512)
        cand = LoadConfiguration(
            context_length=2048, eval_batch_size=1024, flash_attention=True
        )
        delta = config_delta(base, cand)
        assert set(delta) == {"eval_batch_size", "flash_attention"}
        assert delta["eval_batch_size"] == (512, 1024)
        assert delta["flash_attention"] == (None, True)

    def test_empty_delta(self):
        base = LoadConfiguration(context_length=2048)
        assert config_delta(base, LoadConfiguration(context_length=2048)) == {}

    def test_ranking_high_first(self):
        delta = {
            "eval_batch_size": (512, 1024),
            "speculative_draft_mtp": (None, True),
            "flash_attention": (False, True),
        }
        ranked = rank_delta(delta)
        assert ranked[0] == "speculative_draft_mtp"
        assert ranked[1] == "flash_attention"
        assert ranked[2] == "eval_batch_size"
