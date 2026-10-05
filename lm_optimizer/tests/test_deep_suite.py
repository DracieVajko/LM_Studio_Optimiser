"""Deep-research suite definition tests (Task 1)."""

from lm_optimizer.benchmark.deep_suite import DEEP_CASES, deep_metrics
from lm_optimizer.domain.models import BenchmarkMetrics


def test_deep_suite_has_five_fixed_tasks():
    assert [c.name for c in DEEP_CASES] == ["long_context_recall", "multi_hop", "json_discipline", "coding_precision", "instruction_follow"]
    assert all(c.max_tokens >= 1024 for c in DEEP_CASES)


def test_deep_metrics_keys():
    m = BenchmarkMetrics(
        test_name="long_context_recall",
        category="recall",
        success=True,
        prompt_processing_ms=200.0,
        generation_ms=800.0,
        generation_tok_s=25.0,
        output_text="x",
        thinking_text="thought",
    )
    out = deep_metrics(m)
    assert out["recall_accuracy"] is not None
    assert out["thinking_chars"] == len("thought")
    assert out["elapsed_s"] == 1.0
    assert out["gen_tok_s"] == 25.0


def test_deep_metrics_recall_accuracy_counts_planted_facts():
    from lm_optimizer.benchmark.deep_suite import RECALL_FACTS

    m = BenchmarkMetrics(
        test_name="long_context_recall",
        category="recall",
        success=True,
        output_text=" ".join(RECALL_FACTS),
    )
    out = deep_metrics(m)
    assert out["recall_accuracy"] == 1.0
    m2 = BenchmarkMetrics(
        test_name="long_context_recall",
        category="recall",
        success=True,
        output_text="nothing relevant here",
    )
    out2 = deep_metrics(m2)
    assert out2["recall_accuracy"] == 0.0
