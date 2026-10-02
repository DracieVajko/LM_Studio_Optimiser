"""Manual mmap/keep memory duel (Task 5).

mmap / keep_model_in_memory stay MANUAL_ONLY: the tool only re-measures
the auto-best LoadConfiguration after the user toggles the GUI setting,
then issues a keep/revert verdict (>= +5% generation tok/s keeps).
"""

import pytest

from lm_optimizer.services.manual_memory_duel import (
    KEEP_PROMPT,
    MMAP_PROMPT,
    load_auto_best,
    verdict,
)


def test_keep_on_6pct_gain():
    assert verdict(10.0, 10.6, 500.0, 490.0)["decision"] == "keep"


def test_revert_on_noise():
    assert verdict(10.0, 10.2, 500.0, 510.0)["decision"] == "revert"


def test_model_mismatch_refuses():
    with pytest.raises(ValueError):
        load_auto_best("other-model-best.md", "this-model")


def test_keep_at_exact_threshold():
    assert verdict(10.0, 10.5, 500.0, 500.0)["decision"] == "keep"


def test_revert_below_threshold():
    assert verdict(10.0, 10.49, 500.0, 400.0)["decision"] == "revert"


def test_zero_baseline_reverts_without_crash():
    assert verdict(0.0, 10.0, 500.0, 490.0)["decision"] == "revert"


def test_verdict_carries_reason():
    for v in (
        verdict(10.0, 10.6, 500.0, 490.0),
        verdict(10.0, 10.2, 500.0, 510.0),
    ):
        assert isinstance(v["reason"], str) and v["reason"]


def test_prompts_describe_gui_toggle_and_enter():
    assert "mmap" in MMAP_PROMPT and "Enter" in MMAP_PROMPT
    assert "Keep" in KEEP_PROMPT and "Enter" in KEEP_PROMPT
