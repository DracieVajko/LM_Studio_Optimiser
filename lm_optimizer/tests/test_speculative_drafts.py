"""Draft discovery + recommended list (Task 4).

Draft models are discovered heuristically from the LOCAL model list only.
A recommended draft that is not downloaded locally is skipped with a
reason — never auto-downloaded.
"""

import asyncio

from lm_optimizer.domain.models import ModelIdentity
from lm_optimizer.services.speculative import (
    RECOMMENDED_DRAFTS,
    discover_drafts,
    find_local_draft,
)


def m(model_id: str) -> ModelIdentity:
    return ModelIdentity(id=model_id, name=model_id)


def test_discovers_suffixed_draft():
    assert discover_drafts([m("Qwen3-0.6B"), m("Qwen3-32B")]) != []


def test_missing_draft_returns_none():
    assert find_local_draft([m("Qwen3-32B")], "Qwen3-32B") is None


def test_large_model_is_not_a_draft():
    assert discover_drafts([m("Qwen3-32B")]) == []


def test_recommended_qwen_draft_listed_by_name_only():
    assert RECOMMENDED_DRAFTS["qwen"] == ["Qwen3-0.6B"]
    assert RECOMMENDED_DRAFTS["default"] == []


def test_find_local_draft_returns_present_recommended():
    assert find_local_draft([m("Qwen3-0.6B"), m("Qwen3-32B")], "Qwen3-32B") == "Qwen3-0.6B"


def test_find_local_draft_never_downloads():
    # Only local lookup: an absent recommended draft yields None, no fetch.
    assert find_local_draft([], "Qwen3-32B") is None


def test_ab_compare_skips_without_local_draft():
    from lm_optimizer.services.speculative import ab_compare

    res = asyncio.run(ab_compare(None, "Qwen3-32B", None, None))
    assert res == {"model": "Qwen3-32B", "draft": None, "ran": False, "reason": "no local draft"}
