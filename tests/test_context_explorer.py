"""Context Explorer unit tests (P2, spec section 8).

Pure logic, no I/O, no LM Studio.
"""

from lm_optimizer.services.context_explorer import (
    BORDERLINE_HIGH,
    BORDERLINE_LOW,
    MIN_DECODE_TPS,
    NEEDLE_POSITIONS,
    ContextPoint,
    PointVerdict,
    build_filler,
    build_needles,
    build_report,
    classify_point,
    fill_ratios,
    format_report,
    geometric_contexts,
    needs_repeat,
    refine_boundary,
    score_retrieval,
)

# --- 8.3 geometric bracketing (1.5x, NOT power-of-two) ---

def test_geometric_uses_1_5_factor_not_powers_of_two():
    ladder = geometric_contexts(4096, 65536, factor=1.5, round_to=512)
    # 8192 / 16384 / 32768 must NOT appear (that is the power-of-two ladder).
    assert 8192 not in ladder
    assert 16384 not in ladder
    assert 32768 not in ladder
    # 1.5x growth from 4096 -> 6144 -> 9216 ...
    assert ladder[:4] == [4096, 6144, 9216, 13824]


def test_geometric_includes_endpoints():
    ladder = geometric_contexts(2048, 100000)
    assert ladder[0] == 2048
    assert ladder[-1] == 100000


def test_geometric_rounds_to_settable_resolution():
    ladder = geometric_contexts(4096, 65536, round_to=512)
    assert all(c % 512 == 0 for c in ladder)


def test_geometric_handles_degenerate_range():
    assert geometric_contexts(4096, 4096) == [4096]
    assert geometric_contexts(0, 0)[0] > 0


def test_refine_can_find_non_power_of_two_boundary():
    # A PASS at 65536 and a FAIL at 131072: refinement should land on
    # values like 76288 / 98304, not only powers of two.
    cands = refine_boundary(65536, 131072, step=512)
    assert cands
    assert all(65536 < c < 131072 for c in cands)
    assert not all(c & (c - 1) == 0 for c in cands)


def test_refine_stops_at_step_granularity():
    cands = refine_boundary(1000, 1024, step=512)
    assert cands == []


# --- 8.2 speed threshold with hysteresis ---

def test_classify_pass_borderline_too_slow():
    assert classify_point(2.0) is PointVerdict.PASS
    assert classify_point(BORDERLINE_HIGH) is PointVerdict.PASS
    assert classify_point(1.0) is PointVerdict.BORDERLINE
    assert classify_point(BORDERLINE_LOW) is PointVerdict.BORDERLINE
    assert classify_point(0.5) is PointVerdict.FAIL_TOO_SLOW


def test_too_slow_is_not_oom():
    verdict = classify_point(0.2, ok=False, error="timed out after 300s")
    assert verdict is PointVerdict.FAIL_TIMEOUT
    assert verdict is not PointVerdict.FAIL_OOM
    assert verdict is not PointVerdict.FAIL_TOO_SLOW


def test_oom_is_distinguishable():
    assert classify_point(None, ok=False, error="CUDA out of memory") is PointVerdict.FAIL_OOM


def test_borderline_points_are_repeated():
    assert needs_repeat(PointVerdict.BORDERLINE)
    assert not needs_repeat(PointVerdict.PASS)
    assert not needs_repeat(PointVerdict.FAIL_TOO_SLOW)


# --- 8.4 fill ratios ---

def test_fill_ratios_fast_and_detailed():
    assert fill_ratios(detailed=False) == (0.0, 0.5, 0.9)
    assert fill_ratios(detailed=True) == (0.0, 0.25, 0.5, 0.75, 0.9, 0.98)


# --- 8.5 deterministic filler ---

def test_filler_is_deterministic():
    a = build_filler(200, kind="prose", seed=42)
    b = build_filler(200, kind="prose", seed=42)
    assert a == b


def test_filler_seed_changes_content():
    a = build_filler(200, kind="prose", seed=1)
    b = build_filler(200, kind="prose", seed=2)
    assert a != b


def test_filler_respects_token_target():
    text = build_filler(120, kind="mixed", seed=42)
    words = len(text.split())
    assert 100 <= words <= 200


def test_filler_empty_target():
    assert build_filler(0) == ""


# --- 8.6 retrieval validation ---

def test_needles_placed_at_all_positions():
    filler = build_filler(300, kind="prose", seed=42)
    needles = build_needles(filler)
    assert len(needles) == len(NEEDLE_POSITIONS)
    assert needles[-1].prompt
    assert "VERDIGEN" in needles[-1].prompt


def test_needle_prompt_contains_all_positions():
    filler = build_filler(300, kind="prose", seed=42)
    needles = build_needles(filler)
    # Every needle statement is inserted, so the key appears N times.
    assert needles[-1].prompt.count("VERDIGEN-8412") == len(NEEDLE_POSITIONS)


def test_score_retrieval_exact_match():
    assert score_retrieval("The access code is VERDIGEN-8412.")
    assert score_retrieval("verdigen-8412")
    assert not score_retrieval("I do not know.")
    assert not score_retrieval("")


# --- 8.1 report assembly ---

def test_report_separates_capacity_from_usable():
    points = [
        ContextPoint(ctx=4096, verdict=PointVerdict.PASS, median_tok_s=2.0),
        ContextPoint(ctx=8192, verdict=PointVerdict.PASS, median_tok_s=1.5),
        ContextPoint(ctx=16384, verdict=PointVerdict.FAIL_TOO_SLOW, median_tok_s=0.8),
        ContextPoint(ctx=32768, verdict=PointVerdict.FAIL_LOAD, error="oom"),
    ]
    report = build_report(points)
    assert report.max_loadable_context == 16384
    assert report.max_stable_context == 8192
    assert report.max_usable_context == 8192
    assert report.recommended_interactive_context == 8192


def test_report_tracks_retrieval_separately():
    points = [
        ContextPoint(ctx=8192, verdict=PointVerdict.PASS, median_tok_s=1.5,
                     retrieval_passed=True),
        ContextPoint(ctx=16384, verdict=PointVerdict.PASS, median_tok_s=1.2,
                     retrieval_passed=False),
    ]
    report = build_report(points)
    assert report.max_usable_context == 16384
    assert report.max_retrieval_context == 8192


def test_report_is_json_serializable():
    import json

    points = [ContextPoint(ctx=4096, verdict=PointVerdict.PASS, median_tok_s=2.0)]
    payload = build_report(points).as_dict()
    json.dumps(payload)  # must not raise
    assert payload["max_stable_context"] == 4096


def test_format_report_contains_required_rows():
    points = [ContextPoint(ctx=4096, verdict=PointVerdict.PASS, median_tok_s=2.0)]
    text = format_report(build_report(points), "test-model")
    assert "Max loadable context" in text
    assert "Max stable context" in text
    assert "Recommended interactive" in text


def test_min_decode_tps_is_one():
    assert MIN_DECODE_TPS == 1.0
