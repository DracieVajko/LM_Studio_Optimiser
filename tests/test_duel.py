"""Duel unit tests (P4, spec section 14)."""
import pytest

from lm_optimizer.services.duel import (
    DUEL_TASKS,
    TIE_THRESHOLD,
    WEAK_WIN_THRESHOLD,
    BlindJudgeError,
    DuelConfig,
    DuelMode,
    DuelSide,
    DuelType,
    build_blind_pair,
    compute_verdict,
    interpret_margin,
    run_blind_judge,
    run_duel,
)


def _side(model_id: str, **kw) -> DuelSide:
    side = DuelSide(model_id=model_id)
    for key, value in kw.items():
        setattr(side, key, value)
    return side


# --- thresholds ---

def test_interpret_margin_thresholds():
    assert interpret_margin(0.0) == "tie"
    assert interpret_margin(2.9) == "tie"
    assert interpret_margin(3.0) == "weak_win"
    assert interpret_margin(6.9) == "weak_win"
    assert interpret_margin(7.0) == "clear_win"
    assert interpret_margin(-9.0) == "clear_win"


def test_thresholds_constants():
    assert TIE_THRESHOLD == 3.0
    assert WEAK_WIN_THRESHOLD == 7.0


# --- duel types ---

def test_duel_task_sets_defined():
    for duel_type in DuelType:
        assert DUEL_TASKS[duel_type], f"{duel_type} has no tasks"
    assert len(DUEL_TASKS[DuelType.COMPLETE_SANDBOX]) == 3
    assert len(DUEL_TASKS[DuelType.COMPLETE_MODEL]) >= 5


# --- blind evaluation ---

def test_blind_pair_hides_identity():
    a = _side("model-alpha")
    b = _side("model-beta")
    pair = build_blind_pair(a, b, seed=7, outputs={"model-alpha": "AAA", "model-beta": "BBB"})
    assert set(pair.payloads) == {"A", "B"}
    # The judge sees only payloads, never ids.
    assert "model-alpha" not in str(pair.payloads)
    assert "model-beta" not in str(pair.payloads)
    assert {pair.candidate_a, pair.candidate_b} == {"model-alpha", "model-beta"}


def test_blind_pair_order_is_deterministic_for_same_seed():
    a = _side("model-alpha")
    b = _side("model-beta")
    p1 = build_blind_pair(a, b, seed=11)
    p2 = build_blind_pair(a, b, seed=11)
    assert p1.candidate_a == p2.candidate_a
    assert p1.candidate_b == p2.candidate_b


def test_blind_pair_order_varies_with_seed():
    a = _side("model-alpha")
    b = _side("model-beta")
    orders = {build_blind_pair(a, b, seed=s).candidate_a for s in range(20)}
    assert len(orders) == 2, "seed should be able to flip the A/B order"


@pytest.mark.asyncio
async def test_blind_judge_maps_winner_back():
    a = _side("model-alpha")
    b = _side("model-beta")

    async def judge(pair):
        assert set(pair.payloads) == {"A", "B"}
        return {"winner": "A", "confidence": 0.8, "reasoning": "clearer structure"}

    result = await run_blind_judge(
        judge, a, b, outputs={"model-alpha": "x", "model-beta": "y"}
    )
    assert result["winner_model_id"] in ("model-alpha", "model-beta")
    assert result["blind_order"]["A"] == result["winner_model_id"]


@pytest.mark.asyncio
async def test_judge_cannot_judge_itself():
    a = _side("model-alpha")
    b = _side("model-beta")

    async def judge(pair):
        return {"winner": "A"}

    with pytest.raises(BlindJudgeError):
        await run_blind_judge(judge, a, b, judge_model_id="model-alpha")


# --- verdict ---

def test_verdict_quality_dimension():
    a = _side("alpha", outcomes={
        "t": {"score": 1.0, "tokens_per_second": 10.0, "tokens": 100, "gate_passed": True},
    })
    b = _side("beta", outcomes={
        "t": {"score": 0.5, "tokens_per_second": 20.0, "tokens": 100, "gate_passed": True},
    })
    verdict = compute_verdict(a, b, ["t"])
    assert verdict.dimension_winners["quality"] == "alpha"
    assert verdict.dimension_winners["performance"] == "beta"
    assert verdict.quality_delta == pytest.approx(0.5)
    assert verdict.performance_delta == pytest.approx(-10.0)


def test_hard_gate_failure_cannot_win():
    a = _side("alpha", outcomes={
        "t": {"score": 0.9, "tokens_per_second": 30.0, "gate_passed": False,
              "critical_failure": "build_failed"},
    })
    b = _side("beta", outcomes={
        "t": {"score": 0.6, "tokens_per_second": 5.0, "gate_passed": True},
    })
    verdict = compute_verdict(a, b, ["t"])
    assert verdict.winner == "beta"
    assert verdict.hard_gate_failures
    assert verdict.critical_failures


def test_verdict_tie_when_no_data():
    a = _side("alpha")
    b = _side("beta")
    verdict = compute_verdict(a, b, ["t"])
    assert verdict.winner is None
    assert verdict.interpretation == "tie"


def test_missing_data_is_not_a_win():
    a = _side("alpha", outcomes={"t": {"score": 1.0, "gate_passed": True}})
    b = _side("beta")  # no data at all
    verdict = compute_verdict(a, b, ["t"])
    assert verdict.winner == "alpha"
    assert verdict.dimension_winners["performance"] == "tie"


def test_verdict_serializes():
    a = _side("alpha", outcomes={"t": {"score": 1.0, "gate_passed": True}})
    b = _side("beta", outcomes={"t": {"score": 0.0, "gate_passed": True}})
    payload = compute_verdict(a, b, ["t"]).as_dict()
    assert "winner" in payload
    assert "dimension_winners" in payload


# --- controller ---

@pytest.mark.asyncio
async def test_run_duel_executes_all_tasks():
    a = _side("alpha")
    b = _side("beta")
    calls: list[tuple[str, str]] = []

    async def runner(side, task_id):
        calls.append((side.model_id, task_id))
        return {
            "score": 1.0 if side.model_id == "alpha" else 0.5,
            "tokens_per_second": 10.0,
            "tokens": 50,
            "gate_passed": True,
        }

    config = DuelConfig(duel_type=DuelType.MICRO_QUALITY)
    payload = await run_duel(a, b, config, runner)
    assert len(calls) == 2 * len(DUEL_TASKS[DuelType.MICRO_QUALITY])
    assert payload["verdict"]["winner"] == "alpha"
    assert payload["mode"] == DuelMode.CONTROLLED.value


@pytest.mark.asyncio
async def test_run_duel_records_runner_errors():
    a = _side("alpha")
    b = _side("beta")

    async def runner(side, task_id):
        if side.model_id == "beta":
            raise RuntimeError("model exploded")
        return {"score": 1.0, "gate_passed": True}

    payload = await run_duel(a, b, DuelConfig(duel_type=DuelType.MICRO_QUALITY), runner)
    assert payload["verdict"]["winner"] == "alpha"
    assert payload["verdict"]["critical_failures"]
