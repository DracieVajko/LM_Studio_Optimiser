"""Model Duel: controlled, best-profile, blind evaluation, dimension verdicts.

Implements the master spec (LM_STUDIO_OPTIMISER_MASTER_SPEC_SK.md) section 14:

- 14.1 Duel types are expressed as TASK names; each type maps to a task set
  (micro quality, web, research, blender, complete sandbox, complete model).
- 14.2 Controlled duel: identical context, sampling, seed, prompts, tools,
  time/token budget, repair count and sandbox permissions.
- 14.3 Best-profile duel: each side uses its own best validated preset.
- 14.4 Blind evaluation: outputs are anonymized as Candidate A/B; the
  judge must not see model, parameter count, quant, speed or history.
  A/B order is randomized or balanced.
- 14.5 Result: winner/tie, confidence, dimension winners, critical failures,
  hard-gate status, performance / quality / efficiency differences.
  Interpretations: < 3 points = tie, 3-7 = weak win, > 7 = clear win.

The judge NEVER evaluates a model's own output (spec: "a model must not
score its own output"). `BlindJudge` takes an injected judge callable, and
the controller refuses to use the same model id for judge and candidate.
"""

from __future__ import annotations

import hashlib
import random
import statistics
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# --- Duel types (spec 14.1) -----------------------------------------------


class DuelType(str, Enum):
    MICRO_QUALITY = "micro_quality"
    WEB = "web"
    RESEARCH = "research"
    BLENDER = "blender"
    COMPLETE_SANDBOX = "complete_sandbox"
    COMPLETE_MODEL = "complete_model"


#: Task sets per duel type.
DUEL_TASKS: dict[DuelType, tuple[str, ...]] = {
    DuelType.MICRO_QUALITY: ("instruction_list_three", "reasoning_arithmetic"),
    DuelType.WEB: ("web_dashboard",),
    DuelType.RESEARCH: ("research_synthesis",),
    DuelType.BLENDER: ("blender_skyblock",),
    DuelType.COMPLETE_SANDBOX: (
        "web_dashboard",
        "research_synthesis",
        "blender_skyblock",
    ),
    DuelType.COMPLETE_MODEL: (
        "instruction_list_three",
        "structured_json_person",
        "coding_find_duplicates",
        "reasoning_arithmetic",
        "slovak_translation",
        "hallucination_resistance",
    ),
}


class DuelMode(str, Enum):
    """Controlled (identical conditions) vs best-profile (own presets)."""

    CONTROLLED = "controlled"
    BEST_PROFILE = "best_profile"


# --- Thresholds (spec 14.5) ----------------------------------------------


TIE_THRESHOLD = 3.0
WEAK_WIN_THRESHOLD = 7.0


def interpret_margin(delta: float) -> str:
    """< 3 = tie, 3-7 = weak win, > 7 = clear win."""
    if abs(delta) < TIE_THRESHOLD:
        return "tie"
    if abs(delta) < WEAK_WIN_THRESHOLD:
        return "weak_win"
    return "clear_win"


# --- Sides ----------------------------------------------------------------


@dataclass
class DuelSide:
    """One candidate in a duel (never exposes identity to a blind judge)."""

    model_id: str
    label: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    context_length: int = 16384
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 1024
    time_budget_s: float = 300.0
    repair_attempts: int = 2
    sandbox_permissions: dict[str, Any] = field(default_factory=dict)
    #: Measured results per task: task_id -> outcome dict.
    outcomes: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _anon_key(self) -> str:
        return hashlib.sha256(self.model_id.encode()).hexdigest()[:8]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "label": self.label or self.model_id,
            "context_length": self.context_length,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "repair_attempts": self.repair_attempts,
            "outcomes": dict(self.outcomes),
        }


@dataclass
class BlindPair:
    """Anonymized pair handed to the judge."""

    candidate_a: str
    candidate_b: str
    payloads: dict[str, str]
    order_seed: int


def build_blind_pair(
    side_a: DuelSide,
    side_b: DuelSide,
    seed: int = 42,
    outputs: dict[str, str] | None = None,
) -> BlindPair:
    """Anonymize two sides as Candidate A/B with a deterministic order.

    The mapping is derived from the seed, so the same duel replays with the
    same assignment (reproducible) while the judge still cannot infer which
    side is which from the payload.
    """
    rng = random.Random(f"{seed}:{side_a.model_id}:{side_b.model_id}")
    swapped = rng.random() < 0.5
    first, second = (side_b, side_a) if swapped else (side_a, side_b)
    payloads = outputs or {}
    anonymized = {
        "A": payloads.get(first.model_id, ""),
        "B": payloads.get(second.model_id, ""),
    }
    return BlindPair(
        candidate_a=first.model_id,
        candidate_b=second.model_id,
        payloads=anonymized,
        order_seed=seed,
    )


# --- Verdict --------------------------------------------------------------


@dataclass
class DuelVerdict:
    """Final duel outcome (spec 14.5)."""

    winner: str | None = None
    confidence: float = 0.0
    interpretation: str = "tie"
    dimension_winners: dict[str, str] = field(default_factory=dict)
    critical_failures: list[str] = field(default_factory=list)
    hard_gate_failures: list[str] = field(default_factory=list)
    performance_delta: float = 0.0
    quality_delta: float = 0.0
    efficiency_delta: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "confidence": round(self.confidence, 4),
            "interpretation": self.interpretation,
            "dimension_winners": dict(self.dimension_winners),
            "critical_failures": list(self.critical_failures),
            "hard_gate_failures": list(self.hard_gate_failures),
            "performance_delta": round(self.performance_delta, 4),
            "quality_delta": round(self.quality_delta, 4),
            "efficiency_delta": round(self.efficiency_delta, 4),
            "details": dict(self.details),
        }


def _median(values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else 0.0


def compute_verdict(
    side_a: DuelSide,
    side_b: DuelSide,
    task_ids: list[str] | None = None,
) -> DuelVerdict:
    """Compare two sides across the given tasks.

    Quality and hard-gate failures dominate: a side that fails a hard gate
    cannot win on the other dimensions (spec 24: "a quality score may not
    outrank a hard functional failure").
    """
    tasks = list(task_ids or DUEL_TASKS[DuelType.COMPLETE_MODEL])

    def _collect(side: DuelSide) -> tuple[list[float], list[float], list[float], list[str], list[str]]:
        scores: list[float] = []
        speeds: list[float] = []
        tokens: list[float] = []
        critical: list[str] = []
        gates: list[str] = []
        for task in tasks:
            out = side.outcomes.get(task) or {}
            score = out.get("score")
            if isinstance(score, (int, float)):
                scores.append(float(score))
            spd = out.get("tokens_per_second")
            if isinstance(spd, (int, float)):
                speeds.append(float(spd))
            tok = out.get("tokens")
            if isinstance(tok, (int, float)):
                tokens.append(float(tok))
            if out.get("critical_failure"):
                critical.append(f"{side.model_id}:{task}:{out.get('critical_failure')}")
            if out.get("gate_passed") is False:
                gates.append(f"{side.model_id}:{task}")
        return scores, speeds, tokens, critical, gates

    scores_a, speeds_a, tokens_a, crit_a, gates_a = _collect(side_a)
    scores_b, speeds_b, tokens_b, crit_b, gates_b = _collect(side_b)

    quality_a = _median(scores_a)
    quality_b = _median(scores_b)
    perf_a = _median(speeds_a)
    perf_b = _median(speeds_b)
    eff_a = _median(tokens_a)
    eff_b = _median(tokens_b)

    quality_delta = quality_a - quality_b
    performance_delta = perf_a - perf_b
    efficiency_delta = eff_a - eff_b

    # The spec's thresholds ("3 points = tie, 3-7 = weak win") live on a
    # 0-100 rubric, while the raw deltas mix quality (0..1) with tok/s and
    # token counts. Normalize every dimension to comparable units:
    #   quality   -> points (0..100)
    #   speed     -> percent difference
    #   efficiency-> percent difference
    quality_points = quality_delta * 100.0
    perf_pct = (
        (performance_delta / max(perf_a, perf_b) * 100.0)
        if max(perf_a, perf_b) > 0
        else 0.0
    )
    eff_pct = (
        (efficiency_delta / max(eff_a, eff_b) * 100.0)
        if max(eff_a, eff_b) > 0
        else 0.0
    )

    dimension_winners: dict[str, str] = {}
    for name, delta in (
        ("quality", quality_delta),
        ("performance", performance_delta),
        ("efficiency", efficiency_delta),
    ):
        if abs(delta) < 0.0001:
            dimension_winners[name] = "tie"
        else:
            dimension_winners[name] = side_a.model_id if delta > 0 else side_b.model_id

    critical = crit_a + crit_b
    hard_gates = gates_a + gates_b

    # Hard-gate failures decide first: a broken output never wins.
    if gates_a and not gates_b:
        winner = side_b.model_id
        confidence = 0.95
    elif gates_b and not gates_a:
        winner = side_a.model_id
        confidence = 0.95
    else:
        composite = quality_points * 0.6 + perf_pct * 0.25 + eff_pct * 0.15
        if abs(composite) < TIE_THRESHOLD:
            winner = None
            confidence = 0.5
        else:
            winner = side_a.model_id if composite > 0 else side_b.model_id
            confidence = min(1.0, 0.5 + abs(composite) / 50.0)

    margin = abs(quality_points) * 0.6 + abs(perf_pct) * 0.25 + abs(eff_pct) * 0.15
    return DuelVerdict(
        winner=winner,
        confidence=confidence,
        interpretation=interpret_margin(margin),
        dimension_winners=dimension_winners,
        critical_failures=critical,
        hard_gate_failures=hard_gates,
        performance_delta=performance_delta,
        quality_delta=quality_delta,
        efficiency_delta=efficiency_delta,
        details={
            "quality": {"a": quality_a, "b": quality_b},
            "performance": {"a": perf_a, "b": perf_b},
            "efficiency": {"a": eff_a, "b": eff_b},
            "quality_points_delta": round(quality_points, 3),
            "performance_pct_delta": round(perf_pct, 3),
            "efficiency_pct_delta": round(eff_pct, 3),
            "tasks": tasks,
        },
    )


# --- Judge ----------------------------------------------------------------

JudgeFn = Callable[[BlindPair], Awaitable[dict[str, Any]]]


class BlindJudgeError(RuntimeError):
    """Raised when the judge would evaluate a model's own output."""


async def run_blind_judge(
    judge: JudgeFn,
    side_a: DuelSide,
    side_b: DuelSide,
    judge_model_id: str = "",
    seed: int = 42,
    outputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a blind judge over two anonymized sides.

    The judge receives ONLY the anonymized payloads - never model ids,
    parameter counts, quants or measured speeds. A judge that is one of the
    candidates is refused (a model must not score its own output).
    """
    if judge_model_id and judge_model_id in (side_a.model_id, side_b.model_id):
        raise BlindJudgeError(
            "judge model must not be one of the candidates (no self-evaluation)"
        )
    pair = build_blind_pair(side_a, side_b, seed=seed, outputs=outputs)
    result = await judge(pair)
    if not isinstance(result, dict):
        return {"winner": None, "confidence": 0.0, "raw": str(result)[:400]}
    # Map the anonymized winner back to the real model id.
    raw_winner = result.get("winner")
    if raw_winner in ("A", "a"):
        result["winner_model_id"] = pair.candidate_a
    elif raw_winner in ("B", "b"):
        result["winner_model_id"] = pair.candidate_b
    else:
        result["winner_model_id"] = None
    result["blind_order"] = {"A": pair.candidate_a, "B": pair.candidate_b}
    return result


# --- Controller ------------------------------------------------------------


@dataclass
class DuelConfig:
    """Duel setup."""

    duel_type: DuelType = DuelType.MICRO_QUALITY
    mode: DuelMode = DuelMode.CONTROLLED
    seed: int = 42
    context_length: int = 16384
    temperature: float = 0.0
    max_tokens: int = 1024
    time_budget_s: float = 300.0
    repair_attempts: int = 2
    blind: bool = True
    judge_model_id: str = ""


async def run_duel(
    side_a: DuelSide,
    side_b: DuelSide,
    config: DuelConfig,
    runner: Callable[[DuelSide, str], Awaitable[dict[str, Any]]],
    judge: JudgeFn | None = None,
) -> dict[str, Any]:
    """Run a full duel: both sides on the same task set, then the verdict.

    `runner(side, task_id)` executes one task for one side and returns the
    outcome dict (score, tokens_per_second, tokens, gate_passed, ...).
    """
    tasks = list(DUEL_TASKS[config.duel_type])

    for side in (side_a, side_b):
        for task in tasks:
            try:
                side.outcomes[task] = await runner(side, task)
            except Exception as e:
                side.outcomes[task] = {
                    "score": 0.0,
                    "error": f"{type(e).__name__}: {e}"[:200],
                    "gate_passed": False,
                    "critical_failure": "runner_error",
                }

    verdict = compute_verdict(side_a, side_b, tasks)

    payload: dict[str, Any] = {
        "duel_type": config.duel_type.value,
        "mode": config.mode.value,
        "tasks": tasks,
        "side_a": side_a.as_dict(),
        "side_b": side_b.as_dict(),
        "verdict": verdict.as_dict(),
    }

    if judge is not None:
        outputs = {
            side_a.model_id: str(side_a.outcomes.get(tasks[0], {}).get("output", "")) if tasks else "",
            side_b.model_id: str(side_b.outcomes.get(tasks[0], {}).get("output", "")) if tasks else "",
        }
        payload["judge"] = await run_blind_judge(
            judge,
            side_a,
            side_b,
            judge_model_id=config.judge_model_id,
            seed=config.seed,
            outputs=outputs,
        )
    return payload
