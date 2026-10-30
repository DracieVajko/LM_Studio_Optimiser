"""Quality Lab: deterministic tasks, hard gates, validators, reliability.

Implements the master spec (LM_STUDIO_OPTIMISER_MASTER_SPEC_SK.md) section 12:

- 12.2 Hard gates BEFORE any subjective scoring: build, unit tests, schema
  validation, runtime exit code, required files, exact required
  functionality. A non-functional output can never win on appearance.
- 12.3 Separately switchable categories.
- 12.4 Automatic validators preferred over a judge: exact match, JSON
  Schema, compiler/type checker, unit tests, regex/AST constraints,
  expected tool name/arguments, retrieval key, deterministic file checks.
  An LLM judge is secondary, never the only authority.
- 12.5 Robustness: fixed temperature/seed, several seeds, report mean,
  median, worst case, standard deviation and success rate.

Everything is offline-testable: validators and scorers are pure functions
over strings; the runner only needs an async chat callable.
"""

from __future__ import annotations

import ast
import json
import re
import statistics
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# --- Categories (spec 12.3) ------------------------------------------------


class QualityCategory(str, Enum):
    """Separately switchable quality categories."""

    INSTRUCTION_FOLLOWING = "instruction"
    CODING = "coding"
    REASONING = "reasoning"
    STRUCTURED_JSON = "structured_json"
    TOOL_CALLING = "tool_calling"
    SLOVAK = "slovak"
    ENGLISH = "english"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    LONG_CONTEXT_RETRIEVAL = "long_context_retrieval"
    HALLUCINATION_RESISTANCE = "hallucination_resistance"
    RESEARCH_SYNTHESIS = "research_synthesis"


# --- Validators (spec 12.4) ------------------------------------------------


class ValidatorKind(str, Enum):
    EXACT_MATCH = "exact_match"
    CONTAINS_ALL = "contains_all"
    CONTAINS_ANY = "contains_any"
    CONTAINS_NONE = "contains_none"
    COUNTS_ONLY = "counts_only"
    REGEX = "regex"
    JSON_SCHEMA = "json_schema"
    PYTHON_COMPILES = "python_compiles"
    PYTHON_DEFINES = "python_defines"
    JSON_KEYS = "json_keys"
    RETRIEVAL_KEY = "retrieval_key"
    MIN_WORDS = "min_words"
    NO_REPETITION = "no_repetition"


@dataclass
class Validator:
    """One deterministic check over the model output."""

    kind: ValidatorKind
    value: Any = None
    weight: float = 1.0
    label: str = ""

    def check(self, output: str) -> tuple[bool, str]:
        """Return (passed, detail). Never raises: a broken check is a fail."""
        try:
            return self._check(output)
        except Exception as e:
            return False, f"validator error: {type(e).__name__}: {e}"[:160]

    def _check(self, output: str) -> tuple[bool, str]:
        text = output or ""
        if self.kind is ValidatorKind.EXACT_MATCH:
            expected = str(self.value or "").strip()
            got = text.strip()
            return got == expected, f"exact={'yes' if got == expected else 'no'}"
        if self.kind is ValidatorKind.CONTAINS_ALL:
            needles = [str(n) for n in (self.value or [])]
            missing = [n for n in needles if n.lower() not in text.lower()]
            return not missing, f"missing={missing[:3]}" if missing else "all present"
        if self.kind is ValidatorKind.CONTAINS_ANY:
            needles = [str(n) for n in (self.value or [])]
            hit = [n for n in needles if n.lower() in text.lower()]
            return bool(hit), f"hit={hit[:3]}" if hit else "no match"
        if self.kind is ValidatorKind.COUNTS_ONLY:
            # "exactly N items" - count list items or comma-separated parts.
            expected = int(self.value or 0)
            parts = [p for p in re.split(r"[,\n;]+", text.strip()) if p.strip()]
            if len(parts) != expected:
                # Fall back to a line count when the output is a plain list.
                lines = [ln for ln in text.strip().split("\n") if ln.strip()]
                if len(lines) == expected:
                    return True, f"lines={len(lines)}"
                return False, f"count={len(parts)} (expected {expected})"
            return True, f"count={len(parts)}"
        if self.kind is ValidatorKind.CONTAINS_NONE:
            needles = [str(n) for n in (self.value or [])]
            present = [n for n in needles if n.lower() in text.lower()]
            return not present, f"present={present[:3]}" if present else "none present"
        if self.kind is ValidatorKind.REGEX:
            pattern = str(self.value or "")
            hit = re.search(pattern, text, re.IGNORECASE | re.DOTALL) is not None
            return hit, f"regex={'hit' if hit else 'miss'}"
        if self.kind is ValidatorKind.JSON_SCHEMA:
            return _validate_json_schema(text, self.value or {})
        if self.kind is ValidatorKind.PYTHON_COMPILES:
            try:
                ast.parse(text)
                return True, "compiles"
            except SyntaxError as e:
                return False, f"syntax error line {e.lineno}"
        if self.kind is ValidatorKind.PYTHON_DEFINES:
            names = [str(n) for n in (self.value or [])]
            try:
                tree = ast.parse(text)
            except SyntaxError as e:
                return False, f"syntax error line {e.lineno}"
            defined = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            missing = [n for n in names if n not in defined]
            return not missing, f"missing={missing}" if missing else "defined"
        if self.kind is ValidatorKind.JSON_KEYS:
            try:
                doc = json.loads(_strip_fences(text).strip())
            except Exception as e:
                return False, f"invalid json: {str(e)[:80]}"
            if not isinstance(doc, dict):
                return False, "not a json object"
            required = [str(k) for k in (self.value or [])]
            missing = [k for k in required if k not in doc]
            return not missing, f"missing={missing}" if missing else "keys present"
        if self.kind is ValidatorKind.RETRIEVAL_KEY:
            key = str(self.value or "")
            return key.lower() in text.lower(), f"key={'found' if key.lower() in text.lower() else 'missing'}"
        if self.kind is ValidatorKind.MIN_WORDS:
            minimum = int(self.value or 0)
            count = len(text.split())
            return count >= minimum, f"words={count} (min {minimum})"
        if self.kind is ValidatorKind.NO_REPETITION:
            return not _has_repetition(text), "repetition detected" if _has_repetition(text) else "no repetition"
        return False, f"unknown validator {self.kind}"


def _strip_fences(text: str) -> str:
    """Remove ```json fences around a payload."""
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _has_repetition(text: str, threshold: float = 0.3) -> bool:
    """True when the same 5-gram repeats too often (degenerate output)."""
    words = (text or "").lower().split()
    if len(words) < 30:
        return False
    grams: dict[str, int] = {}
    for i in range(len(words) - 4):
        key = " ".join(words[i : i + 5])
        grams[key] = grams.get(key, 0) + 1
    if not grams:
        return False
    return max(grams.values()) / len(grams) > threshold


def _validate_json_schema(text: str, schema: dict) -> tuple[bool, str]:
    """Minimal JSON Schema subset: type, required, properties, enum, items."""
    try:
        doc = json.loads(_strip_fences(text).strip())
    except Exception as e:
        return False, f"invalid json: {str(e)[:80]}"

    def _check(value: Any, sch: dict, path: str) -> str | None:
        expected = sch.get("type")
        if expected:
            types = expected if isinstance(expected, list) else [expected]
            ok = any(
                (t == "object" and isinstance(value, dict))
                or (t == "array" and isinstance(value, list))
                or (t == "string" and isinstance(value, str))
                or (t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool))
                or (t == "integer" and isinstance(value, int) and not isinstance(value, bool))
                or (t == "boolean" and isinstance(value, bool))
                or (t == "null" and value is None)
                for t in types
            )
            if not ok:
                return f"{path}: expected {expected}, got {type(value).__name__}"
        if "enum" in sch and value not in sch["enum"]:
            return f"{path}: not in enum"
        if isinstance(value, dict):
            for key in sch.get("required", []):
                if key not in value:
                    return f"{path}.{key}: required"
            for key, sub in (sch.get("properties") or {}).items():
                if key in value:
                    err = _check(value[key], sub, f"{path}.{key}")
                    if err:
                        return err
        if isinstance(value, list) and isinstance(sch.get("items"), dict):
            for i, item in enumerate(value):
                err = _check(item, sch["items"], f"{path}[{i}]")
                if err:
                    return err
        return None

    err = _check(doc, schema, "$")
    return (err is None), (err or "schema ok")


# --- Hard gates (spec 12.2) ------------------------------------------------


class HardGateKind(str, Enum):
    """Mechanical checks that must pass before subjective scoring."""

    OUTPUT_NOT_EMPTY = "output_not_empty"
    EXIT_CODE_ZERO = "exit_code_zero"
    REQUIRED_FILES = "required_files"
    UNIT_TESTS_PASS = "unit_tests_pass"
    SCHEMA_VALID = "schema_valid"
    EXACT_FUNCTIONALITY = "exact_functionality"


@dataclass
class HardGate:
    """One mechanical gate. A failed gate caps the task score at zero."""

    kind: HardGateKind
    value: Any = None
    label: str = ""

    def check(self, ctx: dict) -> tuple[bool, str]:
        try:
            return self._check(ctx)
        except Exception as e:
            return False, f"gate error: {type(e).__name__}: {e}"[:160]

    def _check(self, ctx: dict) -> tuple[bool, str]:
        output = str(ctx.get("output") or "")
        if self.kind is HardGateKind.OUTPUT_NOT_EMPTY:
            ok = bool(output.strip())
            return ok, "output present" if ok else "output empty"
        if self.kind is HardGateKind.EXIT_CODE_ZERO:
            code = ctx.get("exit_code")
            ok = code in (0, None)
            return ok, f"exit_code={code}"
        if self.kind is HardGateKind.REQUIRED_FILES:
            required = [str(f) for f in (self.value or [])]
            present = {str(f) for f in (ctx.get("files") or [])}
            missing = [f for f in required if f not in present]
            return not missing, f"missing={missing}" if missing else "files present"
        if self.kind is HardGateKind.UNIT_TESTS_PASS:
            passed = ctx.get("tests_passed")
            failed = ctx.get("tests_failed")
            if passed is None:
                return False, "no test results"
            ok = int(passed) > 0 and not int(failed or 0)
            return ok, f"tests {passed} passed / {failed or 0} failed"
        if self.kind is HardGateKind.SCHEMA_VALID:
            ok, detail = _validate_json_schema(output, self.value or {})
            return ok, detail
        if self.kind is HardGateKind.EXACT_FUNCTIONALITY:
            needles = [str(n) for n in (self.value or [])]
            missing = [n for n in needles if n.lower() not in output.lower()]
            return not missing, f"missing={missing}" if missing else "functionality present"
        return False, f"unknown gate {self.kind}"


# --- Tasks -----------------------------------------------------------------


@dataclass
class QualityTask:
    """One deterministic quality task."""

    id: str
    category: QualityCategory
    prompt: str
    validators: list[Validator] = field(default_factory=list)
    hard_gates: list[HardGate] = field(default_factory=list)
    max_tokens: int = 512
    weight: float = 1.0
    retrieval_key: str | None = None


@dataclass
class TaskOutcome:
    """Result of one task run (one seed)."""

    task_id: str
    category: str
    output: str = ""
    passed: bool = False
    score: float = 0.0
    gate_passed: bool = True
    gate_details: list[str] = field(default_factory=list)
    validator_details: list[str] = field(default_factory=list)
    error: str = ""
    duration_ms: float = 0.0
    tokens: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "passed": self.passed,
            "score": round(self.score, 4),
            "gate_passed": self.gate_passed,
            "gate_details": list(self.gate_details),
            "validator_details": list(self.validator_details),
            "error": self.error,
            "duration_ms": round(self.duration_ms, 1),
            "tokens": self.tokens,
        }


# --- Scoring ---------------------------------------------------------------


def score_task(task: QualityTask, output: str, ctx: dict | None = None) -> TaskOutcome:
    """Score one output against a task: hard gates FIRST, then validators.

    A failed hard gate means score 0 - a non-functional output can never
    win on appearance (spec 12.2).
    """
    ctx = dict(ctx or {})
    ctx.setdefault("output", output)

    gate_details: list[str] = []
    gate_ok = True
    for gate in task.hard_gates:
        ok, detail = gate.check(ctx)
        gate_details.append(f"{gate.kind.value}:{detail}")
        if not ok:
            gate_ok = False

    if not gate_ok:
        return TaskOutcome(
            task_id=task.id,
            category=task.category.value,
            output=output,
            passed=False,
            score=0.0,
            gate_passed=False,
            gate_details=gate_details,
        )

    if not task.validators:
        return TaskOutcome(
            task_id=task.id,
            category=task.category.value,
            output=output,
            passed=True,
            score=1.0,
            gate_passed=True,
            gate_details=gate_details,
        )

    details: list[str] = []
    total = 0.0
    weight_sum = 0.0
    for validator in task.validators:
        ok, detail = validator.check(output)
        details.append(f"{validator.kind.value}:{detail}")
        total += validator.weight if ok else 0.0
        weight_sum += validator.weight
    score = (total / weight_sum) if weight_sum else 0.0
    return TaskOutcome(
        task_id=task.id,
        category=task.category.value,
        output=output,
        passed=score >= 0.999,
        score=score,
        gate_passed=True,
        gate_details=gate_details,
        validator_details=details,
    )


# --- Reliability (spec 12.5) ----------------------------------------------


@dataclass
class ReliabilityReport:
    """Aggregate over repeated runs of the same task (several seeds)."""

    task_id: str
    runs: int = 0
    mean: float = 0.0
    median: float = 0.0
    worst: float = 0.0
    best: float = 0.0
    stdev: float = 0.0
    success_rate: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "runs": self.runs,
            "mean": round(self.mean, 4),
            "median": round(self.median, 4),
            "worst": round(self.worst, 4),
            "best": round(self.best, 4),
            "stdev": round(self.stdev, 4),
            "success_rate": round(self.success_rate, 4),
        }


def aggregate_reliability(task_id: str, outcomes: list[TaskOutcome]) -> ReliabilityReport:
    """mean / median / worst / stdev / success rate over repeated runs."""
    scores = [o.score for o in outcomes]
    if not scores:
        return ReliabilityReport(task_id=task_id, runs=0)
    passed = sum(1 for o in outcomes if o.passed)
    return ReliabilityReport(
        task_id=task_id,
        runs=len(scores),
        mean=statistics.fmean(scores),
        median=statistics.median(scores),
        worst=min(scores),
        best=max(scores),
        stdev=statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        success_rate=passed / len(scores),
    )


# --- Runner ----------------------------------------------------------------

ChatFn = Callable[[str, int], Awaitable[str]]


async def run_task(
    task: QualityTask,
    chat: ChatFn,
    ctx: dict | None = None,
) -> TaskOutcome:
    """Run one task through a chat callable and score it."""
    import time

    t0 = time.perf_counter()
    try:
        output = await chat(task.prompt, task.max_tokens)
    except Exception as e:
        return TaskOutcome(
            task_id=task.id,
            category=task.category.value,
            passed=False,
            score=0.0,
            error=f"{type(e).__name__}: {e}"[:200],
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
    outcome = score_task(task, output, ctx)
    outcome.duration_ms = (time.perf_counter() - t0) * 1000
    return outcome


async def run_task_with_seeds(
    task: QualityTask,
    chat: ChatFn,
    seeds: list[int],
    ctx: dict | None = None,
) -> tuple[list[TaskOutcome], ReliabilityReport]:
    """Run a task once per seed and aggregate reliability."""
    outcomes: list[TaskOutcome] = []
    for seed in seeds:
        outcome = await run_task(task, chat, {**(ctx or {}), "seed": seed})
        outcomes.append(outcome)
    return outcomes, aggregate_reliability(task.id, outcomes)


# --- Task library ----------------------------------------------------------


def default_tasks() -> list[QualityTask]:
    """A small deterministic task library covering the main categories."""
    return [
        QualityTask(
            id="instruction_list_three",
            category=QualityCategory.INSTRUCTION_FOLLOWING,
            prompt=(
                "List exactly three primary colors, separated by commas. "
                "Output only the three names, nothing else."
            ),
            validators=[
                Validator(ValidatorKind.COUNTS_ONLY, 3),
                Validator(ValidatorKind.CONTAINS_ALL, ["red", "blue", "green"]),
            ],
            hard_gates=[HardGate(HardGateKind.OUTPUT_NOT_EMPTY)],
            max_tokens=64,
        ),
        QualityTask(
            id="structured_json_person",
            category=QualityCategory.STRUCTURED_JSON,
            prompt=(
                'Return ONLY a JSON object with keys "name" (string), '
                '"age" (integer) and "active" (boolean). No prose.'
            ),
            validators=[
                Validator(ValidatorKind.JSON_SCHEMA, {
                    "type": "object",
                    "required": ["name", "age", "active"],
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "active": {"type": "boolean"},
                    },
                }),
            ],
            hard_gates=[
                HardGate(HardGateKind.OUTPUT_NOT_EMPTY),
                HardGate(HardGateKind.SCHEMA_VALID, {
                    "type": "object",
                    "required": ["name", "age", "active"],
                }),
            ],
            max_tokens=128,
        ),
        QualityTask(
            id="coding_find_duplicates",
            category=QualityCategory.CODING,
            prompt=(
                "Write a Python function `def find_duplicates(items):` that "
                "returns a list of values that appear more than once. "
                "Return only code."
            ),
            validators=[
                Validator(ValidatorKind.PYTHON_COMPILES),
                Validator(ValidatorKind.PYTHON_DEFINES, ["find_duplicates"]),
            ],
            hard_gates=[HardGate(HardGateKind.OUTPUT_NOT_EMPTY)],
            max_tokens=512,
        ),
        QualityTask(
            id="reasoning_arithmetic",
            category=QualityCategory.REASONING,
            prompt=(
                "A shop sells 3 boxes for 12 euro. How much do 7 boxes cost? "
                "Answer with the number only."
            ),
            validators=[
                Validator(ValidatorKind.REGEX, r"\b28\b"),
                Validator(ValidatorKind.CONTAINS_NONE, ["cannot", "unable"]),
            ],
            hard_gates=[HardGate(HardGateKind.OUTPUT_NOT_EMPTY)],
            max_tokens=64,
        ),
        QualityTask(
            id="slovak_translation",
            category=QualityCategory.SLOVAK,
            prompt=(
                "Prelož do slovenčiny: 'The harbour master logged every vessel "
                "before the fog lifted.' Odpovedz iba prekladom."
            ),
            validators=[
                Validator(ValidatorKind.MIN_WORDS, 6),
                Validator(ValidatorKind.CONTAINS_ANY, ["prístav", "lod", "fog", "hml"]),
            ],
            hard_gates=[HardGate(HardGateKind.OUTPUT_NOT_EMPTY)],
            max_tokens=128,
        ),
        QualityTask(
            id="hallucination_resistance",
            category=QualityCategory.HALLUCINATION_RESISTANCE,
            prompt=(
                "What is the capital city of Slovakia? Answer with the name "
                "only. If you are not sure, say UNKNOWN."
            ),
            validators=[
                Validator(ValidatorKind.CONTAINS_ANY, ["bratislava", "bratislav"]),
                Validator(ValidatorKind.CONTAINS_NONE, ["prague", "praha", "budapest"]),
            ],
            hard_gates=[HardGate(HardGateKind.OUTPUT_NOT_EMPTY)],
            max_tokens=64,
        ),
    ]
