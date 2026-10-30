"""Quality Lab unit tests (P3, spec section 12).

Pure logic: validators, hard gates, scoring and reliability aggregation.
No LM Studio, no network.
"""
import pytest

from lm_optimizer.services.quality_lab import (
    HardGate,
    HardGateKind,
    QualityCategory,
    QualityTask,
    TaskOutcome,
    Validator,
    ValidatorKind,
    aggregate_reliability,
    default_tasks,
    run_task,
    run_task_with_seeds,
    score_task,
)

# --- Validators ---

def test_exact_match_validator():
    v = Validator(ValidatorKind.EXACT_MATCH, "Bratislava")
    assert v.check("Bratislava")[0]
    assert not v.check("Praha")[0]


def test_contains_all_and_any_and_none():
    all_v = Validator(ValidatorKind.CONTAINS_ALL, ["red", "blue"])
    assert all_v.check("red, blue and green")[0]
    assert not all_v.check("red only")[0]

    any_v = Validator(ValidatorKind.CONTAINS_ANY, ["pristav", "lod"])
    assert any_v.check("to je pristav")[0]
    assert not any_v.check("nic takeho")[0]

    none_v = Validator(ValidatorKind.CONTAINS_NONE, ["praha"])
    assert none_v.check("bratislava")[0]
    assert not none_v.check("praha")[0]


def test_counts_only_validator():
    v = Validator(ValidatorKind.COUNTS_ONLY, 3)
    assert v.check("red, blue, green")[0]
    assert not v.check("red, blue")[0]


def test_regex_validator():
    v = Validator(ValidatorKind.REGEX, r"\b28\b")
    assert v.check("The answer is 28 euro")[0]
    assert not v.check("The answer is 21 euro")[0]


def test_json_schema_validator():
    schema = {
        "type": "object",
        "required": ["name", "age"],
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    }
    v = Validator(ValidatorKind.JSON_SCHEMA, schema)
    assert v.check('{"name": "Ada", "age": 36}')[0]
    assert not v.check('{"name": "Ada"}')[0]
    assert not v.check('{"name": "Ada", "age": "old"}')[0]
    # Fenced JSON is accepted too.
    assert v.check('```json\n{"name": "Ada", "age": 36}\n```')[0]


def test_json_schema_array_and_enum():
    schema = {"type": "array", "items": {"type": "integer"}}
    v = Validator(ValidatorKind.JSON_SCHEMA, schema)
    assert v.check("[1, 2, 3]")[0]
    assert not v.check('[1, "x"]')[0]

    enum_v = Validator(ValidatorKind.JSON_SCHEMA, {"type": "string", "enum": ["a", "b"]})
    assert enum_v.check('"a"')[0]
    assert not enum_v.check('"c"')[0]


def test_python_compiles_and_defines():
    comp = Validator(ValidatorKind.PYTHON_COMPILES)
    assert comp.check("def f():\n    return 1\n")[0]
    assert not comp.check("def f(:\n")[0]

    defines = Validator(ValidatorKind.PYTHON_DEFINES, ["find_duplicates"])
    assert defines.check("def find_duplicates(x):\n    return []\n")[0]
    assert not defines.check("def other(x):\n    return []\n")[0]


def test_json_keys_validator():
    v = Validator(ValidatorKind.JSON_KEYS, ["name", "age"])
    assert v.check('{"name": "x", "age": 1}')[0]
    assert not v.check('{"name": "x"}')[0]
    assert not v.check("not json")[0]


def test_retrieval_key_validator():
    v = Validator(ValidatorKind.RETRIEVAL_KEY, "VERDIGEN-8412")
    assert v.check("kód je VERDIGEN-8412")[0]
    assert not v.check("kód je iný")[0]


def test_min_words_validator():
    v = Validator(ValidatorKind.MIN_WORDS, 5)
    assert v.check("one two three four five six")[0]
    assert not v.check("one two")[0]


def test_no_repetition_validator():
    v = Validator(ValidatorKind.NO_REPETITION)
    good = " ".join(f"word{i}" for i in range(60))
    assert v.check(good)[0]
    bad = "the same thing " * 40
    assert not v.check(bad)[0]


def test_validator_never_raises():
    v = Validator(ValidatorKind.JSON_SCHEMA, {"type": "object"})
    ok, detail = v.check("{{{ not json")
    assert ok is False
    assert "invalid json" in detail or "error" in detail


# --- Hard gates ---

def test_hard_gate_output_not_empty():
    gate = HardGate(HardGateKind.OUTPUT_NOT_EMPTY)
    assert gate.check({"output": "text"})[0]
    assert not gate.check({"output": "   "})[0]


def test_hard_gate_exit_code_and_files():
    exit_gate = HardGate(HardGateKind.EXIT_CODE_ZERO)
    assert exit_gate.check({"exit_code": 0})[0]
    assert not exit_gate.check({"exit_code": 1})[0]

    files_gate = HardGate(HardGateKind.REQUIRED_FILES, ["index.html", "app.js"])
    assert files_gate.check({"files": ["index.html", "app.js"]})[0]
    assert not files_gate.check({"files": ["index.html"]})[0]


def test_hard_gate_unit_tests():
    gate = HardGate(HardGateKind.UNIT_TESTS_PASS)
    assert gate.check({"tests_passed": 5, "tests_failed": 0})[0]
    assert not gate.check({"tests_passed": 5, "tests_failed": 2})[0]
    assert not gate.check({})[0]


def test_hard_gate_exact_functionality():
    gate = HardGate(HardGateKind.EXACT_FUNCTIONALITY, ["export function render"])
    assert gate.check({"output": "export function render() {}"})[0]
    assert not gate.check({"output": "nothing"})[0]


# --- Scoring ---

def test_failed_hard_gate_caps_score_at_zero():
    task = QualityTask(
        id="t",
        category=QualityCategory.CODING,
        prompt="x",
        validators=[Validator(ValidatorKind.REGEX, r".*")],
        hard_gates=[HardGate(HardGateKind.OUTPUT_NOT_EMPTY)],
    )
    outcome = score_task(task, "")
    assert outcome.score == 0.0
    assert outcome.passed is False
    assert outcome.gate_passed is False
    assert any("output_not_empty" in d for d in outcome.gate_details)


def test_passed_task_scores_weighted_validators():
    task = QualityTask(
        id="t",
        category=QualityCategory.INSTRUCTION_FOLLOWING,
        prompt="x",
        validators=[
            Validator(ValidatorKind.CONTAINS_ALL, ["red"], weight=2.0),
            Validator(ValidatorKind.REGEX, r"blue", weight=1.0),
        ],
    )
    outcome = score_task(task, "red only")
    assert outcome.score == pytest.approx(2 / 3)
    assert outcome.passed is False
    outcome = score_task(task, "red and blue")
    assert outcome.score == pytest.approx(1.0)
    assert outcome.passed is True


def test_task_without_validators_passes_with_one():
    task = QualityTask(id="t", category=QualityCategory.ENGLISH, prompt="x")
    outcome = score_task(task, "anything")
    assert outcome.score == 1.0
    assert outcome.passed is True


# --- Reliability ---

def test_aggregate_reliability_statistics():
    outcomes = [
        TaskOutcome(task_id="t", category="reasoning", score=1.0, passed=True),
        TaskOutcome(task_id="t", category="reasoning", score=0.5, passed=False),
        TaskOutcome(task_id="t", category="reasoning", score=0.0, passed=False),
        TaskOutcome(task_id="t", category="reasoning", score=1.0, passed=True),
    ]
    report = aggregate_reliability("t", outcomes)
    assert report.runs == 4
    assert report.mean == pytest.approx(0.625)
    assert report.median == pytest.approx(0.75)
    assert report.worst == 0.0
    assert report.best == 1.0
    assert report.success_rate == pytest.approx(0.5)
    assert report.stdev > 0


def test_aggregate_reliability_empty():
    report = aggregate_reliability("t", [])
    assert report.runs == 0
    assert report.mean == 0.0


# --- Runner ---

@pytest.mark.asyncio
async def test_run_task_scores_output():
    async def chat(prompt: str, max_tokens: int) -> str:
        return "red, blue, green"

    task = QualityTask(
        id="colors",
        category=QualityCategory.INSTRUCTION_FOLLOWING,
        prompt="list colors",
        validators=[
            Validator(ValidatorKind.COUNTS_ONLY, 3),
            Validator(ValidatorKind.CONTAINS_ALL, ["red", "blue", "green"]),
        ],
    )
    outcome = await run_task(task, chat)
    assert outcome.passed
    assert outcome.score == 1.0


@pytest.mark.asyncio
async def test_run_task_handles_chat_error():
    async def chat(prompt: str, max_tokens: int) -> str:
        raise RuntimeError("boom")

    task = QualityTask(id="t", category=QualityCategory.ENGLISH, prompt="x")
    outcome = await run_task(task, chat)
    assert outcome.passed is False
    assert "boom" in outcome.error


@pytest.mark.asyncio
async def test_run_task_with_seeds_aggregates():
    calls = []

    async def chat(prompt: str, max_tokens: int) -> str:
        calls.append(prompt)
        return "answer" if len(calls) % 2 else "wrong"

    task = QualityTask(
        id="t",
        category=QualityCategory.REASONING,
        prompt="q",
        validators=[Validator(ValidatorKind.CONTAINS_ALL, ["answer"])],
    )
    outcomes, report = await run_task_with_seeds(task, chat, seeds=[1, 2, 3, 4])
    assert len(outcomes) == 4
    assert report.runs == 4
    assert 0.0 < report.success_rate < 1.0


# --- Task library ---

def test_default_tasks_are_wellformed():
    tasks = default_tasks()
    assert tasks
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids)), "task ids must be unique"
    for task in tasks:
        assert task.prompt.strip()
        assert isinstance(task.category, QualityCategory)
        assert task.max_tokens > 0
        # Every task must have at least one validator or a hard gate,
        # otherwise it cannot discriminate anything.
        assert task.validators or task.hard_gates


def test_default_tasks_cover_key_categories():
    cats = {t.category for t in default_tasks()}
    assert QualityCategory.INSTRUCTION_FOLLOWING in cats
    assert QualityCategory.CODING in cats
    assert QualityCategory.REASONING in cats
    assert QualityCategory.STRUCTURED_JSON in cats
    assert QualityCategory.SLOVAK in cats
    assert QualityCategory.HALLUCINATION_RESISTANCE in cats
