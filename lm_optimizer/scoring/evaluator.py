"""Correctness / quality scoring and evaluation for benchmark outputs.

This evaluator uses deterministic heuristics (JSON validation, keyword checks,
code structure, truncation detection, repetition). It does NOT measure
objective model quality with scientific precision. The metric should be
presented as "Correctness / Quality Score" or "X / Y checks passed"
and thresholds are configurable. See docs/OPTIMIZATION_METHOD.md.
"""

import json
from dataclasses import dataclass, field

from lm_optimizer.benchmark.suite import BENCHMARK_SUITE
from lm_optimizer.config import config
from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)


def _checks_from_scores(scores: dict) -> tuple[int, int]:
    """Count heuristic checks passed (score >= 0.9 counts as passed)."""
    total = len(scores)
    passed = sum(1 for v in scores.values() if v >= 0.9)
    return passed, total


@dataclass
class QualityScore:
    """Correctness / quality assessment result.

    This is a heuristic score based on deterministic checks, not an
    objective model quality percentage. Display as checks passed.
    """

    overall: float  # 0.0 to 1.0 - average of heuristic checks
    task_completion: float
    factual_consistency: float
    format_compliance: float
    coding_correctness: float
    no_truncation: float
    no_malformed: float
    confident: bool = True
    details: dict = field(default_factory=dict)
    checks_passed: int | None = None
    checks_total: int | None = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}
        # Auto-fill checks if not provided
        if self.checks_total is None or self.checks_passed is None:
            scores = {
                "task_completion": self.task_completion,
                "factual_consistency": self.factual_consistency,
                "format_compliance": self.format_compliance,
                "coding_correctness": self.coding_correctness,
                "no_truncation": self.no_truncation,
                "no_malformed": self.no_malformed,
            }
            passed, total = _checks_from_scores(scores)
            if self.checks_passed is None:
                self.checks_passed = passed
            if self.checks_total is None:
                self.checks_total = total

    def as_checks_str(self) -> str:
        return f"{self.checks_passed} / {self.checks_total} checks passed"


class QualityEvaluator:
    """Evaluates output quality against reference expectations."""

    def __init__(self):
        self.minimum_score = config.optimization.minimum_quality_score
        self.reference_outputs: dict[str, str] = {}

    def set_reference_outputs(self, model_id: str, outputs: dict[str, str]) -> None:
        """Set reference outputs for a model (from baseline config)."""
        self.reference_outputs[model_id] = outputs

    def evaluate(
        self, model_id: str, test_name: str, output: str, expected: dict | None = None
    ) -> QualityScore:
        """Evaluate a single test output."""
        # Get benchmark prompt definition
        prompt_def = next((p for p in BENCHMARK_SUITE if p.name == test_name), None)
        if not prompt_def:
            return QualityScore(
                overall=1.0,
                task_completion=1.0,
                factual_consistency=1.0,
                format_compliance=1.0,
                coding_correctness=1.0,
                no_truncation=1.0,
                no_malformed=1.0,
                confident=False,
                details={"reason": "Unknown test"},
            )

        # Run category-specific evaluators
        if prompt_def.category == "format":
            return self._evaluate_format(output, prompt_def)
        if prompt_def.category == "coding":
            return self._evaluate_coding(output, prompt_def)
        if prompt_def.category in ("instruction", "reasoning", "context"):
            return self._evaluate_general(output, prompt_def, expected)
        return self._evaluate_general(output, prompt_def, expected)

    def _evaluate_format(self, output: str, prompt_def) -> QualityScore:
        """Evaluate structured output (JSON)."""
        details = {}
        scores = {}

        # Check if valid JSON
        try:
            parsed = json.loads(output.strip())
            scores["format_compliance"] = 1.0
            details["json_valid"] = True
        except json.JSONDecodeError as e:
            scores["format_compliance"] = 0.0
            details["json_valid"] = False
            details["json_error"] = str(e)
            return QualityScore(
                overall=0.0,
                task_completion=0.0,
                factual_consistency=1.0,
                format_compliance=0.0,
                coding_correctness=1.0,
                no_truncation=1.0,
                no_malformed=0.0,
                confident=True,
                details=details,
            )

        # Check required fields
        required = ["name", "age", "skills", "address"]
        missing = [f for f in required if f not in parsed]
        if not missing:
            scores["task_completion"] = 1.0
        else:
            scores["task_completion"] = max(0.0, 1.0 - len(missing) / len(required))
        details["missing_fields"] = missing

        # Check types
        type_ok = True
        if "age" in parsed and not isinstance(parsed["age"], int):
            type_ok = False
        if "skills" in parsed and not isinstance(parsed["skills"], list):
            type_ok = False
        if "address" in parsed and not isinstance(parsed["address"], dict):
            type_ok = False
        scores["factual_consistency"] = 1.0 if type_ok else 0.5
        details["types_correct"] = type_ok

        # No truncation (output should end with })
        scores["no_truncation"] = 1.0 if output.strip().endswith("}") else 0.5
        scores["no_malformed"] = 1.0
        scores["coding_correctness"] = 1.0

        overall = sum(scores.values()) / len(scores)
        passed, total = _checks_from_scores(scores)
        details["checks"] = f"{passed}/{total}"
        return QualityScore(
            overall=overall,
            confident=True,
            details=details,
            checks_passed=passed,
            checks_total=total,
            **scores,
        )

    def _evaluate_coding(self, output: str, prompt_def) -> QualityScore:
        """Evaluate code output."""
        details = {}
        scores = {}

        # Check for function definition
        has_def = "def find_duplicates" in output
        scores["task_completion"] = 1.0 if has_def else 0.0
        details["has_function_def"] = has_def

        # Check for O(n) claim or implementation hints
        has_complexity = any(kw in output.lower() for kw in ["o(n)", "o(1)", "linear", "constant"])
        scores["factual_consistency"] = 1.0 if has_complexity else 0.7
        details["mentions_complexity"] = has_complexity

        # Check for sorting/ascending order
        has_sort = any(kw in output for kw in ["sort", "sorted", "ascending"])
        details["mentions_sort"] = has_sort

        # Format compliance - should be just function
        no_extra = not any(kw in output for kw in ["```", "class ", "if __name__"])
        scores["format_compliance"] = 1.0 if no_extra else 0.7
        details["clean_output"] = no_extra

        # No truncation
        scores["no_truncation"] = 1.0 if output.strip().endswith((")", "]", "}")) else 0.5
        scores["no_malformed"] = 1.0
        scores["coding_correctness"] = scores["task_completion"]

        overall = sum(scores.values()) / len(scores)
        passed, total = _checks_from_scores(scores)
        details["checks"] = f"{passed}/{total}"
        return QualityScore(
            overall=overall,
            confident=True,
            details=details,
            checks_passed=passed,
            checks_total=total,
            **scores,
        )

    def _evaluate_general(self, output: str, prompt_def, expected: dict | None) -> QualityScore:
        """Evaluate general text output."""
        details = {}
        scores = {}

        # Task completion - did it answer?
        min_tokens = prompt_def.min_tokens
        output_tokens = len(output.split())
        scores["task_completion"] = min(1.0, output_tokens / max(min_tokens, 1))
        details["output_tokens"] = output_tokens

        # Factual consistency - check for key terms
        if prompt_def.category == "reasoning":
            # Should mention sequence pattern
            has_pattern = any(
                kw in output.lower() for kw in ["difference", "pattern", "add", "sequence", "72"]
            )
            scores["factual_consistency"] = 1.0 if has_pattern else 0.5
            details["has_reasoning"] = has_pattern
        elif prompt_def.category == "context":
            # Should mention key renewable sources
            keywords = ["solar", "wind", "hydro", "geothermal", "biomass"]
            found = sum(1 for kw in keywords if kw in output.lower())
            scores["factual_consistency"] = min(1.0, found / len(keywords) * 1.5)
            details["keywords_found"] = found
        else:
            scores["factual_consistency"] = 1.0

        # Format compliance - no obvious formatting issues
        scores["format_compliance"] = 1.0
        scores["coding_correctness"] = 1.0  # N/A

        # No truncation - output should end with punctuation
        ends_properly = output.strip()[-1] in ".!?" if output.strip() else False
        scores["no_truncation"] = 1.0 if ends_properly else 0.5
        details["ends_properly"] = ends_properly

        # No malformed - no repetition, no garbage
        has_repetition = self._detect_repetition(output)
        scores["no_malformed"] = 0.0 if has_repetition else 1.0
        details["has_repetition"] = has_repetition

        overall = sum(scores.values()) / len(scores)
        passed, total = _checks_from_scores(scores)
        details["checks"] = f"{passed}/{total}"
        confident = prompt_def.category in ("instruction", "reasoning", "context")
        return QualityScore(
            overall=overall,
            confident=confident,
            details=details,
            checks_passed=passed,
            checks_total=total,
            **scores,
        )

    def _detect_repetition(self, text: str, threshold: float = 0.3) -> bool:
        """Detect excessive repetition in output."""
        words = text.split()
        if len(words) < 10:
            return False

        # Check for repeated n-grams
        for n in [2, 3, 4]:
            ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
            if not ngrams:
                continue
            unique = set(ngrams)
            if len(unique) / len(ngrams) < (1 - threshold):
                return True
        return False

    def evaluate_all(self, model_id: str, metrics: list) -> dict[str, QualityScore]:
        """Evaluate all benchmark metrics."""
        results = {}
        for m in metrics:
            if m.success and m.output_text:
                results[m.test_name] = self.evaluate(model_id, m.test_name, m.output_text)
        return results

    def aggregate_quality(self, scores: dict[str, QualityScore]) -> QualityScore:
        """Aggregate quality scores across tests."""
        if not scores:
            return QualityScore(
                overall=1.0,
                task_completion=1.0,
                factual_consistency=1.0,
                format_compliance=1.0,
                coding_correctness=1.0,
                no_truncation=1.0,
                no_malformed=1.0,
                confident=False,
            )

        confident_scores = [s for s in scores.values() if s.confident]
        if not confident_scores:
            confident_scores = list(scores.values())

        avg = lambda attr: sum(getattr(s, attr) for s in confident_scores) / len(confident_scores)

        # Aggregate checks
        total_passed = sum(s.checks_passed or 0 for s in confident_scores)
        total_checks = sum(s.checks_total or 0 for s in confident_scores)

        return QualityScore(
            overall=avg("overall"),
            task_completion=avg("task_completion"),
            factual_consistency=avg("factual_consistency"),
            format_compliance=avg("format_compliance"),
            coding_correctness=avg("coding_correctness"),
            no_truncation=avg("no_truncation"),
            no_malformed=avg("no_malformed"),
            confident=all(s.confident for s in confident_scores),
            details={"per_test": {k: v.overall for k, v in scores.items()}},
            checks_passed=total_passed,
            checks_total=total_checks,
        )

    def passes_threshold(self, quality: QualityScore) -> bool:
        """Check if quality meets minimum threshold."""
        return quality.overall >= self.minimum_score
