"""Correctness / quality evaluation service.

Uses deterministic heuristics; see docs/OPTIMIZATION_METHOD.md for
what the score actually measures and display guidance.
"""

import json
from dataclasses import dataclass

from lm_optimizer.config import config
from lm_optimizer.domain.models import BenchmarkMetrics, QualityScore
from lm_optimizer.logging_config import get_logger
from lm_optimizer.services.benchmark import BENCHMARK_SUITE, BenchmarkCase

logger = get_logger(__name__)


def _checks_from_scores(scores: dict) -> tuple[int, int]:
    total = len(scores)
    passed = sum(1 for v in scores.values() if v >= 0.9)
    return passed, total


@dataclass
class QualityConfig:
    """Correctness / quality evaluation configuration (heuristic checks)."""

    minimum_score: float = 0.97


class QualityEvaluator:
    """Evaluates output quality against reference expectations."""

    def __init__(self, quality_config: QualityConfig | None = None):
        self.config = quality_config or QualityConfig(
            minimum_score=config.optimization.minimum_quality_score
        )
        self.reference_outputs: dict[str, dict[str, str]] = {}

    def set_reference_outputs(self, model_id: str, outputs: dict[str, str]) -> None:
        """Set reference outputs for a model (from baseline config)."""
        self.reference_outputs[model_id] = outputs

    def evaluate(self, model_id: str, test_name: str, output: str) -> QualityScore:
        """Evaluate a single test output."""
        prompt_def = self._get_prompt_def(test_name)
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

        if prompt_def.category == "format":
            return self._evaluate_format(output, prompt_def)
        if prompt_def.category == "coding":
            return self._evaluate_coding(output, prompt_def)
        if prompt_def.category in ("instruction", "reasoning", "context"):
            return self._evaluate_general(output, prompt_def)
        return self._evaluate_general(output, prompt_def)

    def _get_prompt_def(self, test_name: str) -> BenchmarkCase | None:
        """Get benchmark prompt definition."""
        for p in BENCHMARK_SUITE:
            if p.name == test_name:
                return p
        return None

    def _evaluate_format(self, output: str, prompt_def: BenchmarkCase) -> QualityScore:
        """Evaluate structured output (JSON)."""
        details = {}
        scores = {}

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

        required = ["name", "age", "skills", "address"]
        missing = [f for f in required if f not in parsed]
        scores["task_completion"] = (
            1.0 if not missing else max(0.0, 1.0 - len(missing) / len(required))
        )
        details["missing_fields"] = missing

        type_ok = True
        if "age" in parsed and not isinstance(parsed["age"], int):
            type_ok = False
        if "skills" in parsed and not isinstance(parsed["skills"], list):
            type_ok = False
        if "address" in parsed and not isinstance(parsed["address"], dict):
            type_ok = False
        scores["factual_consistency"] = 1.0 if type_ok else 0.5
        details["types_correct"] = type_ok

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

    def _evaluate_coding(self, output: str, prompt_def: BenchmarkCase) -> QualityScore:
        """Evaluate code output."""
        details = {}
        scores = {}

        has_def = "def find_duplicates" in output
        scores["task_completion"] = 1.0 if has_def else 0.0
        details["has_function_def"] = has_def

        has_complexity = any(kw in output.lower() for kw in ["o(n)", "o(1)", "linear", "constant"])
        scores["factual_consistency"] = 1.0 if has_complexity else 0.7
        details["mentions_complexity"] = has_complexity

        has_sort = any(kw in output for kw in ["sort", "sorted", "ascending"])
        details["mentions_sort"] = has_sort

        no_extra = not any(kw in output for kw in ["```", "class ", "if __name__"])
        scores["format_compliance"] = 1.0 if no_extra else 0.7
        details["clean_output"] = no_extra

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

    def _evaluate_general(self, output: str, prompt_def: BenchmarkCase) -> QualityScore:
        """Evaluate general text output."""
        details = {}
        scores = {}

        min_tokens = prompt_def.max_tokens // 4
        output_tokens = len(output.split())
        scores["task_completion"] = min(1.0, output_tokens / max(min_tokens, 1))
        details["output_tokens"] = output_tokens

        if prompt_def.category == "reasoning":
            has_pattern = any(
                kw in output.lower() for kw in ["difference", "pattern", "add", "sequence", "72"]
            )
            scores["factual_consistency"] = 1.0 if has_pattern else 0.5
            details["has_reasoning"] = has_pattern
        elif prompt_def.category == "context":
            keywords = ["solar", "wind", "hydro", "geothermal", "biomass"]
            found = sum(1 for kw in keywords if kw in output.lower())
            scores["factual_consistency"] = min(1.0, found / len(keywords) * 1.5)
            details["keywords_found"] = found
        else:
            scores["factual_consistency"] = 1.0

        scores["format_compliance"] = 1.0
        scores["coding_correctness"] = 1.0

        ends_properly = output.strip()[-1] in ".!?" if output.strip() else False
        scores["no_truncation"] = 1.0 if ends_properly else 0.5
        details["ends_properly"] = ends_properly

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

        for n in [2, 3, 4]:
            ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
            if not ngrams:
                continue
            unique = set(ngrams)
            if len(unique) / len(ngrams) < (1 - threshold):
                return True
        return False

    def evaluate_all(
        self, model_id: str, metrics: list[BenchmarkMetrics]
    ) -> dict[str, QualityScore]:
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
        return quality.overall >= self.config.minimum_score
