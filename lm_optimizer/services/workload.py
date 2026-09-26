"""Authoritative workload abstraction.

INTERACTIVE: single-user inference. Optimize TTFT + decode latency; parallel
sessions fixed to 1 unless the user explicitly overrides `parallels`.
THROUGHPUT: parallel sessions measured concurrently. Aggregate throughput is
a MEASURED quantity (total generated tokens / wall-clock seconds over N
concurrent requests, see `summarize_throughput` and
BenchmarkService.measure_throughput) — never an inference from
per-request speed multiplied by parallelism. Persisted as `workload_type`
in run.benchmark_params (run metadata, no schema migration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lm_optimizer.domain.models import ConfigurationResult


class WorkloadType(StrEnum):
    INTERACTIVE = "interactive"
    THROUGHPUT = "throughput"


@dataclass(frozen=True)
class WorkloadConfig:
    workload_type: WorkloadType = WorkloadType.INTERACTIVE

    @classmethod
    def parse(cls, value: str | WorkloadType | None) -> WorkloadConfig:
        if isinstance(value, WorkloadType):
            return cls(value)
        v = str(value or "interactive").lower()
        if v in ("throughput", "batch", "server"):
            return cls(WorkloadType.THROUGHPUT)
        return cls(WorkloadType.INTERACTIVE)


def normalize_workload(value: str | WorkloadType | None) -> str:
    """Canonical lowercase workload string for metadata/CLI/API."""
    return WorkloadConfig.parse(value).workload_type.value


def apply_to_space(
    parallels: list[int],
    workload: str | WorkloadType | None,
    explicit_override: list[int] | None = None,
) -> list[int]:
    """Constrain stage-4 parallels by workload.

    Interactive fixes parallel=1 unless the user explicitly passed
    `parallels` (advanced_settings/CLI). Throughput keeps the full list.
    """
    cfg = WorkloadConfig.parse(workload)
    if cfg.workload_type == WorkloadType.INTERACTIVE and not explicit_override:
        return [1] if 1 in parallels else parallels[:1]
    return list(parallels)


@dataclass
class ThroughputMeasurement:
    """Real concurrent throughput measurement (all fields recorded)."""

    parallelism: int = 1
    total_tokens: int = 0
    elapsed_s: float = 0.0
    aggregate_tok_s: float = 0.0
    per_request_tok_s: list[float] = field(default_factory=list)
    per_request_elapsed_ms: list[float] = field(default_factory=list)
    failures: int = 0
    errors: list[str] = field(default_factory=list)
    queue_wait_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "parallelism": self.parallelism,
            "total_tokens": self.total_tokens,
            "elapsed_s": round(self.elapsed_s, 3),
            "aggregate_tok_s": round(self.aggregate_tok_s, 2),
            "per_request_tok_s": [round(v, 2) for v in self.per_request_tok_s],
            "per_request_elapsed_ms": [round(v, 1) for v in self.per_request_elapsed_ms],
            "failures": self.failures,
            "errors": self.errors,
            "queue_wait_ms": round(self.queue_wait_ms, 1),
        }


@dataclass
class _RequestOutcome:
    """One concurrent request outcome (input to summarize_throughput)."""

    tokens: int = 0
    elapsed_s: float = 0.0
    ok: bool = True
    error: str = ""


def summarize_throughput(
    parallelism: int,
    outcomes: list[_RequestOutcome],
    wall_s: float,
) -> ThroughputMeasurement:
    """Pure aggregate calculation (deterministic, unit-testable).

    aggregate_tok_s = total ok tokens / wall-clock seconds. queue_wait_ms
    approximates server-side queuing as wall time minus the fastest request
    (exact per-request server timing is not exposed by the API).
    """
    ok = [o for o in outcomes if o.ok]
    total = sum(o.tokens for o in ok)
    aggregate = total / wall_s if wall_s > 0 else 0.0
    per_tok = [o.tokens / o.elapsed_s if o.elapsed_s > 0 else 0.0 for o in ok]
    per_ms = [o.elapsed_s * 1000 for o in ok]
    fastest = min((o.elapsed_s for o in ok), default=0.0)
    return ThroughputMeasurement(
        parallelism=max(1, parallelism),
        total_tokens=total,
        elapsed_s=max(0.0, wall_s),
        aggregate_tok_s=aggregate,
        per_request_tok_s=per_tok,
        per_request_elapsed_ms=per_ms,
        failures=len(outcomes) - len(ok),
        errors=[o.error for o in outcomes if not o.ok and o.error][:10],
        queue_wait_ms=max(0.0, (wall_s - fastest) * 1000) if ok else 0.0,
    )


def aggregate_throughput(result: ConfigurationResult) -> float:
    """Legacy rough estimate: median gen tok/s * effective parallel.

    Kept for backward compatibility only. It is NOT a measurement — the
    decision path uses `effective_throughput` (measured aggregate when a
    concurrent probe recorded one, else the single-request rate).
    """
    try:
        gen = float(result.get_avg_generation_tok_s())
    except (TypeError, ValueError):
        return 0.0
    try:
        parallel = int(result.config.parallel or 1)
    except (TypeError, ValueError):
        parallel = 1
    return gen * max(1, parallel)


def recorded_aggregate(result: ConfigurationResult) -> float | None:
    """Measured aggregate tok/s recorded by a concurrent probe, if any."""
    try:
        recorded = (result.generation or {}).get("throughput_measured") or {}
        value = recorded.get("aggregate_tok_s")
        return float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def effective_throughput(result: ConfigurationResult) -> float:
    """Throughput used for decisions: measured aggregate when recorded.

    Fallback is the single-request generation rate (parallel=1 semantics).
    It is deliberately NOT multiplied by parallelism: without a concurrent
    measurement there is no evidence for linear scaling.
    """
    measured = recorded_aggregate(result)
    if measured is not None:
        return measured
    try:
        return float(result.get_avg_generation_tok_s())
    except (TypeError, ValueError):
        return 0.0


def workload_tiebreak_key(
    workload: str | WorkloadType | None, result: ConfigurationResult
) -> tuple[float, float]:
    """Secondary ordering inside the selection band by workload.

    Interactive prefers lower TTFT then higher decode speed; throughput
    prefers higher effective (measured-first) throughput then decode speed.
    """
    cfg = WorkloadConfig.parse(workload)
    try:
        ttft = float(result.get_avg_estimated_ttft_ms())
    except (TypeError, ValueError):
        ttft = float("inf")
    try:
        gen = float(result.get_avg_generation_tok_s())
    except (TypeError, ValueError):
        gen = 0.0
    if cfg.workload_type == WorkloadType.THROUGHPUT:
        return (effective_throughput(result), gen)
    return (-ttft if ttft != float("inf") else 0.0, gen)
