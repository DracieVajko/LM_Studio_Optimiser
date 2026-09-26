"""Strict configuration failure state machine (§1-§4, §25).

Phases: CREATED -> LOAD_REQUESTED -> LOADED -> VERIFIED -> PREHEATED ->
BENCHMARKED -> QUALITY_EVALUATED -> SCORED -> CANDIDATE_ELIGIBLE.
Any phase failure lands on a terminal failure class; failures are never
collapsed into one bucket:

- INCOMPATIBLE: hard lock (structural: unsupported/unrecognized parameter,
  model incompatibility). Never retried blindly.
- OOM / TIMEOUT / LOAD_FAILED / BENCHMARK_FAILED / VERIFY_FAILED /
  PREHEAT_FAILED: soft/learned failures.
- QUALITY_FAILED: benchmark metrics preserved, quality recorded, score None,
  excluded from winner selection (not a runtime failure).
"""

from __future__ import annotations

import re

from lm_optimizer.domain.models import SOFT_FAILURES, ConfigurationStatus

_OOM_RE = re.compile(r"\boom\b|out of memory|memory pressure|cuda out of")
_TIMEOUT_SIGNALS = ("timeout", "timed out", "deadline")
_HARD_SIGNALS = (
    "unrecognized",
    "unsupported",
    "not supported",
    "invalid parameter",
    "incompatible",
    "no such model",
    "model_not_found",
)
_LOAD_SIGNALS = ("load failed", "failed to load", "refus", "connection", "econn")


def classify_error(error: str | None) -> ConfigurationStatus:
    """Map a raw error string to a terminal failure class (order matters)."""
    err = (error or "").lower()
    if not err:
        return ConfigurationStatus.BENCHMARK_FAILED
    if _OOM_RE.search(err):
        return ConfigurationStatus.OOM
    if any(s in err for s in _TIMEOUT_SIGNALS):
        return ConfigurationStatus.TIMEOUT
    if any(s in err for s in _HARD_SIGNALS):
        return ConfigurationStatus.INCOMPATIBLE
    if any(s in err for s in _LOAD_SIGNALS):
        return ConfigurationStatus.LOAD_FAILED
    return ConfigurationStatus.BENCHMARK_FAILED


def is_hard(status: ConfigurationStatus) -> bool:
    """Hard lock (INCOMPATIBLE) vs soft/learned failure."""
    return status == ConfigurationStatus.INCOMPATIBLE


def is_soft(status: ConfigurationStatus) -> bool:
    """Soft failure that may succeed under other conditions."""
    return status in SOFT_FAILURES


def is_terminal_failure(status: ConfigurationStatus) -> bool:
    """Any non-passing terminal state (nothing here may be scored a winner)."""
    return status != ConfigurationStatus.PASSED
