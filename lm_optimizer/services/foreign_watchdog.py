"""Foreign-model watchdog for LM Studio and Ollama.

A hostile (or merely curious) local client can load a model into the inference
server while an optimization run is benchmarking a different one. Two things
then go wrong:

1. VRAM/RAM contention. On a small GPU the foreign load fails, and LM Studio
   reports ``Engine protocol startup was aborted`` - which can tear down the
   generation already in flight for the model under test.
2. Wrong measurements. A benchmark answered by the wrong model is not a
   measurement of anything.

Observed on this host: a third-party client POSTing ``/v1/chat/completions``
for ``qwen/qwen3-coder-30b`` (18.6 GB) during a run testing a 14.6 GB model,
producing a long tail of "insufficient system resources" failures.

This module DETECTS such loads and, when explicitly enabled, unloads them.
Design rules:

- **Detect-only by default.** Unloading is a real mutation, so removal is
  opt-in via ``allow_unload``.
- **Never unload an expected model.** The model under test, and the echo-probe
  model that ``connect()`` legitimately loads, are always protected.
- **Evidence, not guesses.** Each finding records the instance id and the
  applied config so a failure can be explained afterwards.
- **Fail closed on detection.** If the host cannot be inspected, the verdict is
  "unknown", not "clean" - a blind "all good" is how corruption sneaks through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)

BACKEND_LMSTUDIO = "lm-studio"
BACKEND_OLLAMA = "ollama"


@dataclass(frozen=True)
class LoadedModel:
    """One model the host currently reports as loaded."""

    backend: str
    model_id: str
    instance_id: str | None = None
    config: dict = field(default_factory=dict)

    def key(self) -> str:
        """Identity used for matching; Ollama names carry a ``:tag`` suffix."""
        base = self.model_id.split(":")[0]
        return base.strip().lower()


@dataclass(frozen=True)
class ForeignLoad:
    """Evidence that a model outside the expected set is loaded."""

    model: LoadedModel
    detected_at: datetime
    reason: str

    def describe(self) -> str:
        inst = f" instance={self.model.instance_id}" if self.model.instance_id else ""
        return (
            f"[{self.model.backend}] {self.model.model_id}{inst} - {self.reason}"
        )


@dataclass
class WatchdogReport:
    """Outcome of one inspection."""

    checked_at: datetime
    backend: str
    inspected: bool
    expected: list[LoadedModel] = field(default_factory=list)
    foreign: list[ForeignLoad] = field(default_factory=list)
    unloaded: list[str] = field(default_factory=list)
    unload_failed: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def clean(self) -> bool:
        """True only when the host was inspected AND nothing foreign is loaded."""
        return self.inspected and not self.foreign

    def summary(self) -> str:
        if not self.inspected:
            return f"{self.backend}: UNKNOWN ({self.error or 'host not inspectable'})"
        if not self.foreign:
            return f"{self.backend}: clean ({len(self.expected)} expected loaded)"
        parts = [self.foreign[0].describe()]
        if len(self.foreign) > 1:
            parts.append(f"(+{len(self.foreign) - 1} more)")
        if self.unloaded:
            parts.append(f"unloaded={len(self.unloaded)}")
        if self.unload_failed:
            parts.append(f"unload_failed={len(self.unload_failed)}")
        return f"{self.backend}: FOREIGN " + "; ".join(parts)


def normalize_expected(model_ids: Iterable[str | None]) -> set[str]:
    """Expected model keys (lowercase, tag stripped, empties dropped)."""
    out: set[str] = set()
    for mid in model_ids:
        if not mid:
            continue
        out.add(str(mid).split(":")[0].strip().lower())
    return out


async def list_loaded(client: Any, backend: str) -> list[LoadedModel] | None:
    """Currently loaded models for a backend.

    Returns ``None`` when the host could not be inspected - the caller must
    treat that as "unknown", never as "clean".
    """
    try:
        if backend == BACKEND_LMSTUDIO:
            instances = await client.get_loaded_instances()
            return [
                LoadedModel(
                    backend=backend,
                    model_id=inst.get("model", ""),
                    instance_id=inst.get("instance_id"),
                    config=inst.get("config") or {},
                )
                for inst in instances
                if inst.get("model")
            ]
        if backend == BACKEND_OLLAMA:
            running = await client.refresh_running_models()
            return [
                LoadedModel(backend=backend, model_id=name)
                for name in (running or set())
                if name
            ]
    except Exception as e:
        logger.warning("Foreign-load inspection failed", backend=backend, error=str(e)[:160])
        return None
    return None


class ForeignModelWatchdog:
    """Detect (and optionally unload) models outside the expected set.

    ``expected_models`` always includes the model under test and any model the
    optimizer legitimately touches (for example the smallest model loaded by the
    connection echo probe).
    """

    def __init__(
        self,
        client: Any,
        backend: str = BACKEND_LMSTUDIO,
        *,
        expected_models: Iterable[str | None] = (),
        allow_unload: bool = False,
    ) -> None:
        self.client = client
        self.backend = backend
        self.allow_unload = allow_unload
        self._expected = normalize_expected(expected_models)
        self.history: list[WatchdogReport] = []

    # ---- expected-set management -------------------------------------

    @property
    def expected(self) -> set[str]:
        return set(self._expected)

    def set_expected(self, model_ids: Iterable[str | None]) -> None:
        """Replace the expected set (call when the run moves to another model)."""
        self._expected = normalize_expected(model_ids)

    def protect(self, model_ids: Iterable[str | None]) -> None:
        """Add to the expected set without dropping existing entries."""
        self._expected |= normalize_expected(model_ids)

    def is_expected(self, model_id: str) -> bool:
        return model_id.split(":")[0].strip().lower() in self._expected

    # ---- inspection ----------------------------------------------------

    async def check(self) -> WatchdogReport:
        """Inspect the host once. Never raises."""
        report = WatchdogReport(
            checked_at=datetime.now(), backend=self.backend, inspected=False
        )
        loaded = await list_loaded(self.client, self.backend)
        if loaded is None:
            report.error = "host not inspectable"
            self.history.append(report)
            return report

        report.inspected = True
        report.expected = [m for m in loaded if self.is_expected(m.model_id)]
        foreign_models = [m for m in loaded if not self.is_expected(m.model_id)]

        for m in foreign_models:
            report.foreign.append(
                ForeignLoad(
                    model=m,
                    detected_at=report.checked_at,
                    reason="loaded while a different model is under test",
                )
            )

        if report.foreign and self.allow_unload:
            await self._unload(report)

        self.history.append(report)
        if report.foreign:
            logger.warning(
                "FOREIGN MODEL LOADED",
                backend=self.backend,
                models=[f.model.model_id for f in report.foreign],
                unloaded=report.unloaded,
                failed=report.unload_failed,
            )
        return report

    async def _unload(self, report: WatchdogReport) -> None:
        """Unload foreign models only; expected models are never touched."""
        for finding in report.foreign:
            m = finding.model
            try:
                if self.backend == BACKEND_LMSTUDIO:
                    ok = await self.client.unload_model(
                        model_id=m.model_id, identifier=m.instance_id
                    )
                else:
                    ok = await self.client.unload(m.model_id)
            except Exception as e:
                report.unload_failed.append(m.model_id)
                logger.warning(
                    "Foreign unload raised",
                    model=m.model_id,
                    error=f"{type(e).__name__}: {e}"[:160],
                )
                continue
            if ok:
                report.unloaded.append(m.model_id)
                logger.info("Unloaded foreign model", model=m.model_id, backend=self.backend)
            else:
                report.unload_failed.append(m.model_id)
                logger.warning(
                    "Foreign unload reported failure",
                    model=m.model_id,
                    backend=self.backend,
                )

    # ---- convenience ---------------------------------------------------

    async def wait_until_clean(
        self,
        *,
        attempts: int = 3,
        delay_s: float = 2.0,
    ) -> WatchdogReport:
        """Inspect until clean or attempts are exhausted; last report is returned."""
        import asyncio

        report = await self.check()
        for _ in range(max(0, attempts - 1)):
            if report.clean:
                return report
            await asyncio.sleep(delay_s)
            report = await self.check()
        return report

    def worst(self) -> WatchdogReport | None:
        """The most recent report that found something foreign, if any."""
        for r in reversed(self.history):
            if r.foreign:
                return r
        return None


def build_watchdog(
    client: Any,
    backend: str,
    *,
    model_under_test: str | None,
    echo_probe_model: str | None = None,
    allow_unload: bool = False,
) -> ForeignModelWatchdog:
    """Watchdog protecting the model under test plus the echo-probe model."""
    return ForeignModelWatchdog(
        client,
        backend,
        expected_models=[model_under_test, echo_probe_model],
        allow_unload=allow_unload,
    )