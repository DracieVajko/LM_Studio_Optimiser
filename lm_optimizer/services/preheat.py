"""Real preheat abstraction: LOAD / VERIFY / PREHEAT / MEASURE.

Lifecycle (BenchmarkService.run_benchmark calls `run_preheat_phase` between
load and measured runs):

- LOAD: model already loaded by the caller (load_time measured by caller).
- VERIFY: single tiny echo generation proves the model serves.
- PREHEAT: identical prompt/config across candidates, timed separately as
  warmup_time_ms, results discarded (never contaminate load_time or metrics).
- MEASURE: caller proceeds with the real suite.

Preheat is configurable (prompt/max_tokens/temperature/repetitions),
cancellable (cancel callback polled between repetitions), checkpoint-aware
(caller logs events; timings recorded into result.generation["preheat"] so
they persist via config_json), and never fakes TTFT (it records only
wall-clock warmup time, no estimated_ttft synthesis).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from lm_optimizer.services.lm_studio import LMStudioClient


class PreheatPhase(StrEnum):
    LOAD = "load"
    VERIFY = "verify"
    PREHEAT = "preheat"
    MEASURE = "measure"


@dataclass
class PreheatConfig:
    """Preheat settings (identical across candidates)."""

    enabled: bool = True
    prompt: str = "Say hi in 5 words."
    max_tokens: int = 30
    temperature: float = 0.3
    repetitions: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "repetitions": self.repetitions,
        }


@dataclass
class PreheatResult:
    """Outcome of the preheat phase."""

    ok: bool
    warmup_time_ms: float = 0.0
    phases: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "warmup_time_ms": round(self.warmup_time_ms, 1),
            "phases": self.phases,
            "error": self.error,
        }


async def run_preheat_phase(
    client: LMStudioClient,
    model_id: str,
    config: PreheatConfig | None = None,
    reasoning: str | None = "off",
    is_cancelled: Callable[[], bool] | None = None,
) -> PreheatResult:
    """Run VERIFY + PREHEAT against an already-loaded model."""
    cfg = config or PreheatConfig()
    phases: list[str] = [PreheatPhase.LOAD.value]
    if not cfg.enabled or cfg.repetitions <= 0:
        return PreheatResult(ok=True, warmup_time_ms=0.0, phases=phases)
    try:
        # VERIFY: model must serve before preheat counts.
        verify = await client.chat_completion(
            model=model_id,
            input_text=cfg.prompt,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_tokens,
            reasoning=reasoning,
        )
        phases.append(PreheatPhase.VERIFY.value)
        _ = verify  # output discarded; serving is the assertion
        # PREHEAT: identical work, timed separately.
        start = time.perf_counter()
        for _ in range(max(1, cfg.repetitions)):
            if is_cancelled is not None and is_cancelled():
                elapsed_ms = (time.perf_counter() - start) * 1000
                return PreheatResult(
                    ok=False,
                    warmup_time_ms=elapsed_ms,
                    phases=phases,
                    error="cancelled during preheat",
                )
            await client.chat_completion(
                model=model_id,
                input_text=cfg.prompt,
                temperature=cfg.temperature,
                max_output_tokens=cfg.max_tokens,
                reasoning=reasoning,
            )
        warmup_ms = (time.perf_counter() - start) * 1000
        phases.append(PreheatPhase.PREHEAT.value)
        return PreheatResult(ok=True, warmup_time_ms=warmup_ms, phases=phases)
    except Exception as e:
        return PreheatResult(
            ok=False, warmup_time_ms=0.0, phases=phases, error=f"{type(e).__name__}: {e}"
        )
