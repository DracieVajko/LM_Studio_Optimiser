"""A/B compare service: same custom tests on two configs, one verdict.

Same fairness rule as the optimizer: both sides run the IDENTICAL case
list (never different tests per side). Results carry phase="compare" so
they never mix with speed/quality/recovery measurements.
"""

from dataclasses import dataclass, field
from pathlib import Path

from lm_optimizer.domain.models import BenchmarkCase, ConfigurationResult, LoadConfiguration
from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)

MAX_COMPARE_TESTS = 10
MAX_COMPARE_TOKENS = 1024
# 5% non-inferiority band (same convention as the speed frontier).
COMPARE_BAND = 0.05


@dataclass
class CompareCase:
    """One custom compare test."""

    name: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.3
    category: str = "custom"
    min_tokens: int = 10
    stop_sequences: list[str] | None = None


@dataclass
class CompareResult:
    """Outcome of an A/B comparison."""

    model_id: str
    result_a: ConfigurationResult | None = None
    result_b: ConfigurationResult | None = None
    side_a_ok: bool = False
    side_b_ok: bool = False
    error_a: str | None = None
    error_b: str | None = None
    speed_a: float = 0.0
    speed_b: float = 0.0
    delta_tok_s: float = 0.0
    speedup: float | None = None
    quality_a: float | None = None
    quality_b: float | None = None
    speed_winner: str | None = None  # "A" | "B" | "draw"
    quality_winner: str | None = None
    verdict: str = "pending"
    cases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model_id,
            "cases": list(self.cases),
            "side_a_ok": self.side_a_ok,
            "side_b_ok": self.side_b_ok,
            "error_a": self.error_a,
            "error_b": self.error_b,
            "speed_a_tok_s": round(self.speed_a, 1),
            "speed_b_tok_s": round(self.speed_b, 1),
            "delta_tok_s": round(self.delta_tok_s, 1),
            "speedup_vs_b": self.speedup,
            "quality_a": self.quality_a,
            "quality_b": self.quality_b,
            "speed_winner": self.speed_winner,
            "quality_winner": self.quality_winner,
            "verdict": self.verdict,
        }


def parse_compare_cases(data: dict) -> list[CompareCase]:
    """Validate custom test JSON: {"tests": [{name, prompt, ...}]}."""
    if not isinstance(data, dict) or not isinstance(data.get("tests"), list):
        raise ValueError('Compare JSON must be {"tests": [{name, prompt, ...}]}')
    raw = data["tests"]
    if not raw:
        raise ValueError("Compare JSON has no tests (at least 1 required)")
    if len(raw) > MAX_COMPARE_TESTS:
        raise ValueError(f"Compare JSON has {len(raw)} tests, max is {MAX_COMPARE_TESTS}")
    cases: list[CompareCase] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Test #{i} must be an object with name+prompt")
        name = str(item.get("name") or "").strip()
        prompt = str(item.get("prompt") or "").strip()
        if not name:
            raise ValueError(f"Test #{i} is missing required field 'name'")
        if not prompt:
            raise ValueError(f"Test '{name or i}' is missing required field 'prompt'")
        if name in seen:
            raise ValueError(f"Duplicate test name '{name}'")
        seen.add(name)
        try:
            max_tokens = int(item.get("max_tokens", 256))
        except (TypeError, ValueError):
            raise ValueError(f"Test '{name}': max_tokens must be an integer")
        if not 1 <= max_tokens <= MAX_COMPARE_TOKENS:
            raise ValueError(f"Test '{name}': max_tokens must be 1-{MAX_COMPARE_TOKENS}")
        try:
            temperature = float(item.get("temperature", 0.3))
        except (TypeError, ValueError):
            raise ValueError(f"Test '{name}': temperature must be a number")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"Test '{name}': temperature must be 0-2")
        stop = item.get("stop_sequences")
        if stop is not None and not isinstance(stop, list):
            raise ValueError(f"Test '{name}': stop_sequences must be a list")
        cases.append(
            CompareCase(
                name=name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                category=str(item.get("category") or "custom"),
                stop_sequences=list(stop) if stop else None,
            )
        )
    return cases


def load_compare_cases_file(path: str | Path) -> list[CompareCase]:
    """Load + validate custom tests from a JSON file (fail-fast)."""
    import json

    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"Tests file not found: {p}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Tests file is not valid JSON: {e}")
    return parse_compare_cases(data)


def default_compare_cases() -> list[CompareCase]:
    """Default suite = the 5 campaign benchmark prompts (comparable history)."""
    from lm_optimizer.benchmark.suite import BENCHMARK_SUITE

    return [
        CompareCase(
            name=p.name,
            prompt=p.prompt,
            max_tokens=p.max_tokens,
            temperature=p.temperature,
            category=p.category,
            min_tokens=getattr(p, "min_tokens", 10),
            stop_sequences=list(p.stop_sequences) if p.stop_sequences else None,
        )
        for p in BENCHMARK_SUITE
    ]


def to_benchmark_cases(cases: list[CompareCase]) -> list[BenchmarkCase]:
    """Adapt compare cases to the benchmark runner input."""
    return [
        BenchmarkCase(
            name=c.name,
            category=c.category,
            prompt=c.prompt,
            max_tokens=c.max_tokens,
            temperature=c.temperature,
            stop_sequences=c.stop_sequences,
        )
        for c in cases
    ]


def load_config_from_dict(data: dict) -> LoadConfiguration:
    """Build LoadConfiguration from a plain dict (unknown keys rejected)."""
    import dataclasses

    valid = {f.name for f in dataclasses.fields(LoadConfiguration)}
    unknown = sorted(set(data) - valid - {"API_KEYS"})
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
    return LoadConfiguration(**{k: v for k, v in data.items() if k in valid})


def load_config_file(path: str | Path) -> LoadConfiguration:
    """Load a side config (config JSON file)."""
    import json

    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"Config file not found: {p}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Config file is not valid JSON: {e}")
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object")
    return load_config_from_dict(data)


def _band_winner(a: float, b: float, band: float = COMPARE_BAND) -> str | None:
    """A/B/draw verdict with a non-inferiority band (never exact-float compare)."""
    top = max(a, b)
    if top <= 0:
        return None
    if abs(a - b) / top <= band:
        return "draw"
    return "A" if a > b else "B"


async def run_ab_compare(
    client,
    model_id: str,
    config_a: LoadConfiguration,
    config_b: LoadConfiguration,
    cases: list[CompareCase] | None,
    repetitions: int = 2,
    context_length: int = 2048,
    evaluator=None,
) -> CompareResult:
    """Run the SAME case list on config A then B; never raises fatally on load fail."""
    from lm_optimizer.services.benchmark import BenchmarkConfig, BenchmarkService
    from lm_optimizer.services.quality import QualityEvaluator

    active = list(cases) if cases else default_compare_cases()
    bench_cases = to_benchmark_cases(active)
    res = CompareResult(model_id=model_id, cases=[c.name for c in active])
    svc = BenchmarkService(
        client,
        benchmark_config=BenchmarkConfig(repetitions=repetitions, warmup_repetitions=0),
    )
    ev = evaluator or QualityEvaluator()

    for tag, cfg in (("A", config_a), ("B", config_b)):
        if tag == "B":
            # Fail-closed boundary: B must not run beside a resident A.
            from lm_optimizer.services.unload_guard import assert_unloaded

            await assert_unloaded(client, purpose=f"compare:{model_id}")
        try:
            out = await svc.run_cases(
                model_id, cfg, context_length, bench_cases,
                repetitions=repetitions, warmup_repetitions=0,
            )
            try:
                gen = dict(out.generation or {})
                gen["phase"] = "compare"
                gen["compare_side"] = tag
                out.generation = gen
            except Exception:
                pass
            ok = out.status not in ("failed", "error") and not out.error
            # A load failure still yields ok=False with preserved error text.
            if tag == "A":
                res.result_a, res.side_a_ok = out, bool(ok)
                if not ok:
                    res.error_a = out.error or f"side A status={out.status}"
            else:
                res.result_b, res.side_b_ok = out, bool(ok)
                if not ok:
                    res.error_b = out.error or f"side B status={out.status}"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:200]
            if tag == "A":
                res.error_a = err
            else:
                res.error_b = err

    # Speed + quality summaries (quality None-safe: unknown custom tests score 1.0).
    try:
        res.speed_a = res.result_a.get_avg_generation_tok_s() if res.result_a else 0.0
    except Exception:
        res.speed_a = 0.0
    try:
        res.speed_b = res.result_b.get_avg_generation_tok_s() if res.result_b else 0.0
    except Exception:
        res.speed_b = 0.0
    res.delta_tok_s = round(res.speed_a - res.speed_b, 1)
    if res.speed_b > 0 and res.speed_a > 0:
        res.speedup = round(res.speed_a / res.speed_b, 2)
    for tag in ("A", "B"):
        out = res.result_a if tag == "A" else res.result_b
        ok = res.side_a_ok if tag == "A" else res.side_b_ok
        if out is not None and ok:
            try:
                scores = ev.evaluate_all(model_id, out.metrics)
                agg = ev.aggregate_quality(scores)
                if tag == "A":
                    res.quality_a = round(agg.overall, 3)
                else:
                    res.quality_b = round(agg.overall, 3)
            except Exception:
                pass

    if res.side_a_ok and res.side_b_ok:
        res.speed_winner = _band_winner(res.speed_a, res.speed_b)
        if res.quality_a is not None and res.quality_b is not None:
            res.quality_winner = _band_winner(res.quality_a, res.quality_b, band=0.01)
        if res.speed_winner == "draw" and (res.quality_winner in (None, "draw")):
            res.verdict = "draw"
        elif res.speed_winner in ("A", "B"):
            res.verdict = f"{res.speed_winner} faster"
            if res.quality_winner in ("A", "B") and res.quality_winner != res.speed_winner:
                res.verdict += f", {res.quality_winner} more correct (split)"
        else:
            res.verdict = "draw"
    elif res.side_a_ok or res.side_b_ok:
        keeper = "A" if res.side_a_ok else "B"
        res.verdict = f"partial ({keeper}-only; other side failed to load)"
    else:
        res.verdict = "failed (both sides failed)"
    logger.info("Compare complete", model=model_id, verdict=res.verdict)
    return res
