"""Opt-in sampling sweep: top_p/top_k × precision/chat/creative.

Unlike the main pipeline (which freezes sampling for comparability), this
stage deliberately varies generation sampling and re-validates quality per
combo with the full suite. Values come from generation_profiles: publisher
numbers where known, ours (suite temps, server-default top_p/top_k)
otherwise. Combos identical to each other are never re-run.
"""

from lm_optimizer.domain.models import BenchmarkCase, LoadConfiguration
from lm_optimizer.logging_config import get_logger
from lm_optimizer.services.generation_profiles import profiles_for

logger = get_logger(__name__)


def sampling_combos(model_id: str, architecture: str | None = None) -> list[dict]:
    """Three profile combos (precision/chat/creative), deduplicated."""
    profiles = profiles_for(model_id, architecture)
    combos: list[dict] = []
    seen: set = set()
    for name in ("precision", "chat", "creative"):
        prof = profiles[name]
        pub = prof.get("publisher") or {}
        ours = prof.get("ours") or {}
        combo = {
            "profile": name,
            "temperature": pub.get("temperature", ours.get("temperature")),
            "top_p": pub.get("top_p"),
            "top_k": pub.get("top_k"),
            "source": "publisher" if pub.get("source") else "ours",
        }
        key = (combo["temperature"], combo["top_p"], combo["top_k"])
        if key in seen:
            continue
        seen.add(key)
        combos.append(combo)
    return combos


def cases_for_combo(base_cases: list[BenchmarkCase], combo: dict) -> list[BenchmarkCase]:
    """Suite cases with the combo's sampling applied."""
    out = []
    for c in base_cases:
        out.append(BenchmarkCase(
            name=c.name,
            category=c.category,
            prompt=c.prompt,
            max_tokens=c.max_tokens,
            temperature=combo["temperature"]
            if combo["temperature"] is not None else c.temperature,
            top_p=combo["top_p"],
            top_k=combo["top_k"],
            stop_sequences=c.stop_sequences,
        ))
    return out


async def run_sampling_sweep(
    benchmark_service,
    quality_evaluator,
    model_id: str,
    load_config: LoadConfiguration,
    context_length: int,
    combos: list[dict] | None = None,
    architecture: str | None = None,
    repetitions: int = 1,
) -> dict:
    """Full suite + quality per sampling combo. Never raises fatally."""
    active = combos if combos is not None else sampling_combos(model_id, architecture)
    base = benchmark_service.create_cases_for_context(context_length)
    results: list[dict] = []
    for combo in active:
        entry: dict = {"profile": combo["profile"], **combo}
        try:
            out = await benchmark_service.run_cases(
                model_id, load_config, context_length,
                cases_for_combo(base, combo),
                repetitions=repetitions, warmup_repetitions=0,
            )
            try:
                gen = dict(out.generation or {})
                gen["phase"] = "sampling"
                gen["sampling_profile"] = combo["profile"]
                out.generation = gen
            except Exception:
                pass
            ok = out.status not in ("failed", "error") and not out.error
            entry["ok"] = bool(ok)
            entry["tok_s"] = round(out.get_avg_generation_tok_s(), 1) if ok else 0.0
            if ok:
                try:
                    scores = quality_evaluator.evaluate_all(model_id, out.metrics)
                    agg = quality_evaluator.aggregate_quality(scores)
                    entry["quality"] = round(agg.overall, 3)
                except Exception:
                    entry["quality"] = None
            else:
                entry["quality"] = None
                entry["error"] = (out.error or f"status={out.status}")[:200]
        except Exception as e:
            entry["ok"] = False
            entry["tok_s"] = 0.0
            entry["quality"] = None
            entry["error"] = f"{type(e).__name__}: {e}"[:200]
        results.append(entry)
    ok_rows = [r for r in results if r["ok"] and r["quality"] is not None]
    best = max(ok_rows, key=lambda r: (r["quality"], r["tok_s"]), default=None)
    logger.info("Sampling sweep complete", model=model_id,
                best=(best["profile"] if best else None))
    return {"model": model_id, "combos": results,
            "best_profile": best["profile"] if best else None}
