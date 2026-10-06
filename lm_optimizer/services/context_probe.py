"""Context needle probe: short speed check + 90%-fill recall gate.

One probe = one load of the best config at ctx N, one short benchmark
chat, one needle recall chat, unload in finally. ok requires load ok
AND tok_s >= min_speed AND recall >= min_recall. recall = hits/asked
(case-insensitive substring over asked facts only).

Filler sizing uses a 4 chars/token heuristic; the measured
prompt_tokens from the needle response is reported alongside the 90%
target (never claimed exact).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

# Local planted facts (ASCII-only). Deliberately NOT shared with other
# branches: distinctive nouns + numbers survive substring matching.
NEEDLE_FACTS: list[str] = [
    "The harbor lighthouse flashes green every eleven seconds.",
    "Marisol keeps a brass compass engraved with the number 4721.",
    "The midnight freight train departs platform nine at 3:40 AM.",
    "A copper plaque in the library honors gardener Tobias Wren.",
    "The recipe calls for exactly seven drops of juniper oil.",
]

SPEED_PROMPT = "Explain in one short sentence what a hash table is."
SPEED_MAX_TOKENS = 64
SPEED_TEMPERATURE = 0.1
NEEDLE_MAX_TOKENS = 256
NEEDLE_TEMPERATURE = 0.0
CHARS_PER_TOKEN = 4
DEFAULT_FILL_RATIO = 0.9

NEEDLE_INSTRUCTION = (
    "Read the document below and repeat the five key sentences "
    "verbatim, one per line.\n\nDocument:\n{doc}\n\n"
    "Now list the five key sentences verbatim, one per line:"
)


class NoBestConfigError(ValueError):
    """Refusal: no stored best config for this model (run optimize first)."""


def resolve_best_for_sweep(model_id: str, repo=None) -> dict:
    """Stored best config for `model_id` from the run DB.

    Mirrors the `manual_memory_duel._best_from_db` pattern: newest runs
    first (`get_by_model`), full run via `get`, measured best via
    `get_best_config`, zero-speed rows skipped. Raises NoBestConfigError
    naming the model when nothing measurable exists; never a silent default.
    """
    if repo is None:
        from lm_optimizer.database.repositories import run_repo as repo
    try:
        lean = repo.get_by_model(model_id, 50)
    except Exception as e:
        raise NoBestConfigError(
            f"no stored best config for '{model_id}': run DB lookup failed "
            f"({e}); run optimize first"
        ) from e
    for row in lean:
        try:
            full = repo.get(str(row.id)) or row
            best = full.get_best_config()
        except Exception:
            continue
        if best is None:
            continue
        gen = best.get_avg_generation_tok_s()
        if gen <= 0:
            continue
        return {
            "model_id": model_id,
            "gen_tok_s": gen,
            "ttft_ms": best.get_avg_estimated_ttft_ms(),
            "context_length": best.context_length,
            "load_config": best.config,
            "style": (best.generation or {}).get("style", "balanced"),
            "source": "run-db",
            "run_id": str(full.id),
        }
    raise NoBestConfigError(
        f"no stored best config for '{model_id}': run optimize first"
    )

_FILLER_SENTENCES = (
    "The quarterly inventory lists grain shipments by weight and origin.",
    "Clerks recorded rainfall totals in the margin of each ledger page.",
    "A revised timetable for river crossings took effect in early spring.",
    "The committee approved funds for roof repairs before the frost.",
    "Surveyors mapped the northern road with chains and stone markers.",
    "The orchard yielded a second harvest after the late rain.",
    "Messengers carried sealed orders between the depot and the mill.",
    "The archive holds tax rolls dating back forty harvests.",
    "Carpenters replaced the dock pilings damaged by ice.",
    "The council published market prices every second Tuesday.",
    "Shepherds moved the flocks to high pasture at first thaw.",
    "The miller logged water levels each morning at sunrise.",
)


def build_filler(target_chars: int) -> str:
    """Scaled synthetic doc (~target_chars) with all NEEDLE_FACTS planted."""
    facts_len = sum(len(f) for f in NEEDLE_FACTS) + len(NEEDLE_FACTS)
    padding_needed = max(0, int(target_chars) - facts_len)
    padding = ""
    i = 0
    while len(padding) < padding_needed:
        padding += _FILLER_SENTENCES[i % len(_FILLER_SENTENCES)] + "\n"
        i += 1
    # Split padding into K+1 buckets so facts spread evenly through the doc.
    buckets: list[str] = [""] * (len(NEEDLE_FACTS) + 1)
    for j, line in enumerate(padding.splitlines(keepends=True)):
        buckets[j % len(buckets)] += line
    parts: list[str] = []
    for j, fact in enumerate(NEEDLE_FACTS):
        parts.append(buckets[j])
        parts.append(fact + "\n")
    parts.append(buckets[-1])
    return "".join(parts)


def _extract(response: dict) -> tuple[dict, str, dict]:
    usage = response.get("usage", {}) or {}
    choices = response.get("choices", []) or []
    choice = choices[0] if choices else {}
    text = choice.get("message", {}).get("content", "") or ""
    stats = response.get("_stats", {}) or {}
    return usage, text, stats


def _thinking(response: dict) -> str:
    """Reasoning trace, if the backend returned one ("" otherwise)."""
    try:
        return response.get("thinking_text", "") or ""
    except Exception:
        return ""


def _speeds(usage: dict, stats: dict, wall_ms: float) -> tuple[float, float, float]:
    """(tok_s, prompt_tok_s, ttft_ms) mirroring benchmark case math."""
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    server_tok_s = float(stats.get("tokens_per_second", 0) or 0)
    if server_tok_s > 0 and completion_tokens > 0:
        tok_s = server_tok_s
        generation_ms = completion_tokens / tok_s * 1000
        prompt_ms = max(0.0, wall_ms - generation_ms)
    else:
        total = prompt_tokens + completion_tokens
        prompt_ms = wall_ms * (prompt_tokens / max(total, 1))
        generation_ms = wall_ms - prompt_ms
        tok_s = completion_tokens / (generation_ms / 1000) if generation_ms > 0 else 0.0
    prompt_tok_s = prompt_tokens / (prompt_ms / 1000) if prompt_ms > 0 else 0.0
    ttft_s = float(stats.get("time_to_first_token_seconds", 0) or 0)
    if ttft_s > 0:
        ttft_ms = ttft_s * 1000
    elif prompt_tokens > 0:
        ttft_ms = wall_ms * 0.1
    else:
        ttft_ms = wall_ms
    return tok_s, prompt_tok_s, ttft_ms


async def run_needle(
    client: Any,
    model_id: str,
    load_config: Any,
    ctx: int,
    fill_ratio: float = DEFAULT_FILL_RATIO,
) -> dict:
    """Needle recall against the already-loaded model (no load/unload here).

    load_config is accepted for interface symmetry; filler size derives
    from ctx. Returns {recall, asked, fill_chars, prompt_tokens,
    output_text, thinking_text, prompt} with recall = hits/asked over
    asked facts only. output_text/thinking_text are the verbatim needle
    response (and reasoning trace, if any); prompt is the needle prompt
    sent; prompt_tokens is the measured server-reported value.
    """
    _ = load_config
    target_chars = int(ctx * fill_ratio * CHARS_PER_TOKEN)
    filler = build_filler(target_chars)
    prompt = NEEDLE_INSTRUCTION.format(doc=filler)
    resp = await client.chat_completion(
        model=model_id,
        input_text=prompt,
        temperature=NEEDLE_TEMPERATURE,
        max_output_tokens=NEEDLE_MAX_TOKENS,
    )
    usage, text, _stats = _extract(resp)
    lowered = text.lower()
    asked = len(NEEDLE_FACTS)
    hits = sum(1 for f in NEEDLE_FACTS if f.lower() in lowered)
    return {
        "recall": (hits / asked) if asked else 0.0,
        "asked": asked,
        "fill_chars": len(filler),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_text": text,
        "thinking_text": _thinking(resp),
        "prompt": prompt,
    }


async def probe_context(
    client: Any,
    model_id: str,
    load_config: Any,
    ctx: int,
    min_speed: float,
    min_recall: float,
    skip_needle: bool = False,
) -> dict:
    """Load best config at ctx, run speed + needle gates, always unload.

    Returns {ctx, ok, tok_s, prompt_tok_s, ttft_ms, recall, skipped_needle,
    error, output_text, thinking_text, prompt, prompt_tokens, fill_chars,
    speed_output, speed_thinking_text}. ok requires load ok AND
    tok_s >= min_speed AND recall >= min_recall. output_text/thinking_text
    are the verbatim needle response (and reasoning trace, if any);
    prompt is the needle prompt sent; prompt_tokens is the measured
    server-reported value; speed_* are the verbatim speed-chat response.
    Keys default to ""/0 on paths where that chat never ran, so the
    report can always render real texts instead of placeholders.
    With skip_needle=True the 90%-fill needle chat is not sent
    (recall reported 1.0 with skipped_needle True; speed gate still applies).
    """

    skipped = bool(skip_needle)

    def _fail(
        tok_s: float,
        prompt_tok_s: float,
        ttft_ms: float,
        recall: float,
        error: str,
        extra: dict | None = None,
    ) -> dict:
        out = {
            "ctx": ctx,
            "ok": False,
            "tok_s": tok_s,
            "prompt_tok_s": prompt_tok_s,
            "ttft_ms": ttft_ms,
            "recall": recall,
            "skipped_needle": skipped,
            "error": error,
            "output_text": "",
            "thinking_text": "",
            "prompt": "",
            "prompt_tokens": 0,
            "fill_chars": 0,
            "speed_output": "",
            "speed_thinking_text": "",
        }
        if extra:
            out.update(extra)
        return out

    try:
        try:
            probe_cfg = dataclasses.replace(load_config, context_length=ctx)
        except Exception as e:
            return _fail(0.0, 0.0, 0.0, 0.0, f"config failed: {e}")
        try:
            res = await client.load_model(model_id, probe_cfg)
        except Exception as e:
            return _fail(0.0, 0.0, 0.0, 0.0, f"load failed: {type(e).__name__}: {e}")
        if not getattr(res, "success", False):
            return _fail(
                0.0, 0.0, 0.0, 0.0, f"load failed: {getattr(res, 'error', 'unknown')}"
            )
        start = time.perf_counter()
        try:
            resp = await client.chat_completion(
                model=model_id,
                input_text=SPEED_PROMPT,
                temperature=SPEED_TEMPERATURE,
                max_output_tokens=SPEED_MAX_TOKENS,
            )
        except Exception as e:
            return _fail(
                0.0, 0.0, 0.0, 0.0, f"generation failed: {type(e).__name__}: {e}"
            )
        wall_ms = (time.perf_counter() - start) * 1000
        usage, speed_text, stats = _extract(resp)
        speed_thinking = _thinking(resp)
        speed_extra = {
            "speed_output": speed_text,
            "speed_thinking_text": speed_thinking,
        }
        tok_s, prompt_tok_s, ttft_ms = _speeds(usage, stats, wall_ms)
        if tok_s < min_speed:
            return _fail(
                tok_s,
                prompt_tok_s,
                ttft_ms,
                0.0,
                f"speed {tok_s:.1f} tok/s below floor {min_speed:.1f} tok/s",
                extra=speed_extra,
            )
        if skipped:
            return {
                "ctx": ctx,
                "ok": True,
                "tok_s": tok_s,
                "prompt_tok_s": prompt_tok_s,
                "ttft_ms": ttft_ms,
                "recall": 1.0,
                "skipped_needle": True,
                "error": "",
                "output_text": "",
                "thinking_text": "",
                "prompt": "",
                "prompt_tokens": 0,
                "fill_chars": 0,
                **speed_extra,
            }
        try:
            needle = await run_needle(client, model_id, probe_cfg, ctx)
        except Exception as e:
            return _fail(
                tok_s,
                prompt_tok_s,
                ttft_ms,
                0.0,
                f"needle failed: {type(e).__name__}: {e}",
                extra=speed_extra,
            )
        recall = float(needle["recall"])
        asked = int(needle["asked"])
        needle_extra = {
            "output_text": needle.get("output_text", ""),
            "thinking_text": needle.get("thinking_text", ""),
            "prompt": needle.get("prompt", ""),
            "prompt_tokens": needle.get("prompt_tokens", 0),
            "fill_chars": needle.get("fill_chars", 0),
            **speed_extra,
        }
        if recall < min_recall:
            hits = int(round(recall * asked))
            return _fail(
                tok_s,
                prompt_tok_s,
                ttft_ms,
                recall,
                f"recall {recall:.2f} ({hits}/{asked}) below floor {min_recall:.2f}",
                extra=needle_extra,
            )
        return {
            "ctx": ctx,
            "ok": True,
            "tok_s": tok_s,
            "prompt_tok_s": prompt_tok_s,
            "ttft_ms": ttft_ms,
            "recall": recall,
            "skipped_needle": skipped,
            "error": "",
            **needle_extra,
        }
    finally:
        try:
            await client.ensure_unloaded(model_id)
        except Exception:
            pass
