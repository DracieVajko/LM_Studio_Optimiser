"""Ollama endpoint control probe (live server, no optimization).

Exercises the full control surface against one model and returns a
structured report: connect, list/show details, generate with thinking
off (with exactly ONE automatic retry with thinking on when the model
answers empty despite spending tokens — the gpt-oss family requires
reasoning), a small num_ctx timing pair, and unload verification via
``GET /api/ps``.

Async functions take an injected client, so unit tests drive them with
a mocked transport (no server needed).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lm_optimizer.backends.ollama.client import OllamaClient

PROBE_PROMPT = "Reply with exactly: OK"
PROBE_MAX_TOKENS = 16
PROBE_THINK_TOKENS = 64


async def _timed_generate(
    client: OllamaClient, model: str, options: dict, think: bool = False
) -> dict[str, Any]:
    """One tiny non-streaming generate with wall time + normalized metrics."""
    wall_start = time.perf_counter()
    try:
        m = await client.generate(model, PROBE_PROMPT, options=dict(options), think=think)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    return {
        "ok": True,
        "text": (m.text or "")[:120],
        "empty": not (m.text or "").strip(),
        "eval_count": m.eval_count,
        "gen_tok_s": round(m.gen_tok_s, 1),
        "load_ms": round(m.load_ms, 1),
        "done_reason": m.done_reason,
        "wall_s": round(time.perf_counter() - wall_start, 1),
    }


async def probe_model(
    client: OllamaClient,
    model: str,
    ctx_pair: tuple[int, int] = (2048, 8192),
    num_predict: int = PROBE_MAX_TOKENS,
) -> dict[str, Any]:
    """Run the endpoint control matrix for one model. Never raises."""
    report: dict[str, Any] = {"model": model, "steps": {}}
    steps = report["steps"]

    try:
        await client.connect(echo_probe=False)
        steps["connect"] = {"ok": True, "version": client.capabilities.version}
    except Exception as e:
        steps["connect"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
        return report

    try:
        models = await client.list_models()
        steps["list"] = {"ok": True, "present": any(m.id == model for m in models)}
    except Exception as e:
        steps["list"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
        return report

    try:
        show = await client.show(model)
        details = show.get("details", {}) if isinstance(show, dict) else {}
        steps["show"] = {
            "ok": True,
            "context_length": details.get("context_length"),
            "family": details.get("family"),
            "quantization": details.get("quantization_level"),
            "parameter_size": details.get("parameter_size"),
        }
    except Exception as e:
        steps["show"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
        return report

    base_options = {"num_predict": num_predict, "temperature": 0.0}
    off = await _timed_generate(client, model, {**base_options, "num_ctx": ctx_pair[0]}, think=False)
    steps["generate_think_off"] = off
    # Exactly ONE automatic retry with thinking on: empty text despite
    # spent tokens means the family answers through the reasoning channel.
    # Reasoning needs a bigger token budget (thinking alone can eat 16).
    needs_think = bool(
        off.get("ok") and off.get("empty") and (off.get("eval_count") or 0) > 0
    )
    if needs_think:
        on = await _timed_generate(
            client, model,
            {"num_predict": PROBE_THINK_TOKENS, "temperature": 0.0, "num_ctx": ctx_pair[0]},
            think=True,
        )
        steps["generate_think_on_retry"] = on
    else:
        steps["generate_think_on_retry"] = {"ok": True, "skipped": True}

    timings: dict[str, Any] = {}
    for ctx_len in ctx_pair[1:]:
        timings[str(ctx_len)] = await _timed_generate(
            client, model,
            {
                "num_predict": PROBE_THINK_TOKENS if needs_think else num_predict,
                "temperature": 0.0,
                "num_ctx": ctx_len,
            },
            think=needs_think,
        )
    steps["ctx_timings"] = timings

    try:
        unloaded = await client.unload(model)
        try:
            running = await client.refresh_running_models()
        except Exception:
            running = set()
        steps["unload_verify"] = {
            "ok": bool(unloaded) and model not in running,
            "still_loaded": sorted(running),
        }
    except Exception as e:
        steps["unload_verify"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    return report
