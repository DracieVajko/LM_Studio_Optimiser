"""Max-context ladder: escalate ctx until load refusal, per KV path.

Lightweight fit discovery for big models (no stage tuning): finds the maximum
loadable context on KV-GPU (speed path) and KV-CPU (capacity path). On a
refusal, the failure interval is refined with ~1024-token steps (a PASS/FAIL
pair never implies the PASS value is the maximum).
"""

from lm_optimizer.domain.models import LoadConfiguration
from lm_optimizer.logging_config import get_logger
from lm_optimizer.services.model_recommendations import diagnose_load_error

logger = get_logger(__name__)

REFINE_STEP = 1024
REFINE_MAX_PROBES = 3


async def _smoke_once(svc, model_id: str, ctx: int, kv: bool) -> dict:
    cfg = LoadConfiguration(
        context_length=ctx, flash_attention=True, offload_kv_cache_to_gpu=kv
    )
    ok, tok, err = await svc.smoke_test(model_id, cfg)
    return {"ok": ok, "tok_s": round(tok, 1), "error": (err or "")[:160]}


async def ladder(svc, model_id: str, contexts: list[int], kv_modes: list[bool]) -> dict:
    """Run the ctx ladder. Returns {model, paths, ceilings, recommended}.

    paths[True] = KV-GPU rungs, paths[False] = KV-CPU rungs:
    {ctx: {ok, tok_s, error}}. ceilings[kv] = max passing ctx (or None).
    recommended[kv] = max stable ctx (returned alongside the failure
    boundary so they can differ by policy later).
    Refusal (oom/load-fail) stops that path; transient errors retry once.
    OOM refusals trigger interval refinement (~1024 steps, max 3 probes).
    """
    out: dict = {"model": model_id, "paths": {}, "ceilings": {}, "recommended": {}}
    for kv in kv_modes:
        path: dict = {}
        last_pass: int | None = None
        failed_at: int | None = None
        failed_err = ""
        for ctx in sorted(contexts):
            result = await _smoke_once(svc, model_id, ctx, kv)
            if not result["ok"]:
                diag = diagnose_load_error(result["error"])
                if diag["category"] == "transient":
                    logger.info(
                        "Transient ladder failure, retrying once", model=model_id, ctx=ctx
                    )
                    result = await _smoke_once(svc, model_id, ctx, kv)
            path[ctx] = result
            if not result["ok"]:
                failed_at, failed_err = ctx, result["error"]
                break
            last_pass = ctx
        if failed_at is not None and last_pass is not None:
            diag = diagnose_load_error(failed_err)
            if diag["category"] in ("oom", "load_refused"):
                probes = 0
                candidate = failed_at - REFINE_STEP
                while candidate > last_pass and probes < REFINE_MAX_PROBES:
                    result = await _smoke_once(svc, model_id, candidate, kv)
                    path[candidate] = result
                    probes += 1
                    if result["ok"]:
                        last_pass = candidate
                    candidate -= REFINE_STEP
        out["paths"][kv] = path
        passed = [c for c, r in path.items() if r["ok"]]
        ceiling = max(passed) if passed else None
        out["ceilings"][kv] = ceiling
        out["recommended"][kv] = ceiling
    return out
