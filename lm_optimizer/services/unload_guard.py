"""Unload guard: fail-closed guarantee that no model stays loaded.

Used at every model boundary (auto/fit/ctx/matrix loops, sandbox duels,
compare, web optimize). A best-effort unload whose result nobody checks is
how two models end up sharing the GPU and silently invalidating speed
measurements — this guard turns that into a loud, early stop instead.
"""

import asyncio
import inspect

from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)


class UnloadNotClean(Exception):
    """Raised when loaded instances survive the unload sweep."""


async def assert_unloaded(client, purpose: str = "", attempts: int = 3,
                          backoff_s: float = 2.0) -> None:
    """Unload-all sweep + server-side verify. Raises UnloadNotClean.

    Never returns a false "clean": the verdict always comes from a fresh
    server listing, never from client-side caches.
    """
    # Test doubles without an *awaitable* client interface cannot be
    # verified: nothing to sweep, nothing to assert. Production clients
    # always implement async unload_all/get_loaded_instances.
    for _name in ("unload_all", "get_loaded_instances"):
        _fn = getattr(client, _name, None)
        if _fn is None or not inspect.iscoroutinefunction(_fn):
            logger.debug("Unload guard skipped (no async client interface)",
                         purpose=purpose)
            return
    last_leftovers: list = []
    for attempt in range(1, attempts + 1):
        try:
            await client.unload_all()
        except Exception as e:
            logger.warning("Unload sweep failed", purpose=purpose,
                           attempt=attempt, error=str(e)[:160])
        try:
            instances = await client.get_loaded_instances()
        except Exception as e:
            logger.warning("State check failed", purpose=purpose,
                           attempt=attempt, error=str(e)[:160])
            instances = None
        if instances is not None and not isinstance(instances, list):
            # Test double returning a placeholder: nothing verifiable.
            logger.debug("Unload guard skipped (non-list state)", purpose=purpose)
            return
        if instances is not None:
            last_leftovers = [i for i in (instances or []) if i.get("instance_id")]
            if not last_leftovers:
                return
        if attempt < attempts:
            await asyncio.sleep(backoff_s * attempt)
    stuck = ", ".join(
        f"{i.get('model')}:{i.get('instance_id')}" for i in last_leftovers[:5]
    ) or "unverifiable server state"
    raise UnloadNotClean(
        f"Host not clean before {purpose or 'next step'} "
        f"({attempts} unload sweeps, still loaded: {stuck}). "
        "If LM Studio has 'Keep Model in Memory' enabled, disable it for "
        "optimizer runs, or unload models manually in LM Studio and retry."
    )
