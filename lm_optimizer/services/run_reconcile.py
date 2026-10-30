"""Non-destructive reconciliation of stale ``running`` runs.

A run with status ``running``/``resumed`` whose checkpoint is missing or older
than the stale window belongs to a dead process. This service marks such runs
``interrupted`` — idempotently, keeping all candidate results, reports and the
checkpoint file. It never touches a run that has a fresh checkpoint or the
currently-live run.

Preview first (``dry_run=True`` default); apply explicitly. Existing production
records are never rewritten without an explicit apply.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable

# A running/resumed run with no checkpoint newer than this is dead.
STALE_AFTER_SECONDS = 2 * 3600
LIVE_STATES = ("running", "resumed")


def _default_is_fresh(run_id: str, now: float) -> bool:
    """True when the run has a checkpoint file updated within the window."""
    try:
        from lm_optimizer.storage.run_checkpoint import checkpoint_path

        p = checkpoint_path(str(run_id))
        return p.exists() and (now - p.stat().st_mtime) <= STALE_AFTER_SECONDS
    except Exception:
        return False


def reconcile_stale_runs(
    dry_run: bool = True,
    current_run_id: Any = None,
    model_id: str | None = None,
    is_fresh: Callable[[str, float], bool] | None = None,
    now: float | None = None,
) -> dict:
    """Preview/apply stale-running reconciliation.

    Returns ``{"dry_run", "changed", "changes": [{"id","from","to","reason"}]}``.
    Idempotent: a second run reports zero changes.
    """
    from lm_optimizer.database.repositories import run_repo
    from lm_optimizer.domain.models import RunStatus

    now = time.time() if now is None else now
    fresh = is_fresh or _default_is_fresh
    changes: list[dict] = []
    try:
        runs = run_repo.list_all(limit=1000)
    except Exception:
        runs = []
    for r in runs:
        st = str(getattr(getattr(r, "status", None), "value", getattr(r, "status", "")))
        if st not in LIVE_STATES:
            continue
        rid = str(getattr(r, "id", ""))
        if current_run_id is not None and rid == str(current_run_id):
            continue
        # Optional scoping: only reconcile stale runs for a specific model.
        if model_id is not None:
            rmid = getattr(getattr(r, "model", None), "id", None)
            if rmid != model_id:
                continue
        if fresh(rid, now):
            continue
        entry = {
            "id": rid,
            "model_id": getattr(getattr(r, "model", None), "id", None),
            "from": st,
            "to": RunStatus.INTERRUPTED.value,
            "reason": "stale: no fresh checkpoint (process gone)",
        }
        changes.append(entry)
        if not dry_run:
            try:
                r.status = RunStatus.INTERRUPTED
                r.completed_at = datetime.now()
                note = "reconciled: stale running, process gone, no fresh checkpoint"
                r.error = f"{r.error}; {note}" if getattr(r, "error", None) else note
                run_repo.save(r)
            except Exception:
                # Never abort the whole reconcile for one bad row.
                continue
    return {"dry_run": bool(dry_run), "changed": len(changes), "changes": changes}
