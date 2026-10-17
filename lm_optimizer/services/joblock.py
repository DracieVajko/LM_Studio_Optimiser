"""Cross-process optimizer job lock: at most one optimizer job at a time.

The lock file lives under the configured results dir
(results/.optimizer.lock, gitignored, created at runtime only).
Creation is atomic (O_CREAT|O_EXCL): two jobs starting in the same second
resolve to exactly one winner. A holder whose PID is dead (kill -9, crash
without finally) is treated as stale and taken over, never a permanent
block. Release removes only the caller's own lock (PID match) and is
idempotent.
"""

import errno
import json
import os
import time
from pathlib import Path
from typing import Any

from lm_optimizer.config import config
from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)

LOCK_NAME = ".optimizer.lock"
LOCK_PATH = config.storage.results_dir / LOCK_NAME


def _default_lock_path() -> Path:
    """Default lock path, resolved from config at call time (not import)."""
    return config.storage.results_dir / LOCK_NAME


class JobBusyError(Exception):
    """Raised when a live optimizer job already holds the lock.

    Carries .holder: the decoded lock content {pid, started, purpose}
    (empty dict when the existing lock was unreadable).
    """

    def __init__(self, message: str = "", holder: Any = None):
        super().__init__(message)
        self.holder = dict(holder) if isinstance(holder, dict) else {}


def _lock_path(lock_dir: str | Path | None = None) -> Path:
    """Resolve the lock file path (lock_dir overrides the configured dir)."""
    if lock_dir is not None:
        return Path(lock_dir) / LOCK_NAME
    return _default_lock_path()


def _self_create_time() -> float | None:
    """This process's creation time, when psutil can supply it."""
    try:
        import psutil  # noqa: PLC0415 - lazy import mirrors hostguard guard style
    except ImportError:
        return None
    try:
        return float(psutil.Process(os.getpid()).create_time())
    except Exception:
        return None


def _holder_alive(holder: Any) -> bool:
    """True when the lock holder PID is (or may be) a live process.

    PID alone is not enough: after a crash the PID may be reused by an
    unrelated process, which would make a stale lock look live forever. When
    the lock records the holder's creation time we compare it, so a reused PID
    is correctly reported as dead and the stale lock is reclaimed.

    Without liveness information we fail closed (the lock blocks) rather than
    risking two concurrent optimizer jobs on one host.
    """
    pid = holder.get("pid") if isinstance(holder, dict) else None
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        import psutil  # noqa: PLC0415 - lazy import mirrors hostguard guard style
    except ImportError:
        return True  # fail-closed: without liveness info a lock blocks
    try:
        if not psutil.pid_exists(pid):
            return False
        recorded = holder.get("create_time")
        if isinstance(recorded, (int, float)):
            # Same PID, different process -> the original holder is gone.
            return abs(float(psutil.Process(pid).create_time()) - float(recorded)) < 2.0
        return True
    except Exception:
        return True


def _read_holder(path: Path) -> dict | None:
    """Decoded lock content, None when no lock file, {} when unreadable."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def acquire_lock(purpose: str = "", lock_dir: str | Path | None = None) -> dict:
    """Atomically claim the optimizer job lock for this process.

    Returns the written record {pid, started, purpose}. Raises JobBusyError
    (with .holder) when a live holder exists; takes over stale locks whose
    PID is dead. A lost create race after a stale unlink re-reads the winner
    and refuses instead of double-running.
    """
    path = _lock_path(lock_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "pid": os.getpid(),
        "started": time.time(),
        "purpose": purpose,
        # Lets a later reader tell "this PID is still my process" from
        # "this PID now belongs to someone else" (PID reuse).
        "create_time": _self_create_time(),
    }
    payload = json.dumps(record).encode("utf-8")
    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            holder = _read_holder(path)
            if holder is None:
                continue  # raced with a release; retry the create
            if _holder_alive(holder):
                raise JobBusyError(
                    f"Optimizer job already running (pid {holder.get('pid')}, "
                    f"purpose {holder.get('purpose')!r}).",
                    holder=holder,
                ) from None
            logger.warning("Taking over stale optimizer lock",
                           holder=holder, purpose=purpose)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue  # retry the atomic create after stale unlink
        else:
            try:
                os.write(fd, payload)
            finally:
                os.close(fd)
            return record
    holder = _read_holder(path) or {}
    raise JobBusyError(
        f"Optimizer job already running (pid {holder.get('pid')}).",
        holder=holder,
    )


def release_lock(lock_dir: str | Path | None = None) -> None:
    """Release the lock when held by this process; idempotent.

    Never removes a foreign lock (PID mismatch) or a lock file that is
    unreadable; missing file is a no-op.
    """
    path = _lock_path(lock_dir)
    holder = _read_holder(path)
    if holder is None:
        return
    if not isinstance(holder, dict) or holder.get("pid") != os.getpid():
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass
