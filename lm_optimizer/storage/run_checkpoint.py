"""Interrupt-safe per-run checkpointing for the AdaptiveOptimizer.

Persists the full P1 state atomically (tmp file + rename) BEFORE a
cancellation is acknowledged, so completed benchmark results are never lost
on Cancel / Ctrl+C / shutdown / browser-triggered cancellation.

Checkpoint file: <results_dir>/checkpoints/run_<run_id>.json
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import orjson

from lm_optimizer.config import config
from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)

CHECKPOINT_VERSION = 1

# Retry policy for transient Windows replace failures (WinError 5 when the
# destination is briefly held by a scanner/sync tool or a racing writer).
_REPLACE_ATTEMPTS = 5
_REPLACE_BACKOFF_S = (0.05, 0.1, 0.2, 0.4, 0.8)
_STALE_TEMP_MAX_AGE_S = 24 * 3600

# Per-run write serialization: no simultaneous writers to one checkpoint.
_locks_guard = threading.Lock()
_run_locks: dict[str, threading.Lock] = {}


def _lock_for(run_id: str) -> threading.Lock:
    """Per-run mutex (process-wide). Never blocks longer than one write."""
    with _locks_guard:
        lock = _run_locks.get(run_id)
        if lock is None:
            lock = threading.Lock()
            _run_locks[run_id] = lock
        return lock


def checkpoint_dir() -> Path:
    d = config.storage.results_dir / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_path(run_id: str | UUID) -> Path:
    return checkpoint_dir() / f"run_{run_id}.json"


def _safe(v: Any) -> Any:
    """Make datetimes/UUIDs/sets JSON-safe."""
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, UUID):
        return str(v)
    if isinstance(v, set):
        return sorted(v, key=str)
    if isinstance(v, Path):
        return str(v)
    return v


def build_checkpoint_payload(state: Any, reason: str = "periodic") -> dict:
    """Build the full P1 checkpoint dict from an OptimizationState."""
    run = state.run
    tested = list(getattr(state, "tested_configs", []))
    failed_cfgs = [c for c in tested if getattr(c.status, "value", c.status) != "passed"]
    passed_cfgs = [c for c in tested if getattr(c.status, "value", c.status) == "passed"]

    def cfg_id(c: Any) -> str:
        return str(getattr(c, "id", ""))

    def cfg_summary(c: Any) -> dict:
        q = getattr(c, "quality_score", None)
        return {
            "id": cfg_id(c),
            "context_length": getattr(c, "context_length", None),
            "status": getattr(getattr(c, "status", ""), "value", str(getattr(c, "status", ""))),
            "error": getattr(c, "error", None),
            "score": getattr(c, "score", 0.0),
            "gen_tok_s": c.get_avg_generation_tok_s()
            if hasattr(c, "get_avg_generation_tok_s")
            else 0,
            "config": c.config.to_dict() if hasattr(c.config, "to_dict") else {},
        }

    # OOM + context search boundaries derived from tested configs + failed regions.
    oom_boundaries: list[dict] = []
    for c in tested:
        st = getattr(getattr(c, "status", ""), "value", str(getattr(c, "status", "")))
        if st == "oom":
            oom_boundaries.append(
                {"ctx": getattr(c, "context_length", None), "config": cfg_summary(c)}
            )
    ctx_tested = sorted({getattr(c, "context_length", 0) for c in tested})
    ctx_pass = sorted({getattr(c, "context_length", 0) for c in passed_cfgs})

    best = getattr(state, "best_config", None)
    hw = getattr(run, "hardware", None)

    def hw_snapshot(h: Any) -> dict | None:
        if h is None:
            return None
        try:
            return {
                "os": getattr(h, "os", ""),
                "cpu_name": getattr(h, "cpu_name", ""),
                "cpu_cores_physical": getattr(h, "cpu_cores_physical", 0),
                "cpu_cores_logical": getattr(h, "cpu_cores_logical", 0),
                "total_ram_gb": getattr(h, "total_ram_gb", 0),
                "gpu_count": getattr(h, "gpu_count", 0),
                "gpus": [
                    {"index": g.index, "name": g.name, "vram_gb": g.vram_gb, "vendor": g.vendor}
                    for g in (getattr(h, "gpus", []) or [])
                ],
            }
        except Exception:
            return None

    payload = {
        "version": CHECKPOINT_VERSION,
        "reason": reason,
        "saved_at": datetime.now().isoformat(),
        "run_id": str(getattr(run, "id", "")),
        "model": getattr(getattr(run, "model", None), "id", ""),
        "model_name": getattr(getattr(run, "model", None), "name", ""),
        "model_context_limit": getattr(getattr(run, "model", None), "context_limit", None),
        "profile": getattr(getattr(run, "profile", ""), "value", str(getattr(run, "profile", ""))),
        "stage": getattr(getattr(run, "stage", ""), "value", str(getattr(run, "stage", ""))),
        "status": getattr(getattr(run, "status", ""), "value", str(getattr(run, "status", ""))),
        "current_candidate": str(getattr(state, "current_config_id", "") or ""),
        "completed_candidate_ids": [cfg_id(c) for c in tested],
        "successful_results": [cfg_summary(c) for c in passed_cfgs],
        "failed_configurations": [cfg_summary(c) for c in failed_cfgs],
        "oom_boundaries": oom_boundaries,
        "context_search_boundaries": {
            "tested_contexts": ctx_tested,
            "passing_contexts": ctx_pass,
            "failed_regions": sorted(
                [list(r) for r in getattr(state, "failed_regions", set())], key=str
            ),
        },
        "best_candidates": [cfg_summary(best)] if best is not None else [],
        "recommendation_candidates": [cfg_id(c) for c in getattr(state, "pareto_frontier", [])],
        "benchmark_params": getattr(run, "benchmark_params", {}) or {},
        "quality_threshold": getattr(run, "quality_threshold", None),
        "profile_weights": getattr(run, "profile_weights", {}) or {},
        "hardware_snapshot": hw_snapshot(hw),
        "search_space": getattr(run, "search_space", {}) or {},
        "errors": list(getattr(state, "errors", []))[-50:],
        "event_log": list(getattr(state, "event_log", []))[-200:],
        "flash_required": bool(getattr(state, "flash_required", False)),
        "phase_ab": {
            "phase": getattr(state, "phase", "speed"),
            "speed_context": getattr(state, "speed_context", None),
            "quality_context": getattr(state, "quality_context", None),
            "budget_class": getattr(state, "budget_class", None),
            "probe_repetitions": getattr(state, "probe_repetitions", 1),
            "baseline_config": _safe(getattr(state.baseline_result, "config", None).to_dict())
            if getattr(state, "baseline_result", None) is not None
            and getattr(state.baseline_result, "config", None) is not None
            else None,
            "speed_finalist_ids": list(getattr(state, "speed_finalists", []) or []),
            "raw_fastest_id": str(getattr(state.raw_fastest, "id", ""))
            if getattr(state, "raw_fastest", None) is not None
            else None,
            "recovery_log": list(getattr(state, "recovery_log", []) or []),
            "recovery_cursor": dict(getattr(state, "recovery_cursor", None) or {}),
            "probe_attempts": dict(getattr(state, "probe_attempts", None) or {}),
            "checkpoint_seq": int(getattr(state, "checkpoint_seq", 0) or 0),
            "pause": (
                (getattr(getattr(state, "run", None), "benchmark_params", {}) or {}).get(
                    "phase_ab", {}
                )
                or {}
            ).get("pause"),
            "excluded_notes": list(getattr(state, "excluded_notes", []) or []),
            "phase_b_enabled": bool(getattr(state, "phase_b_enabled", False)),
            "phase_b_result": getattr(state, "phase_b_result", None),
        },
        "optimizer_state": {
            "stage": getattr(getattr(run, "stage", ""), "value", ""),
            "should_cancel": bool(getattr(state, "should_cancel", False)),
            "should_pause": bool(getattr(state, "should_pause", False)),
        },
    }
    return payload


def _cleanup_stale_temps(directory: Path, keep_prefix: str) -> None:
    """Remove orphaned temp files from crashed/interrupted writers."""
    try:
        now = time.time()
        for tmp in directory.glob(f".{keep_prefix}.*.tmp"):
            try:
                if now - tmp.stat().st_mtime > _STALE_TEMP_MAX_AGE_S:
                    tmp.unlink()
            except OSError:
                pass
    except OSError:
        pass


def save_checkpoint(state: Any, reason: str = "periodic") -> Path | None:
    """Write checkpoint atomically. Returns path or None on failure.

    Windows correctness: per-run serialization, unique temp name (pid +
    counter-safe random suffix), closed handle before replace, bounded retry
    on transient PermissionError (WinError 5), read-back verification, and
    the previous valid checkpoint is retained whenever replacement fails.
    """
    try:
        run_id = str(state.run.id)
        path = checkpoint_path(run_id)
    except Exception as e:
        logger.error("CHECKPOINT_SAVE_FAILED", error=f"bad state: {e}")
        return None

    with _lock_for(run_id):
        try:
            payload = build_checkpoint_payload(state, reason=reason)
            data = orjson.dumps(payload, option=orjson.OPT_INDENT_2)
        except Exception as e:
            logger.error("CHECKPOINT_SAVE_FAILED", error=f"serialize: {e}", run_id=run_id)
            return None

        tmp_path: Path | None = None
        try:
            previous_bytes: bytes | None = None
            if path.exists():
                try:
                    previous_bytes = path.read_bytes()
                except OSError:
                    previous_bytes = None
            tmp_fd, tmp_name = tempfile.mkstemp(
                dir=str(path.parent),
                prefix=f".{path.name}.{os.getpid()}.",
                suffix=".tmp",
            )
            tmp_path = Path(tmp_name)
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())

            last_error: Exception | None = None
            for attempt, backoff in enumerate(_REPLACE_BACKOFF_S):
                try:
                    tmp_path.replace(path)
                    last_error = None
                    break
                except PermissionError as e:
                    last_error = e
                    time.sleep(backoff)
                except OSError as e:
                    last_error = e
                    break
            if last_error is not None:
                logger.error(
                    "CHECKPOINT_SAVE_FAILED",
                    error=f"{type(last_error).__name__}: {last_error} "
                    f"(attempts={len(_REPLACE_BACKOFF_S)}); previous kept",
                    run_id=run_id,
                )
                return None

            # Read-back verification: never leave a corrupt checkpoint behind.
            try:
                probe = orjson.loads(path.read_bytes())
                if not isinstance(probe, dict) or probe.get("run_id") != run_id:
                    raise ValueError("read-back run_id mismatch")
            except Exception as e:
                restored = False
                if previous_bytes is not None:
                    try:
                        path.write_bytes(previous_bytes)
                        restored = True
                    except OSError:
                        restored = False
                logger.error(
                    "CHECKPOINT_SAVE_FAILED",
                    error=f"read-back failed: {e}; previous restored={restored}",
                    run_id=run_id,
                )
                return None

            _cleanup_stale_temps(path.parent, path.name)
            logger.debug("CHECKPOINT_SAVED", run_id=run_id, reason=reason, path=str(path))
            return path
        except Exception as e:
            logger.error("CHECKPOINT_SAVE_FAILED", error=str(e), run_id=run_id)
            return None
        finally:
            if tmp_path is not None:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass


def load_checkpoint(run_id: str | UUID) -> dict | None:
    """Load checkpoint; returns None when missing.

    Raises ValueError("corrupted") when the file exists but is unreadable.
    """
    path = checkpoint_path(run_id)
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        try:
            data = orjson.loads(raw)
        except Exception as e:
            raise ValueError(f"corrupted checkpoint {path}: {e}") from e
        if not isinstance(data, dict) or data.get("run_id") != str(run_id):
            raise ValueError(f"corrupted checkpoint {path}: run_id mismatch")
        if data.get("version") != CHECKPOINT_VERSION:
            logger.warning("Checkpoint version mismatch", path=str(path))
        return data
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"corrupted checkpoint {path}: {e}") from e


def list_checkpoints() -> list[dict]:
    out = []
    try:
        for p in sorted(checkpoint_dir().glob("run_*.json")):
            try:
                d = orjson.loads(p.read_bytes())
                out.append(
                    {
                        "run_id": d.get("run_id"),
                        "model": d.get("model"),
                        "stage": d.get("stage"),
                        "saved_at": d.get("saved_at"),
                        "completed": len(d.get("completed_candidate_ids", [])),
                    }
                )
            except Exception:
                out.append({"run_id": p.stem.replace("run_", ""), "corrupted": True})
    except Exception:
        pass
    return out


def remove_checkpoint(run_id: str | UUID) -> bool:
    try:
        p = checkpoint_path(run_id)
        if p.exists():
            p.unlink()
            return True
        return False
    except Exception:
        return False
