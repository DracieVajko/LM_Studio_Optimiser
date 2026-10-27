"""Persist an optimizer winner as LM Studio per-model load defaults.

LM Studio stores "My Models" defaults in
``<lmstudio-home>/.internal/user-concrete-model-default-config/<model>.json``.
Writing that file is enough: the next load resolves the new values without
restarting LM Studio (verified live on 2026-10-27 by editing
``llm.load.contextLength`` and reading back the echoed load config).

The file is JSON shaped as::

    {
      "preset": "",
      "operation": {"fields": []},
      "load": {"fields": [{"key": "llm.load.contextLength", "value": 8192}]}
    }

This module only changes keys the optimizer actually tested. Unknown keys
(CPU threads, cache quantization, mmap, ...) and unknown top-level sections
are preserved, and malformed files are never overwritten.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lm_optimizer.domain.models import LoadConfiguration

#: LM Studio home can be redirected for portable installs and tests. It is
#: not an LM Studio variable; it only affects this optimizer's file writes.
HOME_ENV = "LM_OPTIMIZER_LMSTUDIO_HOME"

DEFAULTS_SUBDIR = Path(".internal") / "user-concrete-model-default-config"
MODEL_DATA_PATH = Path(".internal") / "model-data.json"
BACKUP_SUFFIX = ".optimizer-backup"
BACKUP_KEEP = 10
_BACKUP_TIMESTAMP_RE = re.compile(r"^\d{8}T\d{6}Z(-\d+)?$")


class DefaultsError(RuntimeError):
    """Base error for defaults resolution and writing."""


class ModelNotFoundError(DefaultsError):
    """The model has no LM Studio metadata and no existing defaults file."""


class NoWinnerError(DefaultsError):
    """The run has no validated winner to persist."""


class UnsafePathError(DefaultsError):
    """A model identifier would escape the LM Studio defaults directory."""


@dataclass(frozen=True)
class DefaultsSaveResult:
    """What was (or would be) written, for CLI/API/UI output."""

    model_id: str
    path: Path
    fields: dict[str, Any]
    created: bool
    backup_path: Path | None
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "path": str(self.path),
            "fields": self.fields,
            "created": self.created,
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "dry_run": self.dry_run,
        }


@dataclass(frozen=True)
class BackupEntry:
    """A single defaults backup discovered on disk."""

    name: str
    path: Path
    modified: float
    legacy: bool


# ``model field -> (canonical LM Studio key, legacy key, value kind)``.
# Canonical keys come from LM Studio's own field list. Speculative decoding
# keys are canonical in the current UI schema; older files use the ``llama.``
# spelling, so an existing legacy key is updated in place instead of adding a
# second, conflicting key.
_FIELD_MAP: tuple[tuple[str, str, str | None, str], ...] = (
    ("context_length", "llm.load.contextLength", None, "raw"),
    (
        "gpu_ratio",
        "llm.load.llama.acceleration.offloadRatio",
        None,
        "raw",
    ),
    ("flash_attention", "llm.load.llama.flashAttention", None, "raw"),
    ("offload_kv_cache_to_gpu", "llm.load.offloadKVCacheToGpu", None, "raw"),
    ("eval_batch_size", "llm.load.llama.evalBatchSize", None, "raw"),
    ("physical_batch_size", "llm.load.llama.physicalBatchSize", None, "raw"),
    ("parallel", "llm.load.numParallelSessions", None, "raw"),
    ("context_checkpoints", "llm.load.llama.contextCheckpoints", None, "raw"),
    ("reasoning_budget_message", "llm.load.llama.reasoningBudgetMessage", None, "raw"),
    ("num_experts", "llm.load.numExperts", None, "raw"),
    ("rope_freq_base", "llm.load.llama.ropeFrequencyBase", None, "checked"),
    ("rope_freq_scale", "llm.load.llama.ropeFrequencyScale", None, "checked"),
    (
        "speculative_draft_max_tokens",
        "llm.load.speculativeDecoding.draftMaxTokens",
        "llm.load.llama.speculativeDecoding.draftMaxTokens",
        "raw",
    ),
    (
        "speculative_draft_min_tokens",
        "llm.load.speculativeDecoding.draftMinTokens",
        "llm.load.llama.speculativeDecoding.draftMinTokens",
        "raw",
    ),
    (
        "speculative_draft_min_continue_probability",
        "llm.load.speculativeDecoding.draftMinContinueProbability",
        "llm.load.llama.speculativeDecoding.draftMinContinueProbability",
        "raw",
    ),
    (
        "speculative_draft_mtp",
        "llm.load.speculativeDecoding.draftMtp",
        "llm.load.llama.speculativeDecoding.draftMtp",
        "raw",
    ),
    (
        "speculative_draft_simple",
        "llm.load.speculativeDecoding.draftSimple",
        "llm.load.llama.speculativeDecoding.draftSimple",
        "raw",
    ),
    (
        "speculative_draft_model",
        "llm.load.speculativeDecoding.draftModel",
        "llm.load.llama.speculativeDecoding.draftModel",
        "raw",
    ),
)

def default_home() -> Path:
    """LM Studio home directory (``~/.lmstudio`` unless overridden)."""
    override = os.environ.get(HOME_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".lmstudio"


def _relative_parts(relative: str) -> tuple[str, ...]:
    """Split a model-relative path and reject traversal/absolute forms."""
    normalized = relative.replace("\\", "/")
    if normalized.startswith("/") or (len(normalized) >= 2 and normalized[1] == ":"):
        raise UnsafePathError(f"Unsafe model path: {relative!r}")
    parts = tuple(p for p in normalized.split("/") if p not in ("", "."))
    if not parts or any(p == ".." for p in parts):
        raise UnsafePathError(f"Unsafe model path: {relative!r}")
    return parts


def _to_path(root: Path, relative: str) -> Path:
    path = root.joinpath(*_relative_parts(relative))
    try:
        path.relative_to(root)
    except ValueError as exc:  # pragma: no cover - defensive
        raise UnsafePathError(f"Unsafe model path: {relative!r}") from exc
    return path


def _defaults_root(home: Path) -> Path:
    return home / DEFAULTS_SUBDIR


def _model_data(home: Path) -> list[tuple[str, dict]]:
    path = home / MODEL_DATA_PATH
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = payload.get("json") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []
    result: list[tuple[str, dict]] = []
    for item in entries:
        if (
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], dict)
        ):
            result.append((item[0], item[1]))
    return result


def _defaults_file_for(root: Path, concrete: str) -> Path:
    """Map a concrete model path to its ``...json`` defaults file."""
    relative = concrete if concrete.endswith(".json") else concrete + ".json"
    return _to_path(root, relative)


def _existing_index(root: Path) -> dict[str, Path]:
    """Index existing defaults files by full and short model identifiers."""
    if not root.exists():
        return {}
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*.json")):
        rel = path.relative_to(root).as_posix()
        without_json = rel[: -len(".json")]
        index.setdefault(without_json, path)
        if without_json.endswith(".gguf"):
            index.setdefault(without_json[: -len(".gguf")], path)
        index.setdefault(path.name[: -len(".json")].removesuffix(".gguf"), path)
    return index


def resolve_defaults_path(model_id: str, *, home: Path | None = None) -> Path:
    """Resolve the per-model defaults file, creating nothing on disk.

    Resolution order:

    1. exact concrete path in ``model-data.json``;
    2. ``uiIdentifier`` in ``model-data.json`` (supports ``model@variant``);
    3. an existing defaults file matching the id or its basename;
    4. direct ``<model_id>.json`` for path-like identifiers.

    A model known only to LM Studio metadata may not have a defaults file
    yet; the returned path is then the location LM Studio uses on its first
    write. Unknown ids raise :class:`ModelNotFoundError` instead of inventing
    a file outside LM Studio's own layout.
    """
    model_id = (model_id or "").strip()
    if not model_id:
        raise ModelNotFoundError("Empty model identifier")
    home = home or default_home()
    root = _defaults_root(home)
    entries = _model_data(home)

    exact = [concrete for concrete, _ in entries if concrete == model_id]
    by_ui = [
        concrete
        for concrete, meta in entries
        if str(meta.get("uiIdentifier") or "") == model_id
    ]
    candidates: list[Path] = []
    for concrete in [*exact, *by_ui]:
        candidate = _defaults_file_for(root, concrete)
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    index = _existing_index(root)
    direct = index.get(model_id)
    if direct is not None:
        return direct
    direct = index.get(Path(model_id).name)
    if direct is not None:
        return direct

    if candidates:
        # A known model whose defaults file does not exist yet.
        return candidates[0]

    if "/" in model_id or "\\" in model_id:
        candidate = _defaults_file_for(root, model_id)
        if candidate.exists():
            return candidate

    raise ModelNotFoundError(
        f"Model {model_id!r} was not found in LM Studio model-data.json; "
        "load or download it once in LM Studio first"
    )


def _new_document() -> dict[str, Any]:
    return {"preset": "", "operation": {"fields": []}, "load": {"fields": []}}


def _read_document(path: Path) -> tuple[dict[str, Any], bool]:
    """Read an existing defaults file; returns (document, existed)."""
    if not path.exists():
        return _new_document(), False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DefaultsError(
            f"Refusing to overwrite unreadable LM Studio defaults file: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise DefaultsError(
            f"Refusing to overwrite non-object LM Studio defaults file: {path}"
        )
    return payload, True


def _field_list(document: dict[str, Any]) -> list[dict[str, Any]]:
    load = document.setdefault("load", {})
    if not isinstance(load, dict):
        raise DefaultsError("LM Studio defaults file has a non-object 'load' section")
    fields = load.setdefault("fields", [])
    if not isinstance(fields, list):
        raise DefaultsError("LM Studio defaults file has a non-list 'load.fields'")
    return fields


def _existing_keys(fields: list[dict[str, Any]]) -> set[str]:
    return {
        str(field.get("key"))
        for field in fields
        if isinstance(field, dict) and field.get("key")
    }


def _wrapped(kind: str, value: Any) -> Any:
    if kind == "checked":
        return {"checked": True, "value": value}
    return value


def _updates_for(
    config: LoadConfiguration, fields: list[dict[str, Any]]
) -> dict[str, Any]:
    existing = _existing_keys(fields)
    updates: dict[str, Any] = {}
    for attr, canonical, legacy, kind in _FIELD_MAP:
        value = getattr(config, attr, None)
        if value is None:
            continue
        key = legacy if legacy and legacy in existing else canonical
        updates[key] = _wrapped(kind, value)
    return updates


def _merge(document: dict[str, Any], updates: dict[str, Any]) -> list[dict[str, Any]]:
    fields = _field_list(document)
    by_key = {
        str(field.get("key")): field
        for field in fields
        if isinstance(field, dict) and field.get("key")
    }
    for key, value in updates.items():
        field = by_key.get(key)
        if field is None:
            field = {"key": key, "value": value}
            fields.append(field)
            by_key[key] = field
        else:
            field["value"] = value
        # Duplicate keys should not survive with a stale value.
        for other in fields:
            if other is not field and isinstance(other, dict) and other.get("key") == key:
                other["value"] = value
    return fields


def _atomic_write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _backup_prefix(path: Path) -> str:
    return f"{path.name}{BACKUP_SUFFIX}"


def _is_service_backup_name(path: Path, name: str) -> bool:
    prefix = _backup_prefix(path)
    if not name.startswith(prefix + "-"):
        return False
    suffix = name[len(prefix) + 1 :]
    return bool(_BACKUP_TIMESTAMP_RE.match(suffix))


def list_backups(path: Path) -> list[BackupEntry]:
    """List defaults backups newest-first.

    Recognises the legacy ``<name>.optimizer-backup`` file plus timestamped
    ``<name>.optimizer-backup-<UTC stamp>`` files (and their ``-N`` collision
    variants).
    """
    directory = path.parent
    if not directory.exists():
        return []
    prefix = _backup_prefix(path)
    entries: list[BackupEntry] = []
    for candidate in directory.iterdir():
        if not candidate.is_file():
            continue
        name = candidate.name
        if name == prefix:
            legacy = True
        elif name.startswith(prefix + "-"):
            legacy = False
        else:
            continue
        try:
            modified = candidate.stat().st_mtime
        except OSError:
            continue
        entries.append(
            BackupEntry(name=name, path=candidate, modified=modified, legacy=legacy)
        )
    entries.sort(key=lambda entry: (entry.modified, entry.name), reverse=True)
    return entries


def prune_backups(path: Path, *, keep: int = 10) -> None:
    """Delete oldest timestamped backups beyond ``keep``.

    Only files whose names were produced by this service are touched; the
    legacy ``.optimizer-backup`` and any unrelated files are preserved.
    """
    directory = path.parent
    if not directory.exists() or keep < 0:
        return
    timestamped: list[Path] = []
    for candidate in directory.iterdir():
        if not candidate.is_file():
            continue
        if not _is_service_backup_name(path, candidate.name):
            continue
        timestamped.append(candidate)
    timestamped.sort(key=lambda p: p.name, reverse=True)
    for stale in timestamped[keep:]:
        try:
            stale.unlink()
        except OSError:
            continue


def create_backup(
    path: Path, *, keep: int = 10, now: datetime | None = None
) -> Path | None:
    """Copy ``path`` to a timestamped backup and prune, or ``None`` if missing."""
    if not path.exists():
        return None
    if now is None:
        now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    base_name = f"{path.name}{BACKUP_SUFFIX}-{stamp}"
    candidate = directory / base_name
    if candidate.exists():
        index = 1
        while (directory / f"{base_name}-{index}").exists():
            index += 1
        candidate = directory / f"{base_name}-{index}"
    shutil.copy2(path, candidate)
    prune_backups(path, keep=keep)
    return candidate


def apply_defaults(
    model_id: str,
    config: LoadConfiguration,
    *,
    home: Path | None = None,
    dry_run: bool = False,
    backup: bool = True,
) -> DefaultsSaveResult:
    """Write ``config`` into the model's LM Studio defaults.

    Only non-``None`` config fields are written. Existing unknown fields and
    sections are preserved. The previous file is copied to
    ``<name>.optimizer-backup`` before an overwrite (unless ``backup=False``).
    """
    home = home or default_home()
    path = resolve_defaults_path(model_id, home=home)
    document, existed = _read_document(path)
    fields = _field_list(document)
    updates = _updates_for(config, fields)
    _merge(document, updates)

    backup_path: Path | None = None
    if not dry_run:
        if existed and backup:
            backup_path = create_backup(path, keep=BACKUP_KEEP)
        _atomic_write(path, document)

    return DefaultsSaveResult(
        model_id=model_id,
        path=path,
        fields=updates,
        created=not existed,
        backup_path=backup_path,
        dry_run=dry_run,
    )


def save_run_defaults(
    run: Any,
    *,
    home: Path | None = None,
    dry_run: bool = False,
    backup: bool = True,
) -> DefaultsSaveResult:
    """Persist the run's validated winner as the model's LM Studio default."""
    best = run.get_best_config() if hasattr(run, "get_best_config") else None
    if best is None:
        raise NoWinnerError("No validated winner in this run")
    config = best.config
    if config.context_length is None and getattr(best, "context_length", None):
        config = replace(config, context_length=best.context_length)
    model_id = getattr(getattr(run, "model", None), "id", "") or ""
    return apply_defaults(
        model_id,
        config,
        home=home,
        dry_run=dry_run,
        backup=backup,
    )
