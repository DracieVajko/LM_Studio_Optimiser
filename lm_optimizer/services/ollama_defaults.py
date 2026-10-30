"""Ollama per-model load defaults persistence.

Mirrors lmstudio_defaults.py but targets Ollama's configuration.
Ollama has no single defaults file per model; we persist the winning
config as a JSON file in ~/.ollama/optimizer-defaults/<model>.json and
expose it via Modelfile-compatible parameters.

Supports the same persistence loop: apply, diff, restore, verify (mocked).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lm_optimizer.services.lmstudio_defaults import (
    DefaultsError,
    ModelNotFoundError,
    create_backup,
    prune_backups,
    list_backups,
    BackupEntry,
    DefaultsSaveResult,
    HOME_ENV as LMSTUDIO_HOME_ENV,
)

# Ollama config home env var
OLLAMA_HOME_ENV = "LM_OPTIMIZER_OLLAMA_HOME"
DEFAULT_OLLAMA_HOME = Path.home() / ".ollama"

# Field mapping: optimizer field -> Ollama Modelfile/API parameter
OLLAMA_FIELD_MAP = {
    "context_length": "num_ctx",
    "gpu_ratio": "num_gpu",
    "flash_attention": "flash_attention",
    "eval_batch_size": "num_batch",
    "physical_batch_size": "num_batch",
    "parallel": "num_parallel",
    "rope_freq_base": "rope_freq_base",
    "rope_freq_scale": "rope_freq_scale",
}


def default_ollama_home() -> Path:
    """Resolve Ollama home directory."""
    env = os.environ.get(OLLAMA_HOME_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_OLLAMA_HOME


def resolve_ollama_defaults_path(model_id: str, home: Path | None = None) -> Path:
    """Resolve the path to an Ollama model's optimizer defaults file."""
    home = home or default_ollama_home()
    # Sanitize model_id for filesystem safety
    safe_id = model_id.replace(":", "_").replace("/", os.sep)
    return home / "optimizer-defaults" / f"{safe_id}.json"


def _read_ollama_document(path: Path) -> tuple[dict, bool]:
    """Read an Ollama defaults document; return (document, existed)."""
    if not path.exists():
        return {"model": "", "config": {}}, False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise DefaultsError(f"Unreadable Ollama defaults file: {path}") from e
    if not isinstance(data, dict):
        raise DefaultsError(f"Invalid Ollama defaults document: {path}")
    return data, True


def _atomic_write(path: Path, data: dict) -> None:
    """Atomic write to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def _updates_for_ollama(config, fields: dict) -> dict:
    """Build updates dict for Ollama fields."""
    updates = {}
    for opt_field, ollama_key in OLLAMA_FIELD_MAP.items():
        value = getattr(config, opt_field, None)
        if value is not None:
            updates[ollama_key] = value
    return updates


@dataclass
class OllamaDefaultsSaveResult:
    model_id: str
    path: Path
    fields: dict
    created: bool
    backup_path: Path | None
    dry_run: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "path": str(self.path),
            "fields": self.fields,
            "created": self.created,
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "dry_run": self.dry_run,
        }


def apply_ollama_defaults(
    model_id: str,
    config: Any,
    *,
    home: Path | None = None,
    dry_run: bool = False,
    backup: bool = True,
) -> OllamaDefaultsSaveResult:
    """Write config into Ollama's optimizer defaults file."""
    home = home or default_ollama_home()
    path = resolve_ollama_defaults_path(model_id, home=home)
    document, existed = _read_ollama_document(path)
    fields = document.get("config", {})
    updates = _updates_for_ollama(config, fields)
    document["model"] = model_id
    document["config"] = {**fields, **updates}

    backup_path = None
    if not dry_run:
        if existed and backup:
            backup_path = create_backup(path, keep=10)
        _atomic_write(path, document)

    return OllamaDefaultsSaveResult(
        model_id=model_id,
        path=path,
        fields=updates,
        created=not existed,
        backup_path=backup_path,
        dry_run=dry_run,
    )


def save_run_ollama_defaults(run: Any, *, dry_run: bool = False) -> OllamaDefaultsSaveResult:
    """Persist a run's validated winner as Ollama defaults."""
    best = run.get_best_config() if hasattr(run, "get_best_config") else None
    if best is None:
        raise DefaultsError("No validated winner to persist")
    config = best.config
    if config is None:
        raise DefaultsError("Best config has no config object")
    model_id = getattr(getattr(run, "model", None), "id", "") or ""
    return apply_ollama_defaults(model_id, config, dry_run=dry_run)


def diff_ollama_defaults(model_id: str, config: Any, *, home: Path | None = None) -> dict:
    """Diff current Ollama defaults against a winner config."""
    home = home or default_ollama_home()
    path = resolve_ollama_defaults_path(model_id, home=home)
    document, existed = _read_ollama_document(path)
    current = document.get("config", {})
    updates = _updates_for_ollama(config, current)
    changes = []
    for key, winner_val in updates.items():
        if key in current:
            if current[key] != winner_val:
                changes.append({"key": key, "current": current[key], "winner": winner_val, "kind": "changed"})
            else:
                changes.append({"key": key, "current": current[key], "winner": winner_val, "kind": "unchanged"})
        else:
            changes.append({"key": key, "current": None, "winner": winner_val, "kind": "added"})
    unchanged = [k for k in current if k not in updates]
    return {
        "model_id": model_id,
        "path": str(path),
        "exists": existed,
        "changes": changes,
        "unchanged": unchanged,
        "preserved": [],
    }


def restore_ollama_defaults(
    model_id: str, *, backup: str = "latest", dry_run: bool = False, keep_backups: bool = True, home: Path | None = None
) -> dict:
    """Restore Ollama defaults from a backup."""
    from lm_optimizer.services.lmstudio_defaults import restore_defaults as _restore
    # Delegate to the same restore logic but with Ollama path
    home = home or default_ollama_home()
    path = resolve_ollama_defaults_path(model_id, home=home)
    # Use the same restore_defaults logic but on Ollama path
    # For simplicity, implement inline
    backups = list_backups(path)
    if backup == "latest":
        if not backups:
            raise DefaultsError(f"No backups available for '{model_id}'")
        selected = backups[0]
    else:
        selected = next((b for b in backups if b.name == backup), None)
        if selected is None:
            raise DefaultsError(f"Unknown backup '{backup}'")
    # Validate backup JSON
    try:
        with open(selected.path, "r", encoding="utf-8") as f:
            backup_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise DefaultsError(f"Invalid backup: {selected.name}") from e
    if not isinstance(backup_data, dict):
        raise DefaultsError(f"Invalid backup: {selected.name}")
    current_bytes = path.read_bytes() if path.exists() else b""
    backup_bytes = selected.path.read_bytes()
    noop = current_bytes == backup_bytes
    if noop:
        return {"model_id": model_id, "path": str(path), "restored_from": selected.name, "backup_path": None, "noop": True, "dry_run": dry_run}
    if dry_run:
        return {"model_id": model_id, "path": str(path), "restored_from": selected.name, "backup_path": None, "noop": False, "dry_run": True}
    # Backup current
    if path.exists() and keep_backups:
        create_backup(path, keep=10)
    _atomic_write(path, backup_data)
    return {"model_id": model_id, "path": str(path), "restored_from": selected.name, "backup_path": str(path), "noop": False, "dry_run": False}
PYEOF
echo "File created"