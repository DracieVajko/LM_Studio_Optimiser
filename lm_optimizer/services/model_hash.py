"""Model file hashing (spec §16.3, §21 ModelIdentity.sha256).

LM Studio does not expose the GGUF path through the REST API, but the files
live under the LM Studio models directory. This module resolves a model id
to its GGUF file, hashes it (streaming, constant memory) and caches the
result so re-hashing a 16 GB file is a one-time cost.

Missing files and unreadable paths return None - never a fabricated hash.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

#: Cache file (keyed by absolute file path + size + mtime).
DEFAULT_CACHE = Path("data") / "model_hashes.json"

_LMSTUDIO_DIRS = (
    Path.home() / ".lmstudio" / "models",
    Path.home() / ".cache" / "lm-studio" / "models",
)


def models_root() -> Path | None:
    """First existing LM Studio models directory, or None."""
    for directory in _LMSTUDIO_DIRS:
        if directory.is_dir():
            return directory
    return None


def _normalise(text: str) -> str:
    """Lowercase alphanumerics only, for fuzzy path matching."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _candidate_paths(model_id: str) -> list[Path]:
    """Possible GGUF files for a model id.

    LM Studio's model key (e.g. `qwen3.8-9b-distill`) does not always match
    the on-disk layout (`empero-ai/Qwen3.8-9B-Distill-GGUF/…`), so the
    resolution is: direct path first, then a normalised name search across
    the publisher directories.
    """
    root = models_root()
    if root is None:
        return []
    parts = model_id.replace("\\", "/").split("/")
    candidates: list[Path] = []

    # 1. Direct path: models/<publisher>/<name>[/gguf]
    if len(parts) >= 2:
        base = root / parts[0] / "/".join(parts[1:])
    else:
        base = root / parts[0]
    base_no_quant = Path(str(base).split("@")[0])
    for directory in {base, base_no_quant}:
        if directory.is_dir():
            candidates.extend(sorted(directory.glob("*.gguf")))

    if candidates:
        return [p for p in candidates if p.is_file()]

    # 2. Fuzzy search: match the model id against repo directory names.
    wanted = _normalise(parts[-1].split("@")[0])
    if not wanted:
        return []
    try:
        for publisher_dir in root.iterdir():
            if not publisher_dir.is_dir():
                continue
            for repo_dir in publisher_dir.iterdir():
                if not repo_dir.is_dir():
                    continue
                name = _normalise(repo_dir.name)
                # Match when either side contains the other (the key drops
                # suffixes like "-gguf" and quant tags).
                if wanted in name or name in wanted:
                    candidates.extend(sorted(repo_dir.glob("*.gguf")))
                    if candidates:
                        break
            if candidates:
                break
    except OSError:
        pass
    return [p for p in candidates if p.is_file()]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Streaming SHA256 of a file (constant memory)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_cache(cache_path: Path) -> dict:
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


def model_sha256(
    model_id: str,
    cache_path: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> dict:
    """Resolve and hash the GGUF file for a model id.

    Returns {"model_id", "path", "size_bytes", "sha256", "cached", "error"}.
    `sha256` is None when the file cannot be found or read.
    """
    result = {
        "model_id": model_id,
        "path": None,
        "size_bytes": None,
        "sha256": None,
        "cached": False,
        "error": "",
    }
    candidates = _candidate_paths(model_id)
    if not candidates:
        result["error"] = "GGUF file not found under the LM Studio models directory"
        return result

    path = candidates[0]
    try:
        stat = path.stat()
    except OSError as e:
        result["error"] = f"cannot stat file: {e}"
        return result

    cache = {} if refresh else _load_cache(cache_path)
    key = str(path.resolve())
    entry = cache.get(key)
    if (
        isinstance(entry, dict)
        and entry.get("size") == stat.st_size
        and entry.get("mtime") == int(stat.st_mtime)
        and entry.get("sha256")
    ):
        result.update(
            path=str(path),
            size_bytes=stat.st_size,
            sha256=entry["sha256"],
            cached=True,
        )
        return result

    try:
        digest = sha256_file(path)
    except OSError as e:
        result["error"] = f"cannot read file: {e}"
        return result

    result.update(path=str(path), size_bytes=stat.st_size, sha256=digest)
    cache[key] = {
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
        "sha256": digest,
        "model_id": model_id,
    }
    _save_cache(cache_path, cache)
    return result


def hash_all_models(
    model_ids: list[str],
    cache_path: Path = DEFAULT_CACHE,
) -> list[dict]:
    """Hash several models, reusing the cache."""
    return [model_sha256(mid, cache_path=cache_path) for mid in model_ids]
