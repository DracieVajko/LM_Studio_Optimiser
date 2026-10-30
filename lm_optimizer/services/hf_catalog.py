"""HuggingFace GGUF catalogue discovery (spec §16.3).

The recommender ships a curated catalogue. This module lets the user widen
it from the HuggingFace API without trusting any single source blindly:

- `search_gguf_models(query)` lists GGUF repos,
- `repo_to_entry()` turns a repo name into a catalog entry, deriving the
  parameter count and quantization from the repo/file naming,
- `sync_catalogue()` merges the discovered entries with the curated ones.

Every discovered entry is marked `source="huggingface"` and confidence `D`
until a local measurement or a community benchmark upgrades it - discovery
is a prior, never evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lm_optimizer.services.recommender import (
    Confidence,
    ModelCatalogEntry,
)

HF_API = "https://huggingface.co/api/models"

#: Approximate GGUF size in GB per 1B params for common quant families.
_QUANT_BPW = {
    "Q8_0": 8.0,
    "Q6_K": 6.6,
    "Q5_K_M": 5.6,
    "Q5_K_S": 5.5,
    "Q4_K_M": 4.8,
    "Q4_K_S": 4.6,
    "Q4_0": 4.5,
    "IQ4_XS": 4.25,
    "IQ4_NL": 4.5,
    "Q3_K_M": 3.9,
    "Q3_K_S": 3.5,
    "IQ3_XS": 3.4,
    "IQ3_M": 3.8,
    "Q2_K": 3.0,
    "IQ2_XS": 2.7,
    "Q1_0": 1.6,
    "MXFP4": 4.25,
    "Q2_K_S": 3.0,
}

_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")
_QUANT_RE = re.compile(
    r"(IQ\d(?:_[A-Z0-9]+)?|Q\d(?:_[A-Z0-9]+)*|MXFP4)", re.IGNORECASE
)


def parse_params_b(name: str) -> float | None:
    """Parameter count in billions from a repo/file name (27B, 8b, 30B-A3B)."""
    if not name:
        return None
    # Prefer the first "<number>B" token; avoid matching "A3B" active params.
    for match in _PARAM_RE.finditer(name):
        start = match.start()
        prefix = name[max(0, start - 1) : start]
        if prefix.upper() == "A":
            continue
        try:
            return float(match.group(1))
        except ValueError:
            continue
    return None


def parse_quant(name: str) -> str | None:
    """Quantization tag from a repo/file name."""
    match = _QUANT_RE.search(name or "")
    if not match:
        return None
    return match.group(0).upper()


def estimate_size_gb(params_b: float | None, quant: str | None) -> float | None:
    """Estimate GGUF size from parameter count and quant family."""
    if not params_b:
        return None
    bpw = _QUANT_BPW.get((quant or "").upper())
    if bpw is None:
        return None
    return round(params_b * bpw / 8.0, 2)


def parse_repo_id(repo_id: str) -> dict:
    """Split `publisher/name` and derive params/quant."""
    parts = (repo_id or "").split("/")
    publisher = parts[0] if len(parts) > 1 else ""
    name = parts[-1]
    params = parse_params_b(name)
    quant = parse_quant(name)
    return {
        "publisher": publisher,
        "name": name,
        "params_b": params,
        "quant": quant,
        "size_gb": estimate_size_gb(params, quant),
    }


@dataclass
class SyncResult:
    """Outcome of a catalogue sync."""

    discovered: int = 0
    added: list[ModelCatalogEntry] = None  # type: ignore[assignment]
    skipped: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.added is None:
            self.added = []
        if self.errors is None:
            self.errors = []


def search_gguf_models(query: str, limit: int = 20, timeout_s: float = 20.0) -> list[str]:
    """Return GGUF repo ids matching the query (network call)."""
    import httpx

    params = {
        "search": query,
        "filter": "gguf",
        "limit": max(1, min(limit, 100)),
        "sort": "downloads",
        "direction": "-1",
    }
    try:
        response = httpx.get(HF_API, params=params, timeout=timeout_s)
        response.raise_for_status()
        payload = response.json()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"HuggingFace search failed: {e}") from e
    ids: list[str] = []
    for item in payload if isinstance(payload, list) else []:
        repo_id = item.get("id") or item.get("modelId")
        if repo_id:
            ids.append(repo_id)
    return ids


def repo_to_entry(repo_id: str) -> ModelCatalogEntry | None:
    """Build a catalog entry from a repo id; None when unparseable."""
    info = parse_repo_id(repo_id)
    if not info["params_b"] or not info["quant"] or not info["size_gb"]:
        return None
    return ModelCatalogEntry(
        repository=repo_id,
        filename=f"{info['name']}.gguf",
        architecture="unknown",
        quant=info["quant"],
        bits_per_weight=round(
            _QUANT_BPW.get(info["quant"].upper(), 4.0), 2
        ),
        file_size_gb=info["size_gb"],
        total_params_b=info["params_b"],
        max_context=0,
        capabilities=["chat"],
        source="huggingface",
        confidence=Confidence.D,
        notes="discovered from the HuggingFace API; no local measurement yet",
    )


def sync_catalogue(
    query: str,
    existing: list[ModelCatalogEntry] | None = None,
    limit: int = 20,
    repo_ids: list[str] | None = None,
) -> SyncResult:
    """Discover repos and merge them with an existing catalogue.

    `repo_ids` lets the caller inject ids (tests, or a hand-picked list)
    instead of hitting the network.
    """
    base = list(existing or [])
    seen = {(e.repository.lower(), e.quant.upper()) for e in base}
    result = SyncResult()
    ids = repo_ids if repo_ids is not None else search_gguf_models(query, limit=limit)
    result.discovered = len(ids)
    for repo_id in ids:
        entry = repo_to_entry(repo_id)
        if entry is None:
            result.skipped += 1
            result.errors.append(f"{repo_id}: cannot derive params/quant/size")
            continue
        key = (entry.repository.lower(), entry.quant.upper())
        if key in seen:
            result.skipped += 1
            continue
        seen.add(key)
        result.added.append(entry)
    return result
