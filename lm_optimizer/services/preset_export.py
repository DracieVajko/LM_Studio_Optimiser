"""LM Studio preset export.

LM Studio exposes no documented API to write its *native GUI* presets, so the
optimizer cannot create a preset inside LM Studio directly. The supported
workflow is:

1. The optimizer writes a validated, importable JSON preset for a model from
   that run's ACTUAL winning candidate configuration (never defaults or stale
   UI state).
2. The user applies it with ``lm-optimizer apply`` (REST) or copies the values
   into LM Studio's GUI preset editor.

Every value is traced back to the source run and winning candidate id. Fields
that LM Studio's REST load endpoint cannot set (e.g. ``gpu_ratio``) are
reported explicitly under ``recorded_only_params`` / ``unsupported_params``
instead of being silently dropped.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

PRESET_FORMAT = "lm-optimizer.lmstudio.preset"
PRESET_VERSION = 1
LMSTUDIO_PRESET_DIR = Path("lmstudio-presets")

# LoadConfiguration fields that REST /models/load cannot set (recorded only).
REST_UNSUPPORTED: tuple[str, ...] = ("gpu_ratio", "rope_freq_base", "rope_freq_scale")


def default_preset_filename(model_id: str, profile: str) -> str:
    """Filesystem-safe ``<model>_<profile>.preset.json`` stem."""
    import re

    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(model_id)).strip("-") or "model"
    return f"{stem}_{profile}.preset.json"


def build_lmstudio_preset(
    *,
    model_id: str,
    profile: str,
    config: dict,
    context_length: int | None = None,
    run_id: str | None = None,
    best_config_id: str | None = None,
    metrics: dict | None = None,
    quality: dict | None = None,
    optimizer_version: str | None = None,
) -> dict:
    """Build an importable LM Studio preset from a winning config.

    ``config`` is the winning candidate's own configuration dict. The winner's
    outer ``context_length`` is authoritative and is forced onto the applied
    params so a stale nested value can never win.
    """
    if not config:
        raise ValueError("Cannot export preset: empty winning configuration")
    applied = {k: v for k, v in config.items() if k not in REST_UNSUPPORTED and v is not None}
    recorded_only = {
        k: config.get(k) for k in REST_UNSUPPORTED if config.get(k) is not None
    }
    ctx = context_length if context_length is not None else config.get("context_length")
    if ctx is not None:
        applied["context_length"] = ctx
    return {
        "format": PRESET_FORMAT,
        "version": PRESET_VERSION,
        "model_id": model_id,
        "profile": profile,
        "context_length": ctx,
        "applied_params": applied,
        "recorded_only_params": recorded_only,
        "unsupported_params": sorted(recorded_only.keys()),
        "source_run_id": str(run_id) if run_id else None,
        "best_config_id": str(best_config_id) if best_config_id else None,
        "metrics": metrics or None,
        "quality": quality or None,
        "optimizer_version": optimizer_version,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": (
            "LM Studio has no direct preset API. Apply with `lm-optimizer apply` "
            "(REST) or copy applied_params into LM Studio's GUI preset editor. "
            "Fields in unsupported_params are NOT applied by REST."
        ),
    }


def render_lmstudio_preset(preset: dict) -> str:
    """Serialize a preset dict to stable, readable JSON."""
    return json.dumps(preset, indent=2, sort_keys=False)


def parse_lmstudio_preset(text: str) -> dict:
    """Parse + validate an exported preset (raises ValueError when malformed)."""
    try:
        data = json.loads(text)
    except Exception as e:
        raise ValueError(f"not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("preset root must be an object")
    if data.get("format") != PRESET_FORMAT:
        raise ValueError(f"unexpected preset format: {data.get('format')!r}")
    applied = data.get("applied_params")
    if not isinstance(applied, dict) or not applied:
        raise ValueError("preset has no applied_params")
    if applied.get("context_length") is None:
        raise ValueError("preset is missing applied context_length")
    return data


def write_lmstudio_preset(
    preset: dict,
    out_dir: str | Path = LMSTUDIO_PRESET_DIR,
    filename: str | None = None,
) -> Path:
    """Write the preset file, defaulting to ``lmstudio-presets/``.

    Raises OSError on write failure (never reports false success).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = filename or default_preset_filename(preset["model_id"], preset["profile"])
    path = out / name
    text = render_lmstudio_preset(preset)
    # Validate our own output before writing a real file.
    parse_lmstudio_preset(text)
    path.write_text(text, encoding="utf-8")
    return path


def preset_from_run(run: Any, profile: str | None = None, best: Any | None = None) -> dict:
    """Build a preset from an OptimizationRun's validated winner."""
    best = best if best is not None else run.get_best_config()
    if best is None:
        raise ValueError("No validated winning configuration to export")
    cfg = best.config.to_dict()
    q = getattr(best, "quality_score", None)
    return build_lmstudio_preset(
        model_id=run.model.id,
        profile=profile or getattr(run.profile, "value", str(run.profile)),
        config=cfg,
        context_length=getattr(best, "context_length", None),
        run_id=getattr(run, "id", None),
        best_config_id=getattr(best, "id", None),
        metrics={
            "generation_tok_s": best.get_avg_generation_tok_s(),
            "prompt_tok_s": best.get_avg_prompt_tok_s(),
            "estimated_ttft_ms": best.get_avg_estimated_ttft_ms(),
        },
        quality={"overall": q.overall} if q is not None else None,
        optimizer_version=getattr(run, "optimizer_version", None),
    )
