"""Staged hierarchical speed search + multi-finalist frontier (mission §7-§9).

No Cartesian products: each stage probes ONE lever (or a small compatible
group) against the current anchor and promotes improvements. Parameters
without a verified programmatic channel are never probed — they are
returned as report-only notes (MANUAL_ONLY / DETECTED_ONLY / unsupported).
"""

from dataclasses import dataclass

from lm_optimizer.domain.models import ConfigurationResult, ConfigurationStatus, LoadConfiguration
from lm_optimizer.services.control_adapter import GUI_ONLY

# GUI-only presets (mission §6/R1): exist, may matter, cannot be auto-tested.
MANUAL_ONLY_NOTE_KEYS = sorted(GUI_ONLY)

# Hard bound: the whole staged plan must stay far below a Cartesian product.
MAX_PLAN_PROBES = 25


@dataclass(frozen=True)
class SpeedCaps:
    """What the speed search may touch (resolved from registry + runtime)."""

    rest_keys: frozenset = frozenset()
    gpu_via_cli: bool = False
    is_moe: bool = False
    has_draft_model: bool = False
    flash_required: bool = False


@dataclass
class SpeedPlanItem:
    """One cheap probe: stage label, config, human reason."""

    stage: str
    config: LoadConfiguration
    reason: str
    kind: str = "latency"  # latency | throughput


def _clone(anchor: LoadConfiguration, context: int, **changes) -> LoadConfiguration:
    data = dict(anchor.to_dict())
    data["context_length"] = context
    data.update(changes)
    # Drop None overrides so anchor values survive when not changed.
    data = {k: v for k, v in data.items() if v is not None or k in changes}
    cfg = LoadConfiguration(**{k: v for k, v in data.items() if hasattr(LoadConfiguration, k)})
    for key, value in changes.items():
        setattr(cfg, key, value)
    cfg.context_length = context
    return cfg


def build_speed_plan(
    anchor: LoadConfiguration, speed_context: int, caps: SpeedCaps
) -> tuple[list[SpeedPlanItem], list[str]]:
    """Staged probe plan at the frozen speed context.

    Returns (probes, notes). Notes explain excluded parameters honestly.
    """
    probes: list[SpeedPlanItem] = []

    def add(stage: str, reason: str, kind: str = "latency", **changes) -> None:
        probes.append(
            SpeedPlanItem(
                stage=stage,
                config=_clone(anchor, speed_context, **changes),
                reason=reason,
                kind=kind,
            )
        )

    # S0 — baseline anchor at the frozen speed context.
    add("S0", "baseline anchor at fixed speed context")

    rest = set(caps.rest_keys or ())
    notes = _plan_s1(anchor, caps, rest, add)
    notes += _plan_s2_s3(anchor, rest, add)
    notes += _s4_notes()

    return probes[:MAX_PLAN_PROBES], notes


def _plan_s1(anchor: LoadConfiguration, caps: SpeedCaps, rest: set, add) -> list[str]:
    """S1 — high-impact levers, one at a time. Returns exclusion notes."""
    notes: list[str] = []
    if caps.gpu_via_cli:
        for ratio in (1.0, 0.8, 0.5):
            add("S1", f"GPU offload ratio {ratio} (CLI)", gpu_ratio=ratio)
    else:
        notes.append("gpu_ratio: CLI offload unavailable — ratio not auto-probed")
    if "offload_kv_cache_to_gpu" in rest:
        kv_now = anchor.offload_kv_cache_to_gpu
        add("S1", "KV cache placement toggle (REST)", offload_kv_cache_to_gpu=not kv_now)
    if "flash_attention" in rest and not caps.flash_required:
        flash_now = anchor.flash_attention
        add("S1", "Flash Attention toggle (REST)", flash_attention=not flash_now)
    elif caps.flash_required:
        notes.append("flash_attention: server requires flash — off not probed")
    if caps.is_moe and "num_experts" in rest:
        add("S1", "MoE expert count (REST)", num_experts=4)
    if caps.has_draft_model:
        add("S1", "speculative draft path (REST)", speculative_draft_mtp=True)
    else:
        notes.append("speculative/MTP: no draft model discovered — not auto-probed")
    return notes


def _plan_s2_s3(anchor: LoadConfiguration, rest: set, add) -> list:
    """S2 batch around the anchor + S3 concurrency probes."""
    if "eval_batch_size" in rest:
        base = anchor.eval_batch_size or 512
        for alt in (128, 1024):
            if alt != base:
                add("S2", f"eval batch {alt} (REST)", eval_batch_size=alt)
    if "physical_batch_size" in rest:
        base = anchor.physical_batch_size or 512
        for alt in (256, 1024):
            if alt != base:
                add("S2", f"physical batch {alt} (REST)", physical_batch_size=alt)
    if "context_checkpoints" in rest:
        base = anchor.context_checkpoints or 32
        for alt in (16, 64):
            if alt != base:
                add("S2", f"context checkpoints {alt} (REST)", context_checkpoints=alt)
    if "parallel" in rest:
        for alt in (1, 2):
            if alt != (anchor.parallel or 4):
                add("S3", f"parallel {alt} (throughput)", kind="throughput", parallel=alt)
    return []


def _s4_notes() -> list[str]:
    """S4 — load/memory: no verified programmatic path → report-only."""
    notes = [
        "try_mmap / keep_model_in_memory: MANUAL_ONLY (GUI) — load-time benefit "
        "reported as guidance, never auto-probed or claimed as tuned",
        "n_threads / cpu_thread_pool_size: MANUAL_ONLY/DETECTED_ONLY — "
        "set to logical max in LM Studio GUI; not auto-probed",
    ]
    noted = {
        "try_mmap",
        "keep_model_in_memory",
        "n_threads",
        "cpu_thread_pool_size",
        "cpu_threads",
    }
    for key in MANUAL_ONLY_NOTE_KEYS:
        if key not in noted:
            notes.append(f"{key}: MANUAL_ONLY — exists, cannot be auto-tuned")
    return notes


def build_interaction_probes(
    winners: list[LoadConfiguration], speed_context: int
) -> list[SpeedPlanItem]:
    """S5: combine ONLY the strongest settings (bounded, never exhaustive)."""
    if len(winners) < 2:
        return []
    merged: dict = {}
    for w in winners[:3]:
        merged.update({k: v for k, v in w.to_dict().items() if v is not None})
    merged["context_length"] = speed_context
    cfg = LoadConfiguration(
        **{k: v for k, v in merged.items() if hasattr(LoadConfiguration, k)}
    )
    cfg.context_length = speed_context
    first = winners[0]
    second = winners[1]
    partial: dict = dict(first.to_dict())
    partial.update({k: v for k, v in second.to_dict().items() if v is not None})
    partial["context_length"] = speed_context
    cfg2 = LoadConfiguration(
        **{k: v for k, v in partial.items() if hasattr(LoadConfiguration, k)}
    )
    cfg2.context_length = speed_context
    return [
        SpeedPlanItem(stage="S5", config=cfg, reason="combined strongest settings"),
        SpeedPlanItem(stage="S5", config=cfg2, reason="pairwise strongest combination"),
    ]


def select_frontier(
    results: list[ConfigurationResult], band: float = 0.05, max_band: int = 5
) -> dict:
    """Speed frontier: winner + 5% band finalists + one contrarian.

    Failed loads never enter. Contrarian = out-of-band config with the most
    different runtime strategy (most changed keys vs winner); None when
    everything is in band or only one result exists.
    """
    passed = [r for r in results if r.status == ConfigurationStatus.PASSED]
    ranked = sorted(passed, key=lambda r: r.get_avg_generation_tok_s(), reverse=True)
    if not ranked:
        return {"winner": None, "finalists": [], "contrarian": None}
    winner = ranked[0]
    top_speed = winner.get_avg_generation_tok_s()
    cutoff = top_speed * (1.0 - band)
    finalists = [r for r in ranked if r.get_avg_generation_tok_s() >= cutoff][:max_band]

    def _diff_keys(r: ConfigurationResult) -> int:
        a, b = winner.config.to_dict(), r.config.to_dict()
        keys = set(a) | set(b)
        return sum(1 for k in keys if a.get(k) != b.get(k))

    out_band = [r for r in ranked if r not in finalists]
    contrarian = max(out_band, key=lambda r: (_diff_keys(r), r.get_avg_generation_tok_s()), default=None)
    return {"winner": winner, "finalists": finalists, "contrarian": contrarian}
