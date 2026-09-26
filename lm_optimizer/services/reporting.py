"""Per-model best-settings markdown reports (ASCII only)."""

import re
from datetime import datetime
from pathlib import Path

from lm_optimizer.domain.models import OptimizationRun
from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)


def sanitize_model_filename(model_id: str) -> str:
    """Make a filesystem-safe stem from a model id."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_id).strip("-")
    return stem or "model"


def _fmt(value: object, na: str = "N/A") -> str:
    if value is None:
        return na
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


SERVER_DEFAULTS = {"physical_batch_size": 512, "parallel": 4, "context_checkpoints": 32}


def _tried(configs: list, get) -> list:
    """Sorted unique non-None tried values for a dimension."""
    vals = {get(c) for c in configs}
    vals.discard(None)
    try:
        return sorted(vals)
    except TypeError:
        return sorted(vals, key=str)


def _verdict(best_val, tried: list, default) -> str:
    """Verified-best vs server-default-vs-tested label.

    Explicit semantics (§27): a retained default is never reported as a
    measured win. "N/A" is used only when nothing was decided.
    """
    if best_val is None:
        if tried:
            return (
                f"server default ({default}; tested {', '.join(map(str, tried))}; "
                "default retained because no tested alternative exceeded "
                "the preference threshold)"
            )
        return f"server default ({default}, not set)"
    others = [v for v in tried if v != best_val]
    if others:
        return f"verified best of tested ({', '.join(map(str, tried))})"
    if best_val == default:
        return f"server default ({default}, alternatives untested)"
    return f"tested ({', '.join(map(str, tried))})"


def _prune_notes(run: OptimizationRun, configs: list) -> list[str]:
    """Why some dimensions/values were never tried."""
    notes = []
    space = run.search_space or {}
    flash_tried = _tried(configs, lambda c: c.config.flash_attention)
    if False in (space.get("flash_attention_options") or []) and False not in flash_tried:
        if space.get("flash_pruned"):
            notes.append(
                "flash=False pruned: server requires flash attention "
                "(non-F16 KV cache quantization); flash=False configs were skipped."
            )
        else:
            notes.append("flash=False was offered but never tried.")
    if space.get("gpu_ratios"):
        notes.append(
            "gpu_ratio (fraction of model layers on GPU, 0-1) is CLI/GUI-only "
            "(`lms load <model> --gpu 0.5|max|off`); never sent via REST."
        )
    notes.append(
        "rope_freq_*, n_threads, try_mmap, keep_model_in_memory, unified_kv_cache, "
        "ttl, num_layers are rejected via REST (GUI/CLI-only or unsupported)."
    )
    return notes


def save_fit_report(
    ladder_out: dict,
    out_dir: str | Path = "results",
    host: dict | None = None,
) -> Path | None:
    """Save a max-context fit report. Returns path."""
    model_id = ladder_out.get("model", "model")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{sanitize_model_filename(model_id)}-fit.md"
    host = host or {}
    ceilings = ladder_out.get("ceilings", {})
    recommended = ladder_out.get("recommended", ceilings)

    def ceil(kv):
        v = ceilings.get(kv)
        return str(v) if v else "none (refused at smallest context)"

    def reco(kv):
        v = recommended.get(kv)
        return str(v) if v else "none"

    lines = [
        f"# Fit report (max context): {model_id}",
        "",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- GPU: {host.get('gpu', 'unknown')} / CPU: {host.get('cpu', 'unknown')} / "
        f"RAM: {host.get('ram', 'unknown')}",
        f"- KV cache quantization: "
        f"{host.get('kv_quant', 'Q4 starter (GUI preset, REST cannot set)')}",
        "",
        "## Ceilings (max passing context)",
        "",
        f"- KV on GPU (speed path): {ceil(True)}",
        f"- KV on CPU (capacity path): {ceil(False)}",
        "",
        f"- Recommended GPU context: {reco(True)}",
        f"- Recommended CPU context: {reco(False)}",
        "",
        "Recommendation: use the largest passing GPU context for speed; "
        "the CPU ceiling for max-capacity work. Full stage tuning later.",
        "",
        "## Ladder (ctx x KV path)",
        "",
        "| Context | KV-GPU | tok/s | KV-CPU | tok/s |",
        "| --- | --- | --- | --- | --- |",
    ]
    paths = ladder_out.get("paths", {})
    gpu_path = paths.get(True, {})
    cpu_path = paths.get(False, {})
    for ctx in sorted(set(gpu_path) | set(cpu_path)):
        g = gpu_path.get(ctx, {})
        c = cpu_path.get(ctx, {})
        g_mark = "OK" if g.get("ok") else "REFUSED"
        c_mark = "OK" if c.get("ok") else "REFUSED"
        lines.append(f"| {ctx} | {g_mark} | {g.get('tok_s', 0)} | {c_mark} | {c.get('tok_s', 0)} |")
    lines += [""]
    for kv, label in ((True, "KV-GPU"), (False, "KV-CPU")):
        for ctx in sorted(paths.get(kv, {})):
            err = paths[kv][ctx].get("error", "")
            if err:
                lines.append(f"- {label} ctx {ctx} refusal: {err}")
    lines += [""]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Fit report saved", model=model_id, path=str(path))
    return path


def _drives_str(drives: list) -> str:
    """One-line drive inventory (detection only)."""
    parts = []
    for d in drives or []:
        if isinstance(d, dict):
            label = d.get("name") or d.get("device") or d.get("id") or "?"
            size = d.get("size_gb")
            size_txt = f" {size}GB" if size else ""
            parts.append(f"{label} ({d.get('media', 'unknown')}{size_txt})")
        else:
            parts.append(str(d))
    return ", ".join(parts) or "unknown"


def _score_str(score) -> str:
    """Score display that never crashes on unscored (None) configs."""
    try:
        return f"{float(score):.3f}" if score is not None else "not scored"
    except (TypeError, ValueError):
        return "not scored"


def _channel_rows(best) -> list[str]:
    """Control-channel rows for the winner's parameters (§8/§34)."""
    try:
        from lm_optimizer.services.control_adapter import channel_table
    except Exception:
        return ["| (channel table unavailable) | | |"]
    names = [
        "context_length",
        "gpu_ratio",
        "flash_attention",
        "offload_kv_cache_to_gpu",
        "eval_batch_size",
        "physical_batch_size",
        "parallel",
        "context_checkpoints",
        "num_experts",
    ]
    rows = []
    for row in channel_table(names):
        prog = "yes" if row["programmatic"] else "no (manual/detected-only)"
        rows.append(f"| {row['parameter']} | {row['channel']} | {prog} |")
    return rows


def _identical_configs(configs: list, best) -> list:
    """All tested configs identical to the winner (same load dimensions)."""
    bc = best.config
    out = []
    for c in configs:
        cc = c.config
        if (
            c.status.value == "passed"
            and c.context_length == best.context_length
            and cc.flash_attention == bc.flash_attention
            and cc.offload_kv_cache_to_gpu == bc.offload_kv_cache_to_gpu
            and cc.eval_batch_size == bc.eval_batch_size
            and cc.physical_batch_size == bc.physical_batch_size
            and cc.parallel == bc.parallel
            and cc.context_checkpoints == bc.context_checkpoints
        ):
            out.append(c)
    return out


def _median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def _typical_rows(configs: list, best) -> list[str]:
    """Winner config typicals across ALL its identical measurements.

    A single validation repeat can land in a slow window (thermal load,
    background activity); the median over identical runs is the honest
    headline. Flags it when the stored repeat is far below typical.
    """
    same = _identical_configs(configs, best)
    if len(same) < 2:
        return ["- Identical repeats: only this one measurement exists."]
    med_gen = _median([c.get_avg_generation_tok_s() for c in same])
    med_prompt = _median([c.get_avg_prompt_tok_s() for c in same])
    med_ttft = _median([c.get_avg_estimated_ttft_ms() for c in same])
    sel_gen = best.get_avg_generation_tok_s() or 0.0
    rows = [
        f"- Identical measurements of this exact config: {len(same)}",
        f"- Typical generation: {med_gen:.1f} tok/s (median)",
        f"- Typical prompt: {med_prompt:.0f} tok/s (median)",
        f"- Typical est. TTFT: {med_ttft:.0f} ms (median)",
        f"- Stored repeat: {sel_gen:.1f} tok/s",
    ]
    if med_gen and sel_gen < 0.9 * med_gen:
        rows.append(
            "- Note: the stored repeat ran slower than typical (slow window "
            "during validation); the config above repeatedly measured faster."
        )
    else:
        rows.append("- Stored repeat is representative of typical performance.")
    return rows


def _per_test_rows(best) -> list[str]:
    """Every benchmark test of the winner, explicitly (§34-transparency).

    Temperature: per-category override when the run overrode it, else the
    suite's frozen per-case default (benchmark temperatures are fixed per
    case; never tuned as runtime parameters).
    """
    gen = best.generation or {}
    overrides = gen.get("temperature_overrides") or {}
    stage = gen.get("stage", "unknown")
    quality_by_test = gen.get("quality_by_test") or {}
    lines = [
        "## Winner: every test explicitly",
        "",
        f"- Tested in stage: {stage}",
        f"- Benchmark style: {gen.get('style', 'balanced')}",
        "",
        "| Test | Category | Temp | Tokens (p/c) | Gen tok/s | Prompt tok/s | Est. TTFT | Quality | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m in best.metrics:
        temp = overrides.get(m.category, "suite default (case-fixed)")
        q = quality_by_test.get(m.test_name, {})
        if isinstance(q, dict) and q.get("checks_total"):
            qual = f"{q.get('overall', '?')} ({q.get('checks_passed')}/{q.get('checks_total')})"
        elif best.quality_score:
            qual = f"{best.quality_score.overall:.3f} (aggregate)"
        else:
            qual = "N/A"
        lines.append(
            f"| {m.test_name} | {m.category} | {temp} | "
            f"{m.prompt_tokens}/{m.completion_tokens} | "
            f"{m.generation_tok_s:.1f} | {m.prompt_tok_s:.0f} | "
            f"{m.estimated_ttft_ms:.0f} | {qual} | "
            f"{'PASS' if m.success else 'FAIL'} |"
        )
    lines += [""]
    return lines


def _verification_rows(best) -> list[str]:
    """Requested vs applied rows from the recorded load verification (§13)."""
    recorded = (best.generation or {}).get("load_verification") or {}
    channel = (best.generation or {}).get("load_channel", "REST")
    verification = recorded.get("verification", "UNKNOWN")
    rows = [
        f"- Load channel: {channel}",
        f"- Verification: {verification}",
    ]
    diffs = recorded.get("diffs") or {}
    if verification == "MATCH":
        rows.append("- Requested == applied for all sent keys.")
    elif verification == "UNKNOWN":
        rows.append("- No echo comparison available; requested values are NOT verified.")
    else:
        for key, pair in diffs.items():
            rows.append(f"- {key}: requested={pair.get('requested')} applied={pair.get('applied')}")
    if best.config.gpu_ratio is not None and channel == "REST":
        rows.append("- gpu_ratio was recorded but NOT applied (REST cannot set it).")
    return rows


def _context_trio(tested: list) -> dict:
    """Maximum stable / performance-optimal / balanced-recommended contexts.

    From measured data only: stable = largest passing ctx; optimal = fastest
    passing ctx; balanced = largest passing ctx within 80% of peak speed.
    """
    if not tested:
        return {"stable": None, "optimal": None, "recommended": None}
    stable = max(tested, key=lambda c: c.context_length)
    optimal = max(tested, key=lambda c: c.get_avg_generation_tok_s())
    peak = optimal.get_avg_generation_tok_s() or 0.0
    eligible = [c for c in tested if c.get_avg_generation_tok_s() >= 0.8 * peak]
    recommended = max(eligible, key=lambda c: c.context_length) if eligible else optimal
    return {
        "stable": stable.context_length,
        "optimal": optimal.context_length,
        "recommended": recommended.context_length,
    }


def save_best_report(
    run: OptimizationRun,
    out_dir: str | Path = "results",
    host: dict | None = None,
    extra: dict | None = None,
    generation: dict | None = None,
) -> Path | None:
    """Save a per-model .md file with the best settings. Returns path or None.

    Filename and every identity claim derive from the run's own stored
    model_id (never execution order or selection state). `generation`:
    recommended sampling settings {temperature, top_p, top_k, min_p,
    presence_penalty, reasoning, source}.
    """
    best = run.get_best_config()
    if best is None:
        logger.warning("No best config to report", model=run.model.id)
        return None

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Canonical model identity from the actual run (§6/§33).
    path = out / f"{sanitize_model_filename(run.model.id)}-best.md"

    bc = best.config
    qs = best.quality_score
    baseline = run.baseline_metrics or {}
    host = host or {}
    extra = extra or {}
    generation = generation or {}
    style = (best.generation or {}).get("style", "balanced")
    configs = list(run.configurations)
    model = run.model
    storage = host.get("storage") or {}
    drives = storage.get("drives") or []
    trio = _context_trio([c for c in configs if c.status.value == "passed"])

    def tried(get):
        return _tried(configs, get)

    lines = (
        [
            f"# Best settings: {run.model.id}",
            "",
            f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"- Profile: {run.profile.value} (style: {style})",
            f"- Run: {run.id} (status: {run.status.value})",
            f"- Optimizer: {run.optimizer_version}",
            f"- Elapsed: {run.duration_seconds:.1f}s" if run.duration_seconds else "- Elapsed: N/A",
            "",
            "## Model identity (from this run, not execution order)",
            "",
            f"- Model ID: {model.id}",
            f"- Display name: {model.name or 'unknown'}",
            f"- Architecture: {model.architecture or 'unknown'}",
            f"- Quantization: {model.quantization or 'unknown'}",
            f"- Variant/size: {model.size_bytes or 'unknown'} bytes",
            f"- MoE: {'yes ' + str(model.num_experts or '') if model.is_moe else 'no'}",
            "",
            "## Host",
            "",
            f"- GPU: {host.get('gpu', 'unknown')}",
            f"- CPU: {host.get('cpu', 'unknown')}",
            f"- CPU threads: {host.get('threads', 'unknown')}",
            f"- RAM: {host.get('ram', 'unknown')}",
            f"- KV cache quantization: "
            f"{host.get('kv_quant', 'Q4 starter (GUI preset, REST cannot set)')}",
            f"- LM Studio: {host.get('version', 'unknown')}",
            f"- Backend: {host.get('backend', 'unknown')}",
            "",
            "## Storage / swap (detection only, never modified)",
            "",
            f"- Drives: {_drives_str(drives)}",
            f"- Pagefile: {storage.get('pagefile', 'unknown')}",
            f"- Swap: free {storage.get('swap_free_gb', 'unknown')} / "
            f"total {storage.get('swap_total_gb', 'unknown')} GB",
            f"- Note: {storage.get('note', 'host snapshot at run start')}",
        ]
        + ([f"- NOTE: {host.get('note')}", ""] if host.get("note") else [""])
        + [
            "## Best load configuration (REST-settable)",
            "",
            "| Parameter | Value | Verdict |",
            "| --- | --- | --- |",
            f"| context_length | {best.context_length} | "
            f"{_verdict(best.context_length, tried(lambda c: c.context_length), 'model max')} |",
            f"| flash_attention | {_fmt(bc.flash_attention)} | "
            f"{_verdict(bc.flash_attention, tried(lambda c: c.config.flash_attention), True)} |",
            f"| offload_kv_cache_to_gpu (KV on GPU) | {_fmt(bc.offload_kv_cache_to_gpu)} | "
            f"{_verdict(bc.offload_kv_cache_to_gpu, tried(lambda c: c.config.offload_kv_cache_to_gpu), True)} |",
            f"| eval_batch_size | {_fmt(bc.eval_batch_size)} | "
            f"{_verdict(bc.eval_batch_size, tried(lambda c: c.config.eval_batch_size), 256)} |",
            f"| physical_batch_size (max concurrency input) | {_fmt(bc.physical_batch_size)} | "
            f"{_verdict(bc.physical_batch_size, tried(lambda c: c.config.physical_batch_size), SERVER_DEFAULTS['physical_batch_size'])} |",
            f"| parallel (max concurrency) | {_fmt(bc.parallel)} | "
            f"{_verdict(bc.parallel, tried(lambda c: c.config.parallel), SERVER_DEFAULTS['parallel'])} |",
            f"| context_checkpoints | {_fmt(bc.context_checkpoints)} | "
            f"{_verdict(bc.context_checkpoints, tried(lambda c: c.config.context_checkpoints), SERVER_DEFAULTS['context_checkpoints'])} |",
            f"| num_experts (MoE only) | {_fmt(bc.num_experts)} | model-dependent |",
            f"| gpu_ratio | {_fmt(bc.gpu_ratio)} | CLI/GUI-only, never sent (see notes) |",
            "",
            "Apply via REST: POST /api/v1/models/load with the flat parameters above "
            "(gpu_ratio excluded). GUI-only items (KV quant, offload ratio, threads) "
            "must be set on the host.",
            "",
            "## Control channels (per parameter, registry-driven)",
            "",
            "| Parameter | Channel | Programmatic |",
            "| --- | --- | --- |",
        ]
        + _channel_rows(best)
        + [
            "",
            "## Requested vs applied (winner load verification)",
            "",
        ]
        + _verification_rows(best)
        + [
            "",
            "## Winner typical across identical runs",
            "",
        ]
        + _typical_rows(configs, best)
        + [""]
        + _per_test_rows(best)
        + [
            "## Contexts (measured, not assumed)",
            "",
            f"- Maximum stable context: {trio['stable']}",
            f"- Performance-optimal context: {trio['optimal']}",
            f"- Balanced recommended context: {trio['recommended']}",
            "",
            "## Recommended generation settings (per chat request, not load)",
            "",
            f"- temperature: {generation.get('temperature', 'suite default per test')}",
            f"- top_p: {generation.get('top_p', 'server default')}",
            f"- top_k: {generation.get('top_k', 'server default')}",
            f"- min_p: {generation.get('min_p', 'server default')}",
            f"- presence_penalty: {generation.get('presence_penalty', 'server default')}",
            f"- reasoning: {generation.get('reasoning', 'off for benchmarks')}",
            f"- Source: {generation.get('source', 'not recorded')}",
            "",
        ]
    )
    if best.score_breakdown:
        lines += ["### Score breakdown", ""]
        # Actual phase weights (Phase A zeroes context); profile defaults
        # only as fallback for legacy data without recorded weights.
        weights = (best.generation or {}).get("score_weights") or run.profile_weights or {}
        if (best.generation or {}).get("score_context") is False:
            lines.append("- Phase A selection: context frozen, context contribution = 0.")
        for comp, norm in best.score_breakdown.items():
            w = weights.get(comp, 0)
            lines.append(f"- {comp}: normalized {norm:.3f} x weight {w:.2f} = {norm * w:.3f}")
        lines += [""]

    if baseline and baseline.get("generation_tok_s"):
        lines += [
            "## Baseline comparison",
            "",
            f"- Baseline generation: {baseline.get('generation_tok_s', 0):.1f} tok/s",
            f"- Baseline context: {baseline.get('context_length', 'N/A')}",
            f"- Baseline quality: {baseline.get('quality_overall', 'N/A')}",
            "",
        ]

    pareto = run.get_pareto_configs()
    tested = [c for c in run.configurations if c.status.value == "passed"]
    max_passing_ctx = max((c.context_length for c in tested), default=None)
    lines += [
        f"Max passing context: {max_passing_ctx} (among {len(tested)} passed configurations)",
        "",
    ]

    if tested:
        fastest = max(tested, key=lambda c: c.get_avg_generation_tok_s())
        best_q = max(
            (c for c in tested if c.quality_score),
            key=lambda c: c.quality_score.overall,
            default=None,
        )
        lines += ["## Alternative crowns (within this run)", ""]
        lines.append(
            f"- Best speed: ctx {fastest.context_length}, "
            f"{fastest.get_avg_generation_tok_s():.1f} tok/s "
            f"(quality {fastest.quality_score.overall:.3f})"
            if fastest.quality_score
            else f"- Best speed: ctx {fastest.context_length}, "
            f"{fastest.get_avg_generation_tok_s():.1f} tok/s"
        )
        if best_q is not None and best_q.quality_score is not None:
            lines.append(
                f"- Best quality: ctx {best_q.context_length}, "
                f"{best_q.quality_score.overall:.3f} "
                f"({best_q.get_avg_generation_tok_s():.1f} tok/s)"
            )
        lines.append(
            f"- Best context: ctx {max_passing_ctx} (largest passing; "
            f"see tested table for its speed)"
        )
        lines.append(
            f"- Best balanced (profile {run.profile.value}): "
            f"ctx {best.context_length}, score {_score_str(best.score)}"
        )
        from lm_optimizer.services.run_summary import speed_range as _speed_range

        _range = _speed_range(tested)
        if _range is not None:
            _fastest_cfg, _slowest_cfg = _range
            lines.append(
                f"- Speed range across run: best "
                f"{_fastest_cfg.get_avg_generation_tok_s():.1f} tok/s "
                f"(ctx {_fastest_cfg.context_length}) / worst "
                f"{_slowest_cfg.get_avg_generation_tok_s():.1f} tok/s "
                f"(ctx {_slowest_cfg.context_length})"
            )
        lines += [""]
    lines += ["## Tested configurations (passed)", ""]
    lines += ["| ctx | flash | KV | batch | gen tok/s | quality | score | status |"]
    lines += ["| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in tested:
        q = f"{c.quality_score.overall:.3f}" if c.quality_score else "N/A"
        lines.append(
            f"| {c.context_length} | {_fmt(c.config.flash_attention)} | "
            f"{'GPU' if c.config.offload_kv_cache_to_gpu else 'CPU'} | "
            f"{_fmt(c.config.eval_batch_size)} | {c.get_avg_generation_tok_s():.1f} | "
            f"{q} | {_score_str(c.score)} | PASS |"
        )
    if len(pareto) != len(tested):
        lines += ["", f"Pareto-optimal: {len(pareto)} of {len(tested)} passed."]
    lines += [""]

    failed = [c for c in run.configurations if c.status.value != "passed"]
    if failed:
        lines += ["## Failed configurations (with reasons)", ""]
        lines += ["| ctx | flash | KV | batch | status | reason |"]
        lines += ["| --- | --- | --- | --- | --- | --- |"]
        for c in failed:
            lines.append(
                f"| {c.context_length} | {_fmt(c.config.flash_attention)} | "
                f"{'GPU' if c.config.offload_kv_cache_to_gpu else 'CPU'} | "
                f"{_fmt(c.config.eval_batch_size)} | {c.status.value} | "
                f"{(c.error or '')[:200]} |"
            )
        lines += [""]

    lines += ["## Tried values per dimension (all attempts, incl. failed)", ""]
    lines += ["| Dimension | Tried | Best |"]
    lines += ["| --- | --- | --- |"]
    dim_specs = [
        ("context_length", lambda c: c.context_length),
        ("flash_attention", lambda c: c.config.flash_attention),
        ("kv_cache (GPU=True)", lambda c: c.config.offload_kv_cache_to_gpu),
        ("eval_batch_size", lambda c: c.config.eval_batch_size),
        ("physical_batch_size", lambda c: c.config.physical_batch_size),
        ("parallel", lambda c: c.config.parallel),
        ("context_checkpoints", lambda c: c.config.context_checkpoints),
    ]
    best_getters = {
        "context_length": best.context_length,
        "flash_attention": bc.flash_attention,
        "kv_cache (GPU=True)": bc.offload_kv_cache_to_gpu,
        "eval_batch_size": bc.eval_batch_size,
        "physical_batch_size": bc.physical_batch_size,
        "parallel": bc.parallel,
        "context_checkpoints": bc.context_checkpoints,
    }
    server_defaults = {
        "physical_batch_size": 512,
        "parallel": 4,
        "context_checkpoints": 32,
    }
    for name, get in dim_specs:
        vals = _tried(configs, get)
        shown = ", ".join(map(str, vals)) if vals else "none"
        best_val = best_getters[name]
        if best_val is None and name in server_defaults:
            best_val = f"server default ({server_defaults[name]})"
        lines.append(f"| {name} | {shown} | {best_val} |")
    lines += [""]

    notes = _prune_notes(run, configs)
    if notes:
        lines += ["## Why some values were never tried", ""]
        lines += [f"- {n}" for n in notes]
        lines += [""]

    pab = (run.benchmark_params or {}).get("phase_ab") or {}
    if pab:
        lines += ["## Phase A (speed first, then correctness)", ""]
        lines.append(
            f"- Speed context (frozen): {pab.get('speed_context', 'N/A')} | "
            f"Quality context (frozen): {pab.get('quality_context', 'N/A')} | "
            f"Budget: {pab.get('budget_class', 'N/A')}"
        )
        raw_id = str(pab.get("raw_fastest_id") or "")
        raw = next((c for c in configs if str(c.id) == raw_id), None)
        if raw is not None:
            q = f"{raw.quality_score.overall:.3f}" if raw.quality_score else "N/A"
            lines.append(
                f"- Raw fastest: ctx {raw.context_length}, "
                f"{raw.get_avg_generation_tok_s():.1f} tok/s, quality {q}, "
                f"status {raw.status.value} (shown even when rejected)"
            )
        lines.append(
            f"- Quality-safe winner: ctx {best.context_length}, "
            f"{best.get_avg_generation_tok_s():.1f} tok/s "
            f"(highest acceptable measured speed above threshold)"
        )
        for entry in pab.get("recovery_log") or []:
            if not isinstance(entry, dict):
                continue
            if entry.get("event") == "reconfirm":
                lines.append(
                    f"- Recovery reconfirm: {entry.get('config_id', '?')} "
                    f"tests {entry.get('tests', [])} passed={entry.get('passed')}"
                )
            elif entry.get("event") == "pairwise":
                lines.append(
                    f"- Recovery pairwise {entry.get('suspects', [])} "
                    f"passed={entry.get('passed')}"
                )
            elif entry.get("suspect"):
                lines.append(
                    f"- Recovery: {entry.get('suspect')} "
                    f"{entry.get('old_value')} -> {entry.get('new_value')} "
                    f"recheck_passed={entry.get('recheck_passed')} "
                    f"quality_after={entry.get('quality_after')}"
                )
        for note in pab.get("excluded_notes") or []:
            lines.append(f"- Not auto-tuned: {note}")
        lines += [""]
        if pab.get("phase_b_enabled"):
            pb = pab.get("phase_b_result") or {}
            lines += ["## Phase B (optional max context, runtime frozen)", ""]
            lines.append(f"- GPU-only max: {pb.get('gpu_only_max', 'N/A')}")
            lines.append(f"- System max: {pb.get('system_max', 'N/A')}")
            lines.append(f"- Model limit: {pb.get('model_limit', 'N/A')}")
            lines.append(f"- Hardware stable limit: {pb.get('hardware_stable_limit', 'N/A')}")
            lines.append(f"- Final capacity: {pb.get('final_capacity', 'N/A')}")
            if pb.get("spill_note"):
                lines.append(f"- Note: {pb.get('spill_note')}")
            lines += [""]

    lines += [
        "## How quality and score work",
        "",
        "- Each test output gets 6 deterministic heuristic checks (JSON validity, "
        "keywords, code structure, truncation, repetition); overall = mean of checks.",
        "- Aggregate quality = mean over tests; checks shown as X/30 total.",
        "- A config counts only if aggregate quality >= profile threshold "
        "(speed 0.95, balanced 0.97, context/quality higher).",
        "- Speeds are medians over repetitions; quality reads the most complete sample.",
        "- Score = run-relative / model-relative / hardware-relative normalized "
        "components x profile weights (see breakdown above).",
        "",
    ]

    for key, value in extra.items():
        lines += [f"## {key}", "", str(value), ""]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Best-settings report saved", model=run.model.id, path=str(path))
    return path
