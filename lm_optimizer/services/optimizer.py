"""Adaptive optimization engine - hardware-agnostic scoring."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from lm_optimizer.config import config
from lm_optimizer.database.repositories import (
    config_repo,
    hardware_repo,
    model_repo,
    run_repo,
    settings_repo,
)
from lm_optimizer.domain.models import (
    DEFAULT_PROFILE_WEIGHTS,
    ConfigurationResult,
    ConfigurationStatus,
    HardwareInfo,
    LoadConfiguration,
    ModelIdentity,
    OptimizationPhase,
    OptimizationProfile,
    OptimizationRun,
    OptimizationStage,
    ProfileWeights,
    RunStatus,
)
from lm_optimizer.logging_config import get_logger
from lm_optimizer.scoring.normalization import (
    compute_bounds,
    score_result_breakdown,
    weighted_score,
)
from lm_optimizer.services.benchmark import BenchmarkService
from lm_optimizer.services.foreign_watchdog import BACKEND_LMSTUDIO, ForeignModelWatchdog
from lm_optimizer.services.fingerprint import (
    compute_fingerprint,
    fingerprint_diff,
    hardware_fingerprint,
    should_revalidate_config,
    filter_configs_for_revalidation,
)
from lm_optimizer.services.lm_studio import LMStudioClient
from lm_optimizer.services.quality import QualityEvaluator
from lm_optimizer.services.search_space import SearchSpace, SearchSpaceGenerator

logger = get_logger(__name__)


class PauseRequested(Exception):
    """Graceful user-requested pause: state already persisted, not a failure."""


EXPERIMENTAL_REASONS = {
    "enable_rope": "RoPE parameters enabled (experimental)",
    "enable_cpu_moe": "CPU-MoE offload enabled (experimental)",
    "enable_speculative": "Speculative decoding enabled (experimental)",
}


def parse_experimental_flags(
    enable_rope: bool | None = False,
    enable_cpu_moe: bool | None = False,
    enable_speculative: bool | None = False,
) -> dict:
    """Experimental tail opt-ins. All default OFF.

    RoPE / CPU-MoE / speculative probes run only as a tail after the FINAL
    winner, never inside the main sweep. `None` means "not specified" and
    resolves to OFF (use `resolve_experimental_optins` when the distinction
    between unspecified and explicit OFF matters for prompting).
    """
    return {
        "enable_rope": bool(enable_rope),
        "enable_cpu_moe": bool(enable_cpu_moe),
        "enable_speculative": bool(enable_speculative),
    }


EXPERIMENTAL_QUESTIONS = {
    "enable_rope": "Enable RoPE (rotary position embedding) probes?",
    "enable_cpu_moe": "Enable CPU-MoE (Mixture of Experts) probes?",
    "enable_speculative": "Enable Speculative decoding probes?",
}


def resolve_experimental_optins(
    requested: dict,
    ask: Callable[[str], bool] | None = None,
) -> dict:
    """Resolve experimental opt-ins distinguishing unspecified from explicit OFF.

    - Explicit True/False values always win (explicit False counts as a user
      decision and is never re-prompted).
    - `None` (unspecified) values are asked via `ask(question)` when an
      asker is provided, otherwise resolve to OFF.
    - `ask` must never be called for explicitly decided flags.
    """
    resolved: dict = {}
    for key in ("enable_rope", "enable_cpu_moe", "enable_speculative"):
        value = (requested or {}).get(key)
        if value is None:
            resolved[key] = bool(ask(EXPERIMENTAL_QUESTIONS[key])) if ask is not None else False
        else:
            resolved[key] = bool(value)
    return resolved


def apply_experimental_mark(run: OptimizationRun, flags: dict | None) -> OptimizationRun:
    """Mark a run experimental when any opt-in flag is set.

    `is_experimental` means "any experiment resolved ON at request time"
    (requested→resolved mapping in `build_experimental_detail`). It does NOT
    mean the tail probe actually ran — see `exercised` in the detail dict.

    Any True flag sets `run.is_experimental = True` with a reason string;
    all-False (or missing) flags leave the run clean.
    """
    flags = flags or {}
    active = [key for key in EXPERIMENTAL_REASONS if flags.get(key)]
    if active:
        run.is_experimental = True
        run.experimental_reason = "; ".join(EXPERIMENTAL_REASONS[key] for key in active)
    else:
        run.is_experimental = False
        run.experimental_reason = None
    return run


EXPERIMENTAL_FLAGS = ("enable_rope", "enable_cpu_moe", "enable_speculative")


# ============================================================
# Preset Definitions (spec §4.2)
# ============================================================

PRESET_DEFINITIONS: dict[str, dict] = {
    "quick": {
        "max_speed_probes": 12,
        "max_finalists": 2,
        "benchmark_repetitions": 1,
        "validation_repetitions_min": 2,
        "validation_repetitions_max": 2,
        "recovery_budget": 1,
        "selected_stages": ["speed", "quality", "validation"],
        "context_strategy_modes": ["winner"],
        "final_sandbox_mode": "off",
    },
    "standard": {
        "max_speed_probes": 25,
        "max_finalists": 4,
        "benchmark_repetitions": 3,
        "validation_repetitions_min": 2,
        "validation_repetitions_max": 5,
        "recovery_budget": 3,
        "selected_stages": ["speed", "quality", "recovery", "validation"],
        "context_strategy_modes": ["winner", "model_first_kv_cpu"],
        "final_sandbox_mode": "off",
    },
    "thorough": {
        "max_speed_probes": 40,
        "max_finalists": 6,
        "benchmark_repetitions": 3,
        "validation_repetitions_min": 3,
        "validation_repetitions_max": 5,
        "recovery_budget": 5,
        "selected_stages": ["speed", "quality", "recovery", "validation", "context"],
        "context_strategy_modes": [
            "winner",
            "gpu_resident",
            "model_first_kv_cpu",
            "system_spillover",
        ],
        "final_sandbox_mode": "all",
    },
    "custom": {
        # User-defined values - no defaults, must be provided
    },
}

VALID_PRESETS = frozenset(PRESET_DEFINITIONS.keys())
VALID_STAGES = frozenset(
    [
        "preflight",
        "speed",
        "quality",
        "recovery",
        "validation",
        "context",
        "sandbox",
        "experimental",
    ]
)
VALID_CONTEXT_STRATEGIES = frozenset(
    [
        "winner",  # frozen runtime from Phase A
        "gpu_resident",  # GPU-resident weights + KV on GPU
        "model_first_kv_cpu",  # model on GPU, KV on CPU
        "system_spillover",  # partial GPU offload (if CLI channel available)
    ]
)
VALID_SANDBOX_MODES = frozenset(["off", "one", "all", "custom"])


def resolve_preset(
    preset: str, overrides: dict | None = None
) -> tuple[dict, list[str]]:
    """Resolve a preset to concrete budgets, applying user overrides.

    Returns (resolved_settings, warnings).
    """
    if preset not in VALID_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset}. Valid: {', '.join(sorted(VALID_PRESETS))}"
        )

    base = PRESET_DEFINITIONS[preset].copy()
    overrides = overrides or {}
    warnings: list[str] = []

    # Apply overrides. Custom mode accepts any documented key (it has no
    # defaults of its own); named presets only extend their own keys.
    _known_keys = {
        "max_speed_probes",
        "max_finalists",
        "benchmark_repetitions",
        "validation_repetitions_min",
        "validation_repetitions_max",
        "recovery_budget",
        "selected_stages",
        "context_strategy_modes",
        "final_sandbox_mode",
        "final_sandbox_preset_ids",
        "final_sandbox_custom_prompts",
        "sandbox_compare_baseline",
    }
    if preset == "custom":
        base = {k: v for k, v in overrides.items() if k in _known_keys}
    else:
        for key, value in overrides.items():
            if key in base:
                base[key] = value
            elif key in _known_keys:
                # A documented budget the preset did not define: accept it.
                base[key] = value
            else:
                warnings.append(f"Unknown preset override: {key}")

    # Validate stage dependencies
    stage_warnings = validate_stage_dependencies(base.get("selected_stages", []))
    warnings.extend(stage_warnings)

    # Validate context strategies
    for strat in base.get("context_strategy_modes", []):
        if strat not in VALID_CONTEXT_STRATEGIES:
            warnings.append(f"Unknown context strategy: {strat}")

    # Validate sandbox mode
    sandbox_mode = base.get("final_sandbox_mode", "off")
    if sandbox_mode not in VALID_SANDBOX_MODES:
        warnings.append(f"Unknown sandbox mode: {sandbox_mode}")

    return base, warnings


def validate_stage_dependencies(selected_stages: list[str]) -> list[str]:
    """Validate stage dependencies and return warnings.

    Rules:
    - preflight is mandatory (safety + context baseline)
    - quality requires speed (or a user-selected compatible saved config)
    - recovery requires quality failure + safe rollback reference
    - validation requires quality-passing candidate (or explicit prior config)
    - context requires a selected runtime config
    - sandbox requires a valid selected config
    - experimental requires explicit opt-in
    """
    warnings: list[str] = []
    stages = set(selected_stages)

    # preflight is always implied
    if "preflight" not in stages:
        warnings.append("'preflight' is mandatory for safety; implicitly added")
        stages.add("preflight")

    if "quality" in stages and "speed" not in stages:
        warnings.append(
            "'quality' requires 'speed' phase (or a compatible saved config); "
            "results will be marked 'quality not verified'"
        )

    if "recovery" in stages:
        if "quality" not in stages:
            warnings.append(
                "'recovery' requires 'quality' phase; will be skipped if no quality failure"
            )

    if "validation" in stages:
        if "quality" not in stages:
            warnings.append(
                "'validation' normally requires 'quality-passing' candidate; "
                "without it, labeled 'benchmark-retested but quality not verified'"
            )

    if "context" in stages:
        if not stages & {"speed", "quality", "validation"}:
            warnings.append(
                "'context' requires a selected runtime config from a prior phase"
            )

    if "sandbox" in stages:
        if not stages & {"validation", "context"}:
            warnings.append(
                "'sandbox' runs against a validated config; "
                "without prior validation, labeled 'unvalidated config'"
            )

    if "experimental" in stages:
        warnings.append("'experimental' runs only with explicit opt-in and is marked experimental")

    return warnings


def build_experimental_detail(
    requested_raw: dict | None,
    resolved: dict | None,
    advanced_settings: dict | None = None,
) -> dict:
    """Requested vs resolved vs applied vs exercised experimental metadata.

    - `requested`: raw CLI/API values (True/False/None; None = unspecified).
    - `resolved`: bool dict after prompting/default-OFF (what `is_experimental` uses).
    - `applied`: resolved minus capability blocks (speculative without a usable
      draft model is never applied).
    - `exercised`: all False until a tail probe actually runs (no tail runner
      exists yet; the runner flips its flag to True when it runs).
    - `skipped`: {flag: reason} for capability blocks.
    """
    requested_raw = requested_raw or {}
    resolved = resolved or {}
    advanced_settings = advanced_settings or {}
    requested = {k: requested_raw.get(k) for k in EXPERIMENTAL_FLAGS}
    resolved_b = {k: bool(resolved.get(k, False)) for k in EXPERIMENTAL_FLAGS}
    applied = dict(resolved_b)
    skipped: dict[str, str] = {}
    draft = advanced_settings.get("speculative_draft_model") or advanced_settings.get(
        "speculative_draft"
    )
    if resolved_b.get("enable_speculative") and not draft:
        applied["enable_speculative"] = False
        skipped["enable_speculative"] = "no usable draft model"
    exercised = {k: False for k in EXPERIMENTAL_FLAGS}
    return {
        "requested": requested,
        "resolved": resolved_b,
        "applied": applied,
        "exercised": exercised,
        "skipped": skipped,
    }


def format_diagnostic(exc: BaseException) -> dict:
    """Diagnostic record for unexpected failures.

    Keeps the user-friendly message and exception type separate from the
    full traceback: normal UI/reporting surfaces show type + message only,
    while logs retain the traceback for debugging.
    """
    import traceback as _tb

    return {
        "type": type(exc).__name__,
        "message": str(exc) or type(exc).__name__,
        "traceback": "".join(_tb.format_exception(type(exc), exc, exc.__traceback__)),
    }


# Canonical skippable phases (CLI --skip-phases, API skip_phases, GUI).
PHASE_SKIP_NAMES = ("speed", "quality", "recovery", "validation", "context")


def parse_phase_skips(value) -> set[str]:
    """Parse user phase skips into a validated set (case-insensitive).

    Accepts a comma string or a list/tuple/set. Raises ValueError naming
    the first unknown phase (callers fail fast: CLI exit 2).
    """
    # Typer may hand us an OptionInfo object when a CLI default is not
    # resolved (e.g. resume() called directly from optimize()). Unwrap it.
    try:
        import typer.models as _tm
        if isinstance(value, _tm.OptionInfo):
            value = getattr(value, "default", None)
    except Exception:
        pass
    if value is None:
        return set()
    if isinstance(value, str):
        items = [p.strip() for p in value.split(",")]
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = [str(p).strip() for p in value]
    else:
        # Unknown scalar -> skip nothing (never crash on a default value).
        return set()
    names = {p.lower() for p in items if p}
    unknown = sorted(names - set(PHASE_SKIP_NAMES))
    if unknown:
        raise ValueError(
            f"Unknown phase(s) to skip: {', '.join(unknown)} "
            f"(expected one of: {', '.join(PHASE_SKIP_NAMES)})"
        )
    return names


def phase_skips_of(advanced_settings) -> set[str]:
    """Lenient skip set for run/resume paths (warn + ignore unknowns)."""
    try:
        return parse_phase_skips((advanced_settings or {}).get("skip_phases"))
    except ValueError as e:
        logger.warning("Ignoring invalid skip_phases", error=str(e)[:160])
        return set()


def compose_granular_skips(advanced_settings) -> list[str]:
    """Merge granular mode flags into an explicit phase-skip list.

    Shared by the CLI preflight and the web start endpoint so both surfaces
    produce the SAME skip set for the same flags (§23 acceptance: the web
    endpoint accepts granular flags).
    """
    adv = advanced_settings or {}
    skips: set[str] = set()
    if adv.get("skip_phases"):
        try:
            skips |= parse_phase_skips(adv.get("skip_phases"))
        except ValueError:
            pass
    if adv.get("speed_only"):
        skips |= {"quality", "recovery", "validation", "context"}
    if adv.get("quality_only") or adv.get("reasoning_sweep"):
        skips |= {"speed", "recovery", "validation", "context"}
    if adv.get("mtp_test"):
        skips |= {"quality", "recovery", "validation", "context"}
    return sorted(skips)


def granular_quality_frontier(
    skips: set[str], advanced: dict | None, baseline_result
) -> dict:
    """Frontier for quality-only / reasoning-only runs (speed phase skipped).

    Granular quality modes have no speed frontier; the baseline (anchor)
    config becomes the single candidate so the quality battery still measures
    something meaningful. Returns the frontier dict (empty finalists when no
    baseline is available).
    """
    frontier = {"winner": None, "finalists": [], "contrarian": None}
    if "speed" not in skips:
        return frontier
    adv = advanced or {}
    if not (adv.get("quality_only") or adv.get("reasoning_sweep")):
        return frontier
    if baseline_result is not None:
        frontier["finalists"] = [baseline_result]
    return frontier


def run_skip_phases(run) -> set[str]:
    """Skip set for a persisted run: memory first, DB fallback.

    `advanced_settings` lives only in memory; `benchmark_params` is the
    persisted copy (written by prepare_run and _apply_resume_skips).
    """
    skips = phase_skips_of(getattr(run, "advanced_settings", None))
    if skips:
        return skips
    return phase_skips_of(getattr(run, "benchmark_params", None))


#: Run fields frozen at start and restored verbatim on resume (spec §9).
PERSISTED_RUN_SETTINGS = (
    "min_generation_tok_s",
    "enforce_min_generation_speed",
    "run_preset",
    "selected_stages",
    "max_speed_probes",
    "max_finalists",
    "benchmark_repetitions_override",
    "validation_repetitions_min",
    "validation_repetitions_max",
    "recovery_budget",
    "context_strategy_modes",
    "final_sandbox_mode",
    "final_sandbox_preset_ids",
    "final_sandbox_custom_prompts",
    "sandbox_compare_baseline",
)


def build_resolved_settings(run) -> dict:
    """Snapshot of every frozen run setting (persisted in benchmark_params)."""
    return {key: getattr(run, key, None) for key in PERSISTED_RUN_SETTINGS}


def apply_persisted_run_settings(run):
    """Restore frozen settings on resume; legacy runs keep safe defaults.

    New fields are not DB columns (no migration). They are persisted as
    ``benchmark_params["resolved_settings"]`` so a resumed run keeps its
    original meaning even when global defaults changed meanwhile.
    """
    params = getattr(run, "benchmark_params", None) or {}
    saved = params.get("resolved_settings")
    if not isinstance(saved, dict):
        return run
    for key in PERSISTED_RUN_SETTINGS:
        if key in saved and saved[key] is not None:
            try:
                setattr(run, key, saved[key])
            except Exception:
                continue
    return run


@dataclass
class OptimizationState:
    """Current state of optimization run."""

    run: OptimizationRun
    search_space: SearchSpace
    tested_configs: list[ConfigurationResult] = field(default_factory=list)
    best_config: ConfigurationResult | None = None
    pareto_frontier: list[ConfigurationResult] = field(default_factory=list)
    current_config_id: UUID | None = None
    errors: list[str] = field(default_factory=list)
    should_pause: bool = False
    should_cancel: bool = False
    # Track failed/OOM configs to avoid refining inferior regions
    failed_regions: set = field(default_factory=set)
    # Server demands flash attention (non-F16 KV quant): skip flash=False (R28)
    flash_required: bool = False
    # Compact live event log for the Web UI (capped, no verbose internals).
    event_log: list[dict] = field(default_factory=list)
    started_at: datetime | None = None
    # Probe attempt counts {repr(candidate_key): n} for resume retry policy.
    probe_attempts: dict = field(default_factory=dict)
    # Recovery execution cursor (see _persist_recovery_step).
    recovery_cursor: dict | None = None
    # Monotonic checkpoint counter (stale/newer comparison on resume).
    checkpoint_seq: int = 0
    # Phase A/B strategy state (all defaulted; legacy flows ignore them).
    phase: str = "speed"
    speed_context: int | None = None
    quality_context: int | None = None
    budget_class: str | None = None
    probe_repetitions: int = 1
    speed_finalists: list = field(default_factory=list)  # config id strings
    raw_fastest: ConfigurationResult | None = None
    baseline_result: ConfigurationResult | None = None
    recovery_log: list = field(default_factory=list)  # RecoveryAttempt dicts
    excluded_notes: list = field(default_factory=list)
    phase_b_enabled: bool = False
    phase_b_result: dict | None = None
    planned_total: int = 0
    # Phase A scoring: context is frozen, so it must not enter the winner
    # score. Legacy flows keep score_context=True (behavior unchanged).
    score_context: bool = True


def _display_total(state) -> int:
    """Progress denominator: planned Phase-A units when known, else estimate.

    `planned_total` grows as phases unfold (speed plan → quality suites →
    recovery probes → validation reps); never below tested units, minimum 1.
    Legacy flows without planning fall back to the Cartesian estimate.
    """
    try:
        tested = len(getattr(state, "tested_configs", []) or [])
    except Exception:
        tested = 0
    try:
        planned = int(getattr(state, "planned_total", 0) or 0)
    except Exception:
        planned = 0
    if planned > 0:
        return max(planned, tested, 1)
    try:
        space = getattr(state, "search_space", None)
        est = int(space.estimate_size()) if space else 0
    except Exception:
        est = 0
    return max(est, tested, 1)


class AdaptiveOptimizer:
    """Adaptive optimizer with 5-stage search, hardware-agnostic scoring."""

    def __init__(
        self,
        client: LMStudioClient,
        benchmark_service: BenchmarkService,
        quality_evaluator: QualityEvaluator,
        search_generator: SearchSpaceGenerator,
    ):
        self.client = client
        self.benchmark = benchmark_service
        self.quality = quality_evaluator
        self.search_generator = search_generator
        self.state: OptimizationState | None = None

    async def optimize(
        self,
        model: ModelIdentity,
        hardware: HardwareInfo,
        profile: OptimizationProfile,
        profile_weights: ProfileWeights | None = None,
        quality_threshold: float = 0.97,
        advanced_settings: dict | None = None,
        baseline_config: LoadConfiguration | None = None,
        style: str = "balanced",
        progress_cb=None,
    ) -> OptimizationRun:
        """Run full optimization with baseline capture and hardware-agnostic scoring.

        `progress_cb(stage, tested, total)` is informational only (CLI bars,
        never control flow); None disables it.
        """
        run, baseline_config, advanced_settings = await self.prepare_run(
            model, hardware, profile, profile_weights, quality_threshold,
            advanced_settings, style, baseline_config, progress_cb,
        )
        return await self.execute(model, profile, run, baseline_config, advanced_settings)

    async def prepare_run(
        self,
        model: ModelIdentity,
        hardware: HardwareInfo,
        profile: OptimizationProfile,
        profile_weights: ProfileWeights | None = None,
        quality_threshold: float = 0.97,
        advanced_settings: dict | None = None,
        style: str = "balanced",
        baseline_config: LoadConfiguration | None = None,
        progress_cb=None,
    ) -> tuple[OptimizationRun, LoadConfiguration | None, dict]:
        """Create run + state synchronously (route returns before background work).

        Same steps optimize() always ran first (now shared); returns the run,
        resolved baseline and normalized settings. execute() continues from here.
        """
        self._progress_cb = progress_cb
        # Save hardware and model
        hardware_id = hardware_repo.save(hardware)
        model_repo.save(model)

        advanced_settings = advanced_settings or {}
        self.benchmark.style = style

        # --- Resolve preset and apply budgets (spec §2, §4.2) ---
        preset = advanced_settings.get("run_preset", "standard")

        def _override(key):
            """Explicit non-empty override only; empty CLI lists are 'unset'."""
            value = advanced_settings.get(key)
            if isinstance(value, (list, tuple)) and not value:
                return None
            return value

        preset_overrides = {
            k: _override(k)
            for k in (
                "max_speed_probes",
                "max_finalists",
                "benchmark_repetitions",
                "validation_repetitions_min",
                "validation_repetitions_max",
                "recovery_budget",
                "selected_stages",
                "context_strategy_modes",
                "final_sandbox_mode",
                "final_sandbox_preset_ids",
                "final_sandbox_custom_prompts",
                "sandbox_compare_baseline",
            )
            if _override(k) is not None
        }
        resolved_preset, preset_warnings = resolve_preset(preset, preset_overrides)
        for w in preset_warnings:
            logger.warning("Preset resolution", warning=w)

        # --- Speed floor (spec §3) ---
        min_generation_tok_s = advanced_settings.get("min_generation_tok_s", 2.0)
        enforce_min_generation_speed = advanced_settings.get(
            "enforce_min_generation_speed", True
        )

        # --- Experimental tail opt-ins ---
        requested_raw = {k: advanced_settings.get(k) for k in EXPERIMENTAL_FLAGS}
        exp_flags = parse_experimental_flags(
            enable_rope=advanced_settings.get("enable_rope", False),
            enable_cpu_moe=advanced_settings.get("enable_cpu_moe", False),
            enable_speculative=advanced_settings.get("enable_speculative", False),
        )
        advanced_settings.update(exp_flags)
        if exp_flags["enable_rope"]:
            quality_threshold = max(quality_threshold, 0.98)
            logger.warning(
                "RoPE experimental enabled - applying stronger quality validation",
                threshold=quality_threshold,
            )

        # Use profile's default threshold if not explicitly overridden
        if quality_threshold is None:
            from lm_optimizer.profiles.registry import ProfileRegistry

            reg = ProfileRegistry()
            try:
                prof = reg.get(profile.value)
                quality_threshold = prof.minimum_quality
            except Exception:
                quality_threshold = 0.97

        # Keep the quality evaluator in sync with the (possibly RoPE-raised)
        # threshold. Otherwise the run stores 0.98 while evaluation uses 0.97.
        try:
            if getattr(self, "quality", None) is not None and getattr(
                self.quality, "config", None
            ) is not None:
                self.quality.config.minimum_score = quality_threshold
        except Exception:
            pass

        # Resolve profile weights
        resolved_weights = profile_weights or DEFAULT_PROFILE_WEIGHTS.get(profile)
        if resolved_weights:
            weights_dict = resolved_weights.to_dict()
        else:
            weights_dict = {}

        # Compose phase skips from preset + granular flags
        ordered_stages: list[str] = []
        # Custom mode: an explicit stage list is the authority (--stage).
        if preset == "custom":
            explicit = (
                advanced_settings.get("selected_stages")
                or resolved_preset.get("selected_stages")
                or []
            )
            ordered_stages = [s for s in explicit if s in VALID_STAGES]
        else:
            ordered_stages = list(resolved_preset.get("selected_stages", []))
        preset_stages = set(ordered_stages)
        # An explicit --optimize-context opt-in wins over a preset that does
        # not list the context stage (granular flags can still skip it).
        if advanced_settings.get("optimize_context") and "context" not in preset_stages:
            preset_stages.add("context")
            ordered_stages.append("context")
        granular_skips = set(compose_granular_skips(advanced_settings))
        if preset_stages:
            # Preset stages define what TO run; convert to skip set.
            all_stages = {"speed", "quality", "recovery", "validation", "context"}
            preset_skips = sorted(all_stages - preset_stages)
            final_skips = sorted(set(preset_skips) | granular_skips)
        else:
            # No stage plan: granular skips + explicit skip_phases only.
            explicit_skips = set(phase_skips_of(advanced_settings))
            final_skips = sorted(explicit_skips | granular_skips)

        # Resolve budget parameters from preset (explicit overrides win).
        def _budget(key: str):
            value = advanced_settings.get(key)
            if value is None:
                value = resolved_preset.get(key)
            return value

        max_speed_probes = _budget("max_speed_probes")
        max_finalists = _budget("max_finalists")
        benchmark_repetitions = _budget("benchmark_repetitions")
        validation_repetitions_min = _budget("validation_repetitions_min")
        validation_repetitions_max = _budget("validation_repetitions_max")
        recovery_budget = _budget("recovery_budget")

        # Apply the resolved repetition budget to the benchmark service so a
        # preset actually changes the workload (previously it was only stored).
        try:
            if benchmark_repetitions is not None and getattr(
                self.benchmark, "config", None
            ) is not None:
                self.benchmark.config.repetitions = max(1, int(benchmark_repetitions))
        except (TypeError, ValueError) as e:
            logger.warning("Invalid benchmark_repetitions override", error=str(e))

        # Create run
        run = OptimizationRun(
            model=model,
            hardware=hardware,
            hardware_id=hardware_id,
            profile=profile,
            profile_weights=weights_dict,
            quality_threshold=quality_threshold,
            search_space={},
            is_experimental=False,
            experimental_reason=None,
            advanced_settings=advanced_settings or {},
            benchmark_params={
                "benchmark_repetitions": benchmark_repetitions,
                "validation_repetitions": validation_repetitions_max,
                "validation_repetitions_min": validation_repetitions_min,
                "validation_repetitions_max": validation_repetitions_max,
                "warmup_runs": config.optimization.warmup_runs,
                "seed": 42,
                "deterministic": True,
                "temperature": "per-benchmark-case-fixed",
                "style": style,
                "candidate_generation": "deterministic-sorted-sampling",
                "selection_threshold": float(advanced_settings.get("selection_threshold", 0.05)),
                "workload_type": str(advanced_settings.get("workload_type", "interactive")).lower(),
                "speed_probe_repetitions": advanced_settings.get("speed_probe_repetitions"),
                "recovery_budget": recovery_budget,
                "quality_finalist_limit": max_finalists,
                "max_speed_probes": max_speed_probes,
                "optimize_context": bool(advanced_settings.get("optimize_context", False)),
                "skip_phases": final_skips,
                # Preset metadata
                "run_preset": preset,
                "resolved_preset": resolved_preset,
            },
            # Speed floor
            min_generation_tok_s=min_generation_tok_s,
            enforce_min_generation_speed=enforce_min_generation_speed,
            # Preset
            run_preset=preset,
            # Stage selection
            selected_stages=ordered_stages,
            max_speed_probes=max_speed_probes,
            max_finalists=max_finalists,
            benchmark_repetitions_override=benchmark_repetitions,
            validation_repetitions_min=validation_repetitions_min,
            validation_repetitions_max=validation_repetitions_max,
            recovery_budget=recovery_budget,
            # Context strategies
            context_strategy_modes=resolved_preset.get("context_strategy_modes", []),
            # Sandbox
            final_sandbox_mode=resolved_preset.get("final_sandbox_mode", "off"),
            final_sandbox_preset_ids=resolved_preset.get("final_sandbox_preset_ids", []),
            final_sandbox_custom_prompts=resolved_preset.get("final_sandbox_custom_prompts", []),
            sandbox_compare_baseline=resolved_preset.get("sandbox_compare_baseline", False),
            # Resolved preset for reporting
            resolved_preset=resolved_preset,
        )
        # Any experimental opt-in marks the run (single marking path).
        apply_experimental_mark(run, exp_flags)
        run.benchmark_params["experimental_detail"] = build_experimental_detail(
            requested_raw, exp_flags, advanced_settings
        )
        # Freeze every resolved setting for resume/reporting (spec §9).
        run.benchmark_params["resolved_settings"] = build_resolved_settings(run)

        # Compute and store fingerprint for this run
        from lm_optimizer.services import lms_cli as _lms_cli
        from lm_optimizer.benchmark.suite_version import BENCHMARK_SUITE_VERSION

        lms_version = "unknown"
        try:
            lms_version = _lms_cli.lms_version()
        except Exception:
            pass

        hw_fp = hardware_fingerprint(hardware)
        fp_components = {
            "model_id": model.id,
            "quantization": model.quantization,
            "architecture": model.architecture,
            "lm_studio_version": lms_version,
            "hardware_fingerprint": hw_fp,
            "benchmark_suite_version": BENCHMARK_SUITE_VERSION,
        }
        run.benchmark_params["fingerprint"] = compute_fingerprint(
            fp_components["model_id"],
            fp_components["quantization"],
            fp_components["architecture"],
            fp_components["lm_studio_version"],
            fp_components["hardware_fingerprint"],
            fp_components["benchmark_suite_version"],
        )
        run.benchmark_params["fingerprint_components"] = fp_components

        # Generate search space
        search_space = self.search_generator.generate(model, hardware, profile, advanced_settings)
        run.search_space = search_space.to_dict()

        # Initialize state
        self.state = OptimizationState(run=run, search_space=search_space)
        # Cross-run lesson: preskip flash-off when a previous run learned it.
        self._apply_persisted_flash_advice()

        # Baseline config captured for the Phase A anchor (no expensive
        # baseline benchmark; the S0 speed probe is the anchor reference).
        baseline_config = await self._resolve_baseline(model, baseline_config)

        # Save initial run
        run.status = RunStatus.RUNNING
        run.started_at = datetime.now()
        run_repo.save(run)
        return run, baseline_config, advanced_settings or {}

    async def execute(
        self,
        model: ModelIdentity,
        profile: OptimizationProfile,
        run: OptimizationRun,
        baseline_config: LoadConfiguration | None,
        advanced_settings: dict,
    ) -> OptimizationRun:
        """Run all phases from a prepared run (background-safe entry point)."""
        # First Ctrl+C arms graceful pause (second aborts); restored on exit.
        self._install_pause_handler()

        # Initialize state before stages so Ctrl+C always has something to persist.
        from lm_optimizer.storage.run_checkpoint import save_checkpoint as _save_ckpt

        self.state.started_at = datetime.now()
        self._log_event("Optimization started", f"model={model.id} profile={profile.value}")
        _save_ckpt(self.state, reason="started")
        await self._broadcast({"stage": run.stage.value, "event": "started"})

        try:
            # Phase A/B strategy: SPEED -> QUALITY -> RECOVERY ->
            # FINAL_VALIDATION -> optional CONTEXT. Legacy _stage_* methods
            # below are retained (tested) but no longer drive optimization.
            await self._run_phase_ab(advanced_settings, baseline_config)

        except asyncio.CancelledError:
            # Interrupt-safe: persist BEFORE acknowledging cancellation.
            run.status = RunStatus.CANCELLED
            run.stage = OptimizationStage.CANCELLED
            run.completed_at = datetime.now()
            if run.started_at:
                run.duration_seconds = (run.completed_at - run.started_at).total_seconds()
            self._log_event(
                "Cancelled — checkpoint saved", f"tested={len(self.state.tested_configs)}"
            )
            _save_ckpt(self.state, reason="cancelled")
            run_repo.save(run)
            await self._broadcast({"stage": "cancelled", "event": "cancelled"})
            raise
        except KeyboardInterrupt:
            run.status = RunStatus.INTERRUPTED
            run.stage = OptimizationStage.INTERRUPTED
            run.completed_at = datetime.now()
            if run.started_at:
                run.duration_seconds = (run.completed_at - run.started_at).total_seconds()
            self._log_event(
                "Interrupted — checkpoint saved", f"tested={len(self.state.tested_configs)}"
            )
            _save_ckpt(self.state, reason="interrupted")
            run_repo.save(run)
            await self._broadcast({"stage": "interrupted", "event": "interrupted"})
            raise
        except PauseRequested:
            # Graceful user pause: _pause_now already persisted everything
            # (checkpoint, DB, unload) and marked PAUSED; just return.
            return await self._return_paused(run)
        except Exception as e:
            diag = format_diagnostic(e)
            logger.error(
                "Optimization failed",
                error=diag["message"],
                exc_type=diag["type"],
                traceback=diag["traceback"],
            )
            run.status = RunStatus.FAILED
            run.stage = OptimizationStage.ERROR
            run.error = diag["message"]
            self.state.errors.append(f"{diag['type']}: {diag['message']}"[:200])
            self._log_event("Run failed", diag["message"][:200])
            _save_ckpt(self.state, reason="failed")
        finally:
            self._restore_pause_handler()
            if not run.completed_at:
                run.completed_at = datetime.now()
            if run.started_at and not run.duration_seconds:
                run.duration_seconds = (run.completed_at - run.started_at).total_seconds()
            # Do not overwrite explicit terminal states set above.
            run_repo.save(run)
            _save_ckpt(self.state, reason=f"final:{run.status.value}")
            # Ensure model unloaded and verified
            if not await self.client.ensure_unloaded(model.id):
                logger.warning("Model may still be loaded after run", model=model.id)

        return run

    # ---------- P1 operability helpers ----------

    def _report_progress(self) -> None:
        """Best-effort progress callback (informational only, never raises)."""
        try:
            cb = getattr(self, "_progress_cb", None)
            if cb is None or self.state is None:
                return
            cb(self.state.run.stage.value, len(self.state.tested_configs),
               _display_total(self.state))
        except Exception:
            pass

    def _log_event(self, message: str, detail: str = "") -> None:
        """Append a compact live-log event (no verbose internals)."""
        try:
            entry = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "message": message,
                "detail": detail[:200] if detail else "",
            }
            if self.state is not None:
                self.state.event_log.append(entry)
                if len(self.state.event_log) > 200:
                    self.state.event_log = self.state.event_log[-200:]
        except Exception:
            pass
        logger.info(message, detail=detail)

    async def _broadcast(self, extra: dict) -> None:
        """Best-effort WebSocket progress broadcast (never raises)."""
        try:
            if self.state is None:
                return
            from lm_optimizer.api.websocket import broadcast_progress

            from lm_optimizer.services.failure_states import is_terminal_failure

            st = self.state
            tested = len(st.tested_configs)
            total = _display_total(st)
            passed = sum(1 for c in st.tested_configs if c.status == ConfigurationStatus.PASSED)
            failed = sum(
                1
                for c in st.tested_configs
                if is_terminal_failure(c.status)
                and c.status not in (ConfigurationStatus.OOM, ConfigurationStatus.TIMEOUT)
            )
            oom = sum(1 for c in st.tested_configs if c.status == ConfigurationStatus.OOM)
            best = st.best_config
            elapsed = (datetime.now() - st.started_at).total_seconds() if st.started_at else 0.0
            # Context boundary progress from tested contexts.
            ctxs = sorted({c.context_length for c in st.tested_configs})
            cur_speed = best.get_avg_generation_tok_s() if best else 0.0
            payload = {
                "stage": st.run.stage.value,
                "model": st.run.model.id,
                "current_config": (st.best_config.config.to_dict() if st.best_config else None),
                "configs_tested": tested,
                "configs_total": total,
                "configs_passed": passed,
                "configs_failed": failed,
                "configs_oom": oom,
                "best_score": best.score if best else 0.0,
                "current_speed": round(cur_speed, 1),
                "vram_gb": best.peak_vram_gb if best else None,
                "ram_gb": best.peak_ram_gb if best else None,
                "context_progress": ctxs,
                "elapsed_s": round(elapsed, 1),
                # Adaptive search has no meaningful fixed ETA.
                "remaining": "adaptive / unknown",
                "events": st.event_log[-5:],
                **extra,
            }
            await broadcast_progress(st.run.id, payload)
        except Exception:
            pass
        self._report_progress()

    def _checkpoint(self, reason: str = "periodic") -> None:
        try:
            from lm_optimizer.storage.run_checkpoint import save_checkpoint as _save_ckpt

            if self.state is not None:
                self.state.checkpoint_seq = int(getattr(self.state, "checkpoint_seq", 0) or 0) + 1
                self._sync_phase_params()
                _save_ckpt(self.state, reason=reason)
        except Exception:
            pass

    # ---------- host guard: foreign model loads ----------

    def _host_guard(self) -> "ForeignModelWatchdog | None":
        """Watchdog protecting the model under test, or None when unavailable."""
        w = getattr(self, "_watchdog", None)
        if w is None:
            try:
                from lm_optimizer.services.foreign_watchdog import build_watchdog

                self._watchdog = w = build_watchdog(
                    self.client,
                    BACKEND_LMSTUDIO,
                    model_under_test=self.state.run.model.id,
                    echo_probe_model=self._echo_probe_model(),
                    allow_unload=bool(
                        (self.state.run.benchmark_params or {}).get(
                            "auto_unload_foreign", True
                        )
                    ),
                )
            except Exception as e:
                logger.debug("Foreign watchdog unavailable", error=str(e))
                return None
        return w

    def _echo_probe_model(self) -> str | None:
        """Smallest non-embedding model: the one ``connect()`` legitimately loads."""
        try:
            models = list(getattr(self.client, "_models_cache", []) or [])
            cands = [
                m for m in models if "embed" not in (getattr(m, "id", "") or "").lower()
            ]
            if not cands:
                return None
            return min(cands, key=lambda m: getattr(m, "size_bytes", 0) or 0).id
        except Exception:
            return None

    async def guard_host(self) -> dict:
        """Clear foreign loads before a measurement; never touches our own models.

        Returns a small record so the caller can mark a measurement as
        contested when a foreign model was present during the attempt.
        """
        out = {"foreign": [], "unloaded": [], "failed": [], "inspected": False}
        if self.state is None:
            return out
        wd = self._host_guard()
        if wd is None:
            return out
        try:
            report = await wd.check()
        except Exception as e:
            logger.warning("Host guard failed", error=f"{type(e).__name__}: {e}"[:160])
            return out

        out["inspected"] = report.inspected
        out["foreign"] = [f.model.model_id for f in report.foreign]
        out["unloaded"] = list(report.unloaded)
        out["failed"] = list(report.unload_failed)

        if report.foreign:
            self._log_event(
                "FOREIGN_MODEL_DETECTED",
                f"backend={report.backend} models={out['foreign']} "
                f"unloaded={out['unloaded']} failed={out['failed']}",
            )
            for f in report.foreign:
                self.state.errors.append(
                    f"foreign_model_loaded: {f.model.model_id}"
                    + (f" instance={f.model.instance_id}" if f.model.instance_id else "")
                )
            try:
                if self.state.event_log is not None:
                    self.state.event_log = self.state.event_log[-200:]
                if self.state.errors is not None:
                    self.state.errors = self.state.errors[-50:]
            except Exception:
                pass
        return out

    def checkpoint_now(self, reason: str = "manual") -> None:
        """Synchronous checkpoint for Cancel/shutdown paths (before ack)."""
        self._checkpoint(reason)
        try:
            if self.state is not None:
                run_repo.save(self.state.run)
        except Exception:
            pass

    async def _resolve_baseline(
        self, model: ModelIdentity, baseline_config: LoadConfiguration | None
    ) -> LoadConfiguration | None:
        """Provided baseline wins; otherwise capture the as-loaded config."""
        if baseline_config is not None:
            return baseline_config
        return await self._capture_baseline_config(model)

    async def _capture_baseline_config(self, model: ModelIdentity) -> LoadConfiguration | None:
        """Capture actual currently loaded configuration as baseline."""
        try:
            # Try to get currently loaded model info
            loaded = await self.client.get_model(model.id)
            if loaded and loaded.loaded and loaded.load_config:
                logger.info("Captured baseline from loaded model", model=model.id)
                return LoadConfiguration(
                    context_length=loaded.load_config.context_length,
                    gpu_ratio=loaded.load_config.gpu_ratio,
                    flash_attention=loaded.load_config.flash_attention,
                    offload_kv_cache_to_gpu=loaded.load_config.offload_kv_cache_to_gpu,
                    eval_batch_size=loaded.load_config.eval_batch_size,
                    num_experts=loaded.load_config.num_experts,
                )
            # Check client's loaded models cache
            if hasattr(self.client, "_loaded_models") and model.id in self.client._loaded_models:
                # We don't know exact config, use minimal baseline
                logger.info(
                    "Model is loaded but config unknown, using minimal baseline", model=model.id
                )
                return LoadConfiguration(context_length=model.context_limit or 4096)
        except Exception as e:
            logger.debug("Baseline capture failed", error=str(e))
        return None

    async def _run_baseline(self, baseline_config: LoadConfiguration) -> None:
        """Run baseline benchmark and save metrics."""
        logger.info("Running baseline benchmark", config=baseline_config.to_dict())
        self.state.run.stage = OptimizationStage.DISCOVERY
        run_repo.save(self.state.run)

        result = await self.benchmark.run_benchmark(
            self.state.run.model.id,
            baseline_config,
            self.state.run.model.context_limit or baseline_config.context_length or 4096,
        )
        # Convert status to enum
        if result.status == "passed":
            result.status = ConfigurationStatus.PASSED
        elif result.status == "failed":
            result.status = ConfigurationStatus.FAILED
            if (
                "OOM" in (result.error or "").upper()
                or "out of memory" in (result.error or "").lower()
            ):
                result.status = ConfigurationStatus.OOM

        # Evaluate quality for baseline even if we keep it regardless
        try:
            quality_scores = self.quality.evaluate_all(self.state.run.model.id, result.metrics)
            agg_quality = self.quality.aggregate_quality(quality_scores)
            result.quality_score = agg_quality
        except Exception:
            pass

        # Score baseline with current weights (will be recomputed later)
        result.run_id = self.state.run.id
        config_repo.save(result)

        self.state.run.baseline_config_id = result.id
        # Store baseline metrics for comparison
        self.state.run.baseline_metrics = {
            "generation_tok_s": result.get_avg_generation_tok_s(),
            "prompt_tok_s": result.get_avg_prompt_tok_s(),
            "estimated_ttft_ms": result.get_avg_estimated_ttft_ms(),
            "context_length": result.context_length,
            "peak_vram_gb": result.peak_vram_gb,
            "peak_ram_gb": result.peak_ram_gb,
            "quality_overall": result.quality_score.overall if result.quality_score else None,
            "quality_checks": result.quality_score.as_checks_str()
            if result.quality_score and hasattr(result.quality_score, "as_checks_str")
            else None,
            "stability": result.stability_score,
        }
        # Baseline is also a tested config
        self.state.tested_configs.append(result)
        run_repo.save(self.state.run)

    async def _stage_discovery(self) -> None:
        """Stage 1: Discovery - already done during search space generation."""
        logger.info("Stage 1: Discovery complete", search_space=self.state.search_space.to_dict())
        self.state.run.stage = OptimizationStage.DISCOVERY
        run_repo.save(self.state.run)
        self._log_event(
            "Discovery complete", f"space={self.state.search_space.estimate_size()} candidates"
        )
        self._checkpoint("discovery")
        await self._broadcast({"stage": "discovery", "event": "discovery"})
        self._report_progress()

    async def _stage_coarse_search(self) -> None:
        """Stage 2: Coarse search with deterministic intelligent sampling."""
        logger.info("Stage 2: Coarse search")
        self.state.run.stage = OptimizationStage.COARSE_SEARCH
        run_repo.save(self.state.run)

        space = self.state.search_space
        configs = self._generate_coarse_configs(space)

        # Resume: skip already-completed candidate keys (unless corrupted).
        completed_keys = {
            (
                c.context_length,
                c.config.gpu_ratio,
                c.config.flash_attention,
                c.config.offload_kv_cache_to_gpu,
                c.config.eval_batch_size,
                c.config.physical_batch_size,
                c.config.parallel,
                c.config.context_checkpoints,
            )
            for c in self.state.tested_configs
        }
        skipped = 0
        for i, cfg in enumerate(configs):
            if self.state.should_cancel:
                break
            while self.state.should_pause:
                await asyncio.sleep(1)

            key = (
                cfg.context_length,
                cfg.gpu_ratio,
                cfg.flash_attention,
                cfg.offload_kv_cache_to_gpu,
                cfg.eval_batch_size,
                cfg.physical_batch_size,
                cfg.parallel,
                cfg.context_checkpoints,
            )
            if key in completed_keys:
                skipped += 1
                continue
            logger.info("Testing coarse config", idx=i + 1, total=len(configs), **cfg.to_dict())
            self.state.current_config_id = None
            self._log_event("Next candidate", f"ctx={cfg.context_length} gpu={cfg.gpu_ratio}")
            result = await self._test_config(
                cfg, cfg.context_length or self.state.run.model.context_limit or 4096,
                min_speed=self.state.run.advanced_settings.get("min_speed", 0.0)
            )

            if result and result.status == ConfigurationStatus.PASSED:
                self.state.tested_configs.append(result)
                completed_keys.add(key)
                self._update_best(result)
            elif result is not None:
                # Any terminal failure (LOAD_FAILED/OOM/TIMEOUT/INCOMPATIBLE/
                # QUALITY_FAILED/...): record the region, never a candidate.
                from lm_optimizer.services.failure_states import is_soft

                if is_soft(result.status) or result.status == ConfigurationStatus.OOM:
                    self.state.failed_regions.add((cfg.context_length, cfg.gpu_ratio))
                self._log_event("CONFIG_REJECTED", str(result.status.value))

            # Recompute scores periodically for hardware-agnostic normalization
            if (i + 1) % 5 == 0:
                self._recompute_all_scores()

            run_repo.save(self.state.run)
            self._checkpoint("coarse")
            await self._broadcast({"stage": "coarse_search", "event": "candidate"})
        if skipped:
            self._log_event("Resume skip", f"skipped={skipped} already completed")

    def _deterministic_sample(self, items: list, k: int) -> list:
        """Deterministically sample k items covering low/high.

        Picks evenly spaced indices to ensure coverage of range.
        Deterministic (sorted order, no randomness).
        """
        if not items:
            return []
        if len(items) <= k:
            return sorted(items)
        sorted_items = sorted(items)
        # Evenly spaced indices including first and last
        indices = []
        for i in range(k):
            idx = int(round(i * (len(sorted_items) - 1) / (k - 1))) if k > 1 else 0
            indices.append(idx)
        # Dedup and preserve order
        seen = set()
        result = []
        for idx in indices:
            val = sorted_items[idx]
            if val not in seen:
                result.append(val)
                seen.add(val)
        return sorted(result)

    def _generate_coarse_configs(self, space: SearchSpace) -> list[LoadConfiguration]:
        """Generate coarse search configurations with deterministic intelligent sampling.

        Covers search dimensions reasonably: low/high context, low/high GPU,
        KV alternatives, Flash alternatives, batch alternatives.
        Uses deterministic sampling (sorted, evenly spaced) for reproducibility.
        """
        # Sample dimensions to ensure coverage, not just first N
        ctx_samples = self._deterministic_sample(
            space.context_lengths, min(4, len(space.context_lengths))
        )
        gpu_samples = self._deterministic_sample(space.gpu_ratios, min(3, len(space.gpu_ratios)))
        batch_samples = self._deterministic_sample(
            space.batch_sizes, min(2, len(space.batch_sizes))
        )
        flash_samples = sorted(space.flash_attention_options)
        if getattr(self.state, "flash_required", False) and True in flash_samples:
            flash_samples = [True]
        kv_samples = sorted(space.kv_cache_options)

        # Generate ALL combinations in deterministic sorted order
        all_configs = []
        for ctx in sorted(ctx_samples):
            for gpu in sorted(gpu_samples):
                for flash in flash_samples:
                    for kv in kv_samples:
                        for batch in sorted(batch_samples):
                            all_configs.append(
                                LoadConfiguration(
                                    context_length=ctx,
                                    gpu_ratio=gpu,
                                    flash_attention=flash,
                                    offload_kv_cache_to_gpu=kv,
                                    eval_batch_size=batch,
                                )
                            )

        # If still too many, deterministically sample 20 covering dimensions
        # Sort deterministically by tuple to ensure reproducibility
        all_configs.sort(
            key=lambda c: (
                c.context_length or 0,
                c.gpu_ratio or 0,
                c.eval_batch_size or 0,
                c.flash_attention or False,
                c.offload_kv_cache_to_gpu or False,
            )
        )

        if len(all_configs) > 20:
            # Evenly spaced deterministic sampling to cover dimensions
            step = len(all_configs) / 20
            sampled = []
            for i in range(20):
                idx = int(round(i * step))
                if idx >= len(all_configs):
                    idx = len(all_configs) - 1
                sampled.append(all_configs[idx])
            # Dedup preserve order
            seen = set()
            deduped = []
            for c in sampled:
                key = (
                    c.context_length,
                    c.gpu_ratio,
                    c.flash_attention,
                    c.offload_kv_cache_to_gpu,
                    c.eval_batch_size,
                )
                if key not in seen:
                    seen.add(key)
                    deduped.append(c)
            return deduped

        return all_configs[:20]

    async def _stage_refinement(self) -> None:
        """Stage 3: Refinement around promising candidates (not just best)."""
        logger.info("Stage 3: Refinement")
        self.state.run.stage = OptimizationStage.REFINEMENT
        run_repo.save(self.state.run)

        if not self.state.best_config:
            logger.warning("No best config found, skipping refinement")
            return

        # Select promising candidates: best + top Pareto frontier (up to 3)
        promising = [self.state.best_config]
        # Add Pareto frontier configs that are not already best, sorted by generation speed
        pareto = self._compute_pareto_frontier(self.state.tested_configs)
        pareto_sorted = sorted(pareto, key=lambda r: r.get_avg_generation_tok_s(), reverse=True)
        for pc in pareto_sorted:
            if pc.id != self.state.best_config.id and len(promising) < 3:
                promising.append(pc)

        space = self.state.search_space
        tested_keys = {
            (
                c.context_length,
                c.config.gpu_ratio,
                c.config.eval_batch_size,
                c.config.physical_batch_size,
                c.config.parallel,
                c.config.context_checkpoints,
            )
            for c in self.state.tested_configs
        }

        for best in promising:
            bc = best.config  # LoadConfiguration; best itself is a ConfigurationResult
            # Refine around each promising candidate, avoiding failed regions
            ctx_candidates = self._refine_context(best.context_length, space.context_lengths)
            gpu_candidates = self._refine_gpu_ratio(bc.gpu_ratio or 0.8, space.gpu_ratios)
            batch_candidates = self._refine_batch_size(bc.eval_batch_size or 256, space.batch_sizes)

            # Also test important parameter interactions: toggle flash and kv at best context/gpu
            interaction_configs = []
            prune_flash = getattr(self.state, "flash_required", False)
            for flash in space.flash_attention_options:
                if prune_flash and flash is False:
                    continue
                if flash != bc.flash_attention:
                    interaction_configs.append(
                        LoadConfiguration(
                            context_length=best.context_length,
                            gpu_ratio=bc.gpu_ratio,
                            flash_attention=flash,
                            offload_kv_cache_to_gpu=bc.offload_kv_cache_to_gpu,
                            eval_batch_size=bc.eval_batch_size,
                            physical_batch_size=bc.physical_batch_size,
                            parallel=bc.parallel,
                            context_checkpoints=bc.context_checkpoints,
                            num_experts=bc.num_experts,
                        )
                    )
            for kv in space.kv_cache_options:
                if kv != bc.offload_kv_cache_to_gpu:
                    interaction_configs.append(
                        LoadConfiguration(
                            context_length=best.context_length,
                            gpu_ratio=bc.gpu_ratio,
                            flash_attention=bc.flash_attention,
                            offload_kv_cache_to_gpu=kv,
                            eval_batch_size=bc.eval_batch_size,
                            physical_batch_size=bc.physical_batch_size,
                            parallel=bc.parallel,
                            context_checkpoints=bc.context_checkpoints,
                            num_experts=bc.num_experts,
                        )
                    )

            for cfg in interaction_configs:
                key = (
                    cfg.context_length,
                    cfg.gpu_ratio,
                    cfg.eval_batch_size,
                    cfg.physical_batch_size,
                    cfg.parallel,
                    cfg.context_checkpoints,
                )
                if key in tested_keys:
                    continue
                if (cfg.context_length, cfg.gpu_ratio) in self.state.failed_regions:
                    continue
                if self.state.should_cancel:
                    break
                while self.state.should_pause:
                    await asyncio.sleep(1)
                result = await self._test_config(cfg, cfg.context_length or 4096, min_speed=self.state.run.advanced_settings.get("min_speed", 0.0))
                tested_keys.add(key)
                if result and result.status == ConfigurationStatus.PASSED:
                    self.state.tested_configs.append(result)
                    self._update_best(result)
                elif result is not None:
                    self._log_event("CONFIG_REJECTED", str(result.status.value))
                run_repo.save(self.state.run)
                self._report_progress()

            for ctx in ctx_candidates:
                for gpu in gpu_candidates:
                    for batch in batch_candidates:
                        key = (
                            ctx,
                            gpu,
                            batch,
                            bc.physical_batch_size,
                            bc.parallel,
                            bc.context_checkpoints,
                        )
                        if key in tested_keys:
                            continue
                        # Avoid clearly inferior regions (failed before)
                        if (ctx, gpu) in self.state.failed_regions:
                            continue
                        if self.state.should_cancel:
                            break
                        while self.state.should_pause:
                            await asyncio.sleep(1)

                        config = LoadConfiguration(
                            context_length=ctx,
                            gpu_ratio=gpu,
                            flash_attention=bc.flash_attention,
                            offload_kv_cache_to_gpu=bc.offload_kv_cache_to_gpu,
                            eval_batch_size=batch,
                            physical_batch_size=bc.physical_batch_size,
                            parallel=bc.parallel,
                            context_checkpoints=bc.context_checkpoints,
                            num_experts=bc.num_experts,
                        )

                        result = await self._test_config(config, ctx, min_speed=self.state.run.advanced_settings.get("min_speed", 0.0))
                        tested_keys.add(key)
                        if result and result.status == ConfigurationStatus.PASSED:
                            self.state.tested_configs.append(result)
                            self._update_best(result)
                        elif result is not None:
                            from lm_optimizer.services.failure_states import is_soft

                            if is_soft(result.status):
                                self.state.failed_regions.add((ctx, gpu))
                            self._log_event("CONFIG_REJECTED", str(result.status.value))

                        run_repo.save(self.state.run)
                        self._report_progress()

            # Recompute scores after each promising candidate's refinement
            self._recompute_all_scores()

    def _refine_context(self, best_ctx: int, all_ctx: list[int]) -> list[int]:
        """Generate context candidates around best, using hardware-aware steps.

        Uses model-relative steps: max(512, best//4) but ensures we stay
        within search space bounds. Avoids hardcoding 32768.
        """
        if not all_ctx:
            return [best_ctx]
        candidates = [best_ctx]
        step = max(512, best_ctx // 4)

        for offset in [-step, step, -2 * step, 2 * step]:
            ctx = best_ctx + offset
            if ctx in all_ctx and ctx not in candidates:
                candidates.append(ctx)

        # Also include nearest neighbors in sorted search space for fine granularity
        sorted_ctx = sorted(all_ctx)
        try:
            idx = sorted_ctx.index(best_ctx)
            for neighbor_off in [-1, 1]:
                nidx = idx + neighbor_off
                if 0 <= nidx < len(sorted_ctx):
                    candidates.append(sorted_ctx[nidx])
        except ValueError:
            pass

        return sorted(set(candidates))

    def _refine_gpu_ratio(self, best_gpu: float, all_gpu: list[float]) -> list[float]:
        """Generate GPU ratio candidates around best, deterministically."""
        candidates = [best_gpu]
        # Use fine-grained offsets, filtered to search space
        for offset in [-0.1, -0.05, 0.05, 0.1, -0.15, 0.15]:
            gpu = round(best_gpu + offset, 2)
            if 0.0 <= gpu <= 1.0 and gpu in all_gpu and gpu not in candidates:
                candidates.append(gpu)
        return sorted(set(candidates))

    def _refine_batch_size(self, best_batch: int, all_batch: list[int]) -> list[int]:
        """Generate batch size candidates around best, deterministic."""
        candidates = [best_batch]
        for mult in [0.5, 0.75, 1.5, 2.0]:
            batch = int(best_batch * mult)
            batch = max(32, min(2048, batch))
            batch = round(batch / 32) * 32
            if batch in all_batch and batch not in candidates:
                candidates.append(batch)
        return sorted(set(candidates))

    def _generate_stage4_configs(
        self,
        space: SearchSpace,
        anchor: LoadConfiguration,
        tested: set,
    ) -> list[LoadConfiguration]:
        """Stage-4 candidates: vary one load dimension at a time around the anchor."""
        out = []
        seen = set()

        def add(batch, physical, parallel, checkpoints):
            key = (batch, physical, parallel, checkpoints)
            if key in tested or key in seen:
                return
            seen.add(key)
            out.append(
                LoadConfiguration(
                    context_length=anchor.context_length,
                    gpu_ratio=anchor.gpu_ratio,
                    flash_attention=anchor.flash_attention,
                    offload_kv_cache_to_gpu=anchor.offload_kv_cache_to_gpu,
                    eval_batch_size=batch,
                    physical_batch_size=physical,
                    parallel=parallel,
                    context_checkpoints=checkpoints,
                    num_experts=anchor.num_experts,
                )
            )

        for batch in space.batch_sizes:
            add(batch, anchor.physical_batch_size, anchor.parallel, anchor.context_checkpoints)
        for physical in space.physical_batch_sizes:
            add(anchor.eval_batch_size, physical, anchor.parallel, anchor.context_checkpoints)
        for parallel in space.parallels:
            add(
                anchor.eval_batch_size,
                anchor.physical_batch_size,
                parallel,
                anchor.context_checkpoints,
            )
        for checkpoints in space.context_checkpoints_options:
            add(anchor.eval_batch_size, anchor.physical_batch_size, anchor.parallel, checkpoints)
        return out

    async def _stage_batch_optimization(self) -> None:
        """Stage 4: Optimize eval batch size and load concurrency parameters."""
        logger.info("Stage 4: Batch optimization")
        self.state.run.stage = OptimizationStage.BATCH_OPTIMIZATION
        run_repo.save(self.state.run)

        if not self.state.best_config:
            return

        best = self.state.best_config
        tested = {
            (
                c.config.eval_batch_size,
                c.config.physical_batch_size,
                c.config.parallel,
                c.config.context_checkpoints,
            )
            for c in self.state.tested_configs
            if c.context_length == best.context_length
            and c.config.gpu_ratio == best.config.gpu_ratio
        }
        for config in self._generate_stage4_configs(self.state.search_space, best.config, tested):
            if self.state.should_cancel:
                break
            while self.state.should_pause:
                await asyncio.sleep(1)

            result = await self._test_config(config, best.context_length, min_speed=self.state.run.advanced_settings.get("min_speed", 0.0))
            if result and result.status == ConfigurationStatus.PASSED:
                self.state.tested_configs.append(result)
                self._update_best(result)
                await self._maybe_measure_throughput(result)
            elif result is not None:
                self._log_event("CONFIG_REJECTED", str(result.status.value))

            run_repo.save(self.state.run)
            self._report_progress()

    async def _maybe_measure_throughput(self, result: ConfigurationResult) -> None:
        """Record a REAL concurrent aggregate probe for THROUGHPUT runs.

        Only for passed configs with parallel > 1 (bounded: stage-4/micro
        candidates). The measured block lands in result.generation so it is
        persisted and the workload tie-break prefers measured evidence.
        """
        from lm_optimizer.services.workload import normalize_workload

        workload = normalize_workload((self.state.run.benchmark_params or {}).get("workload_type"))
        if workload != "throughput":
            return
        try:
            parallel = int(result.config.parallel or 1)
        except (TypeError, ValueError):
            return
        if parallel <= 1 or self.state.should_cancel:
            return
        measured = await self.benchmark.measure_throughput(
            self.state.run.model.id, result.config, parallel
        )
        result.generation = {**(result.generation or {}), "throughput_measured": measured}
        config_repo.save(result)
        try:
            agg = measured.get("aggregate_tok_s", 0.0)
            self._log_event(f"Measured aggregate {agg} tok/s", f"parallel={parallel}")
        except Exception:
            pass

    async def _stage_validation(self) -> None:
        """Stage 5: Final validation with multiple runs."""
        logger.info("Stage 5: Final validation")
        self.state.run.stage = OptimizationStage.VALIDATION
        run_repo.save(self.state.run)

        if not self.state.best_config:
            self.state.run.stage = OptimizationStage.COMPLETE
            return

        best = self.state.best_config
        bc = best.config
        config = LoadConfiguration(
            context_length=best.context_length,
            gpu_ratio=bc.gpu_ratio,
            flash_attention=bc.flash_attention,
            offload_kv_cache_to_gpu=bc.offload_kv_cache_to_gpu,
            eval_batch_size=bc.eval_batch_size,
            physical_batch_size=bc.physical_batch_size,
            parallel=bc.parallel,
            context_checkpoints=bc.context_checkpoints,
            num_experts=bc.num_experts,
        )
        # Preserve RoPE if experimental and enabled
        if self.state.run.is_experimental:
            config.rope_freq_base = bc.rope_freq_base
            config.rope_freq_scale = bc.rope_freq_scale

        # Run validation repetitions
        validation_results = []
        for i in range(self.state.run.validation_repetitions):
            if self.state.should_cancel:
                break
            result = await self._test_config(config, best.context_length, min_speed=self.state.run.advanced_settings.get("min_speed", 0.0))
            if result and result.status == ConfigurationStatus.PASSED:
                validation_results.append(result)

        if validation_results:
            # Use median result
            self.state.best_config = self._median_result(validation_results)
            # Ensure best is in tested list
            if self.state.best_config.id not in [c.id for c in self.state.tested_configs]:
                self.state.tested_configs.append(self.state.best_config)
            self.state.run.best_config_id = self.state.best_config.id

        # Calculate Pareto frontier
        self.state.pareto_frontier = self._compute_pareto_frontier(self.state.tested_configs)
        self.state.run.pareto_config_ids = [c.id for c in self.state.pareto_frontier]

        self.state.run.stage = OptimizationStage.COMPLETE
        run_repo.save(self.state.run)
        self._report_progress()

    def _phase_weights(self) -> dict:
        """Scoring weights for the current flow.

        Phase A freezes context, so the context component must not enter the
        winner score (mission §22). Legacy flows (score_context=True) are
        untouched.
        """
        weights = dict(self.state.run.profile_weights or {})
        if not self.state.score_context:
            weights["context"] = 0.0
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
        return weights

    FLASH_REQUIRED_SETTING = "flash_required"

    def _note_flash_requirement(self, result: ConfigurationResult) -> None:
        """Remember server-mandated flash attention (prunes flash=False).

        Persists across runs (KV quant is a host-wide GUI setting): the next
        run preskips flash-off probes instead of re-paying one failed load.
        """
        if "requires flash attention" in (result.error or "").lower():
            if not self.state.flash_required:
                logger.warning("Server requires flash attention: pruning flash=False")
                self.state.errors.append("flash_required: flash=False pruned for this run")
            self.state.flash_required = True
            try:
                settings_repo.set(
                    self.FLASH_REQUIRED_SETTING, "1",
                    "KV quant requires flash attention (learned, host-wide)",
                )
            except Exception as e:
                logger.debug("Flash lesson persist failed", error=str(e))

    def _apply_persisted_flash_advice(self) -> None:
        """Preskip flash-off probes from a previous run's lesson."""
        try:
            if settings_repo.get(self.FLASH_REQUIRED_SETTING) != "1":
                return
        except Exception:
            return
        if not self.state.flash_required:
            logger.info("Applying persisted flash requirement (preskipping flash=False)")
            self.state.errors.append(
                "flash_required: learned from previous runs (KV quant needs flash)"
            )
        self.state.flash_required = True

    def _clear_flash_requirement(self, load_config: LoadConfiguration) -> None:
        """A flash-off load that served disproves the lesson: clear it."""
        if load_config.flash_attention is not False:
            return
        if self.state.flash_required:
            logger.info("Flash-off served: clearing flash requirement")
        self.state.flash_required = False
        try:
            settings_repo.set(self.FLASH_REQUIRED_SETTING, "0", "flash-off served")
        except Exception:
            pass

    def _check_rope_quality(self, breakdown: dict) -> None:
        """Experimental RoPE guard: never claim faster-is-better below 0.98."""
        if self.state.run.is_experimental and breakdown.get("quality", 1.0) < 0.98:
            logger.warning(
                "Experimental RoPE config quality insufficient",
                quality=breakdown.get("quality"),
            )
            # Do not treat as best even if faster

    def _score_eligible(self, result: ConfigurationResult, score_ctx: bool) -> None:
        """Score a passed result (bounds over tested + this result)."""
        weights = self._phase_weights() if not score_ctx else self.state.run.profile_weights
        temp_results = self.state.tested_configs + [result]
        bounds = compute_bounds(
            temp_results, self.state.run.hardware, self.state.run.model.context_limit
        )
        breakdown = score_result_breakdown(
            result,
            bounds,
            self.state.run.hardware,
            self.state.run.model.context_limit,
            weights,
        )
        result.score_breakdown = breakdown
        result.score = weighted_score(breakdown, weights)
        result.generation = {
            **(result.generation or {}),
            "score_context": score_ctx,
            "score_weights": dict(weights),
        }

    async def _test_config(
        self,
        load_config: LoadConfiguration,
        context_length: int,
        score_context: bool | None = None,
        min_speed: float = 0.0,
    ) -> ConfigurationResult | None:
        """Test a single configuration through the strict phase machine.

        CREATED -> LOAD_REQUESTED -> LOADED -> VERIFIED -> PREHEATED ->
        BENCHMARKED -> QUALITY_EVALUATED -> SCORED -> CANDIDATE_ELIGIBLE.
        Any phase failure returns the result with a terminal failure status
        and score None (never scored, never a winner). Only CANDIDATE_ELIGIBLE
        (PASSED + score set) may enter winner selection.

        `score_context=False` zeroes the context score component (Phase A:
        context is frozen, §22). Defaults to the state's flow flag.

        Duration convention: every returned (persisted) result from a started
        attempt carries `duration_ms > 0` (wall time around the attempt).
        Flash-skipped attempts return None (never started, nothing persisted).
        Unexpected exceptions return a stamped failed result (score None,
        never a winner) instead of None so the attempt is recorded.
        """
        from lm_optimizer.services.failure_states import classify_error, is_soft

        if (
            self.state is not None
            and getattr(self.state, "flash_required", False)
            and load_config.flash_attention is False
        ):
            logger.info("Skipping flash-off config (server requires flash)")
            return None
        if self._unverified_unload_block():
            return None
        t0 = time.perf_counter()

        def _stamp(res: ConfigurationResult) -> ConfigurationResult:
            try:
                res.duration_ms = (time.perf_counter() - t0) * 1000.0
            except Exception:
                pass
            return res

        try:
            # Clear any foreign model load BEFORE measuring: a stray local
            # client competes for VRAM and can abort this generation, which
            # turns a real measurement into noise.
            guard = await self.guard_host()
            self._log_event("LOAD_REQUESTED", f"ctx={context_length}")
            result = await self.benchmark.run_benchmark(
                self.state.run.model.id, load_config, context_length
            )
            result.run_id = self.state.run.id
            try:
                self.state.current_config_id = result.id
            except Exception:
                pass

            # Phase gate: only a passed benchmark may continue. Failed loads
            # MUST NOT receive metrics evaluation, quality, score or Pareto.
            if result.status != "passed":
                result.status = classify_error(result.error)
                result.score = None
                result.quality_score = None
                result.score_breakdown = None
                result.generation = {
                    **(result.generation or {}),
                    "stage": self.state.run.stage.value,
                }
                try:
                    config_repo.save(result)
                except Exception as e:
                    # Failure recording must never take down the loop.
                    logger.error("Failure record save failed", error=str(e))
                if result.status == ConfigurationStatus.OOM:
                    self._log_event("OOM", f"ctx={context_length}")
                    self._log_event("Failed region recorded", f"ctx={context_length}")
                else:
                    self._log_event(str(result.status.value).upper(), (result.error or "")[:120])
                logger.info(
                    "Load/benchmark failed",
                    config_id=str(result.id),
                    status=result.status.value,
                    error=(result.error or "")[:160],
                )
                # Learn server-mandated flash here (failures return early and
                # never reach the passed-path note below).
                self._note_flash_requirement(result)
                return _stamp(result)

            # LOADED + VERIFIED (benchmark success implies the load served).
            result.status = ConfigurationStatus.PASSED
            # A served flash-off load disproves any persisted lesson.
            self._clear_flash_requirement(load_config)
            self._log_event("LOAD_SUCCEEDED", f"ctx={context_length}")
            self._log_event("VERIFY_SUCCEEDED", f"ctx={context_length}")
            preheat = (result.generation or {}).get("preheat") or {}
            if preheat:
                self._log_event(
                    "PREHEAT_STARTED",
                    f"warmup_ms={preheat.get('warmup_time_ms', 0)}",
                )
            self._log_event("BENCHMARK_STARTED", f"ctx={context_length}")
            self._log_event("BENCHMARK_COMPLETED", f"ctx={context_length}")

            # Check minimum speed threshold (if set)
            if min_speed > 0.0:
                gen_tok_s = result.get_avg_generation_tok_s()
                if gen_tok_s < min_speed:
                    result.status = ConfigurationStatus.FAILED
                    result.error = f"Generation speed {gen_tok_s:.1f} tok/s below minimum threshold {min_speed:.1f} tok/s"
                    result.score = None
                    result.quality_score = None
                    result.score_breakdown = None
                    result.generation = {
                        **(result.generation or {}),
                        "stage": self.state.run.stage.value,
                    }
                    try:
                        config_repo.save(result)
                    except Exception as e:
                        logger.error("Failure record save failed", error=str(e))
                    self._log_event(
                        "SPEED_BELOW_THRESHOLD",
                        f"ctx={context_length} gen={gen_tok_s:.1f} tok/s < min={min_speed:.1f} tok/s",
                    )
                    logger.warning(
                        "SPEED_BELOW_THRESHOLD",
                        config_id=str(result.id),
                        speed=gen_tok_s,
                        min_speed=min_speed,
                    )
                    return _stamp(result)

            # Server demands flash attention (non-F16 KV quant): prune flash=False (R28)
            self._note_flash_requirement(result)

            # Evaluate quality / correctness (metrics preserved on failure).
            quality_scores = self.quality.evaluate_all(self.state.run.model.id, result.metrics)
            agg_quality = self.quality.aggregate_quality(quality_scores)
            result.quality_score = agg_quality
            # Provenance: stage + per-test quality travel with the record so
            # reports can show every test explicitly (nothing fabricated).
            # quality_completed marks "suite ran to a quality result" for BOTH
            # outcomes, so resume treats QUALITY_FAILED as completed work and
            # never re-runs a finished suite merely because it failed.
            result.generation = {
                **(result.generation or {}),
                "stage": self.state.run.stage.value,
                "quality_completed": True,
                "quality_by_test": {
                    name: {
                        "overall": round(q.overall, 3),
                        "checks_passed": q.checks_passed,
                        "checks_total": q.checks_total,
                    }
                    for name, q in (quality_scores or {}).items()
                },
            }

            if not self.quality.passes_threshold(agg_quality):
                result.status = ConfigurationStatus.QUALITY_FAILED
                result.error = f"Quality/correctness {agg_quality.overall:.3f} ({agg_quality.checks_passed}/{agg_quality.checks_total} checks) below threshold {self.quality.config.minimum_score}"
                result.score = None
                result.score_breakdown = None
                try:
                    config_repo.save(result)
                except Exception as e:
                    logger.error("Failure record save failed", error=str(e))
                self._log_event(
                    "QUALITY_FAILED",
                    f"{agg_quality.checks_passed}/{agg_quality.checks_total} ctx={context_length}",
                )
                logger.warning(
                    "QUALITY_FAILED",
                    config_id=str(result.id),
                    quality=agg_quality.overall,
                    checks=f"{agg_quality.checks_passed}/{agg_quality.checks_total}",
                    threshold=self.quality.config.minimum_score,
                )
                return _stamp(result)
            self._log_event(
                "QUALITY_EVALUATED",
                f"{agg_quality.checks_passed}/{agg_quality.checks_total}",
            )

            # Hardware-agnostic scoring with breakdown
            score_ctx = self.state.score_context if score_context is None else score_context
            self._score_eligible(result, score_ctx)
            breakdown = result.score_breakdown or {}
            self._check_rope_quality(breakdown)

            config_repo.save(result)

            logger.info(
                "CONFIG_ELIGIBLE",
                config_id=str(result.id),
                gen_tok_s=result.get_avg_generation_tok_s(),
                quality=f"{agg_quality.checks_passed}/{agg_quality.checks_total}",
                score=result.score,
                breakdown=breakdown,
            )
            self._log_event(
                f"{result.get_avg_generation_tok_s():.1f} tok/s",
                f"ctx={context_length}",
            )
            self._log_event("CONFIG_ELIGIBLE", f"score={result.score:.3f}")

            return _stamp(result)

        except Exception as e:
            logger.error("Config error", error=str(e))
            self.state.errors.append(f"{load_config}: {e!s}")
            self._log_event("Candidate failed", str(e)[:120])
            try:
                await self.client.ensure_unloaded(self.state.run.model.id)
            except Exception:
                pass
            failed = ConfigurationResult(
                config=load_config,
                context_length=context_length,
                status=classify_error(str(e)),
                error=f"{type(e).__name__}: {e}"[:500],
                tested_at=datetime.now(),
                generation={"stage": self.state.run.stage.value},
            )
            failed.score = None
            try:
                failed.run_id = self.state.run.id
            except Exception:
                pass
            _stamp(failed)
            try:
                config_repo.save(failed)
            except Exception as se:
                logger.error("Failure record save failed", error=str(se))
            return failed

    def _update_best(self, result: ConfigurationResult) -> None:
        """Update best configuration using hardware-agnostic scores.

        Only CANDIDATE_ELIGIBLE results (PASSED + score set) may win.
        """
        if result.status != ConfigurationStatus.PASSED or result.score is None:
            self._log_event("CONFIG_REJECTED", "ineligible for winner selection")
            return
        if not self.state.best_config:
            self.state.best_config = result
            self.state.run.best_config_id = result.id
            return

        # Recompute both scores with current bounds for fair comparison
        bounds = compute_bounds(
            self.state.tested_configs, self.state.run.hardware, self.state.run.model.context_limit
        )
        # Update existing best's breakdown
        weights = self._phase_weights()
        best_breakdown = score_result_breakdown(
            self.state.best_config,
            bounds,
            self.state.run.hardware,
            self.state.run.model.context_limit,
            weights,
        )
        self.state.best_config.score_breakdown = best_breakdown
        self.state.best_config.score = weighted_score(best_breakdown, weights)

        # Result already scored, but recompute with latest bounds
        new_breakdown = score_result_breakdown(
            result,
            bounds,
            self.state.run.hardware,
            self.state.run.model.context_limit,
            weights,
        )
        result.score_breakdown = new_breakdown
        result.score = weighted_score(new_breakdown, weights)

        current_score = self.state.best_config.score
        new_score = result.score

        # Authoritative 5% non-inferiority selection (services/selection.py).
        from lm_optimizer.services.selection import SelectionConfig, select_by_score

        try:
            threshold = float(
                (self.state.run.benchmark_params or {}).get("selection_threshold", 0.05)
            )
        except (TypeError, ValueError):
            threshold = 0.05
        wins, explanation = select_by_score(
            self.state.best_config,
            result,
            profile=self.state.run.profile.value,
            config=SelectionConfig(preference_threshold=threshold),
        )
        if not wins and explanation.get("band"):
            # Workload actually changes behavior: within the band, a workload
            # tie-break can still promote the challenger (TTFT-first for
            # interactive, aggregate-throughput-first for throughput).
            from lm_optimizer.services.workload import workload_tiebreak_key

            workload = (self.state.run.benchmark_params or {}).get("workload_type", "interactive")
            if workload_tiebreak_key(workload, result) > workload_tiebreak_key(
                workload, self.state.best_config
            ):
                wins = True
                explanation = {
                    **explanation,
                    "reason": explanation.get("reason", "")
                    + f"; workload={workload} tie-break prefers challenger",
                }
        if wins:
            logger.info(
                "New best config",
                old_score=current_score,
                new_score=new_score,
                breakdown=new_breakdown,
                selection=explanation.get("reason", ""),
            )
            if explanation.get("band"):
                self._log_event("Selection within 5% band", explanation.get("reason", "")[:200])
            self.state.best_config = result
            self.state.run.best_config_id = result.id

    def _recompute_all_scores(self) -> None:
        """Recompute scores for all tested configs with final bounds (for explainability)."""
        if not self.state.tested_configs:
            return
        bounds = compute_bounds(
            self.state.tested_configs, self.state.run.hardware, self.state.run.model.context_limit
        )
        for r in self.state.tested_configs:
            if r.status == ConfigurationStatus.PASSED:
                # SPEED probes rank by measured tok_s (frontier), never by
                # score: they stay scoreless so no synthetic quality can ever
                # promote an unvalidated probe into winner selection.
                if (r.generation or {}).get("phase") == OptimizationPhase.SPEED.value:
                    r.score = None
                    r.score_breakdown = None
                    try:
                        config_repo.save(r)
                    except Exception:
                        pass
                    continue
                weights = self._phase_weights()
                breakdown = score_result_breakdown(
                    r,
                    bounds,
                    self.state.run.hardware,
                    self.state.run.model.context_limit,
                    weights,
                )
                r.score_breakdown = breakdown
                r.score = weighted_score(breakdown, weights)
                config_repo.save(r)
        if self.state.best_config:
            # Re-evaluate best
            best = next(
                (c for c in self.state.tested_configs if c.id == self.state.best_config.id),
                self.state.best_config,
            )
            self.state.best_config = best

    def _compute_pareto_frontier(
        self, results: list[ConfigurationResult]
    ) -> list[ConfigurationResult]:
        """Compute Pareto frontier - hardware-agnostic multi-objective."""
        if not results:
            return []

        frontier = []
        for r in results:
            if r.status != ConfigurationStatus.PASSED or r.score is None:
                continue
            if r.quality_score is None:
                continue

            dominated = False
            for other in results:
                if (
                    other.status != ConfigurationStatus.PASSED
                    or other.score is None
                    or other.quality_score is None
                    or other is r
                ):
                    continue

                other_better = (
                    other.quality_score.overall >= r.quality_score.overall
                    and other.get_avg_generation_tok_s() >= r.get_avg_generation_tok_s()
                    and other.get_avg_prompt_tok_s() >= r.get_avg_prompt_tok_s()
                    and other.context_length >= r.context_length
                    and (other.peak_vram_gb or float("inf")) <= (r.peak_vram_gb or float("inf"))
                )
                other_strictly_better = (
                    other.quality_score.overall > r.quality_score.overall
                    or other.get_avg_generation_tok_s() > r.get_avg_generation_tok_s()
                    or other.get_avg_prompt_tok_s() > r.get_avg_prompt_tok_s()
                    or other.context_length > r.context_length
                    or (other.peak_vram_gb or float("inf")) < (r.peak_vram_gb or float("inf"))
                )

                if other_better and other_strictly_better:
                    dominated = True
                    break

            if not dominated:
                frontier.append(r)

        return frontier

    def _median_result(self, results: list[ConfigurationResult]) -> ConfigurationResult:
        """Get median result by generation speed."""
        speeds = [(r.get_avg_generation_tok_s(), r) for r in results]
        speeds.sort(key=lambda x: x[0])
        return speeds[len(speeds) // 2][1]

    async def _stage_micro_refinement(self, max_candidates: int = 6) -> None:
        """Profile-aware runtime micro-stage: bounded neighborhood of the winner.

        Temperature is NEVER part of runtime optimization; this stage only tests
        interactions between the already-selected runtime parameters
        (gpu/kv/flash/batch neighborhood). Hard-capped to `max_candidates`.
        """
        from lm_optimizer.domain.models import OptimizationProfile as _Prof

        if not self.state.best_config:
            return
        self.state.run.stage = OptimizationStage.MICRO_REFINEMENT
        run_repo.save(self.state.run)
        best = self.state.best_config
        bc = best.config
        profile = self.state.run.profile
        self._log_event("Micro-stage: runtime neighborhood", f"profile={profile.value}")

        # Profile focus decides which single-dimension neighbors to try.
        candidates: list[LoadConfiguration] = []

        def _cfg(**kw: Any) -> LoadConfiguration:
            base = {
                "context_length": best.context_length,
                "gpu_ratio": bc.gpu_ratio,
                "flash_attention": bc.flash_attention,
                "offload_kv_cache_to_gpu": bc.offload_kv_cache_to_gpu,
                "eval_batch_size": bc.eval_batch_size,
                "physical_batch_size": bc.physical_batch_size,
                "parallel": bc.parallel,
                "context_checkpoints": bc.context_checkpoints,
                "num_experts": bc.num_experts,
            }
            base.update(kw)
            return LoadConfiguration(**base)

        space = self.state.search_space
        # Candidate pool per profile (throughput vs memory/context vs stability).
        if profile == _Prof.SPEED:
            # Throughput: batch + physical/parallel neighbors.
            for b in (space.batch_sizes or [])[:4]:
                if b != bc.eval_batch_size:
                    candidates.append(_cfg(eval_batch_size=b))
            for p in (space.physical_batch_sizes or [])[:2]:
                if p != bc.physical_batch_size:
                    candidates.append(_cfg(physical_batch_size=p))
        elif profile == _Prof.CONTEXT:
            # Context stability: keep ctx, vary KV/checkpoints neighbors only.
            for kv in space.kv_cache_options:
                if kv != bc.offload_kv_cache_to_gpu:
                    candidates.append(_cfg(offload_kv_cache_to_gpu=kv))
            for ck in (space.context_checkpoints_options or [])[:2]:
                if ck != bc.context_checkpoints:
                    candidates.append(_cfg(context_checkpoints=ck))
        elif profile == _Prof.QUALITY:
            # Correctness/stability: minimal moves (batch down, checkpoints up).
            smaller = [b for b in (space.batch_sizes or []) if b < (bc.eval_batch_size or 10**9)]
            if smaller:
                candidates.append(_cfg(eval_batch_size=max(smaller)))
            larger_ck = [
                c
                for c in (space.context_checkpoints_options or [])
                if c > (bc.context_checkpoints or 0)
            ]
            if larger_ck:
                candidates.append(_cfg(context_checkpoints=min(larger_ck)))
        else:  # BALANCED / CUSTOM
            for b in (space.batch_sizes or [])[:3]:
                if b != bc.eval_batch_size:
                    candidates.append(_cfg(eval_batch_size=b))
            for kv in space.kv_cache_options:
                if kv != bc.offload_kv_cache_to_gpu:
                    candidates.append(_cfg(offload_kv_cache_to_gpu=kv))
            for ck in (space.context_checkpoints_options or [])[:1]:
                if ck != bc.context_checkpoints:
                    candidates.append(_cfg(context_checkpoints=ck))

        # Dedup + hard cap (never a brute-force search).
        seen: set = set()
        capped: list[LoadConfiguration] = []
        tested_keys = {
            (
                c.context_length,
                c.config.gpu_ratio,
                c.config.flash_attention,
                c.config.offload_kv_cache_to_gpu,
                c.config.eval_batch_size,
                c.config.physical_batch_size,
                c.config.parallel,
                c.config.context_checkpoints,
            )
            for c in self.state.tested_configs
        }
        for cfg in candidates:
            key = (
                cfg.context_length,
                cfg.gpu_ratio,
                cfg.flash_attention,
                cfg.offload_kv_cache_to_gpu,
                cfg.eval_batch_size,
                cfg.physical_batch_size,
                cfg.parallel,
                cfg.context_checkpoints,
            )
            if key in seen or key in tested_keys:
                continue
            if (cfg.context_length, cfg.gpu_ratio) in self.state.failed_regions:
                continue
            seen.add(key)
            capped.append(cfg)
            if len(capped) >= max_candidates:
                break

        for cfg in capped:
            if self.state.should_cancel:
                break
            while self.state.should_pause:
                await asyncio.sleep(1)
            result = await self._test_config(cfg, cfg.context_length or best.context_length)
            if result and result.status == ConfigurationStatus.PASSED:
                self.state.tested_configs.append(result)
                self._update_best(result)
                await self._maybe_measure_throughput(result)
            elif result is not None:
                self._log_event("CONFIG_REJECTED", str(result.status.value))
            run_repo.save(self.state.run)
            self._checkpoint("micro")
            await self._broadcast({"stage": "micro_refinement", "event": "micro_candidate"})
        self._log_event("Micro-stage complete", f"tested={len(capped)}/{max_candidates} max")

    # ---------- Phase A/B strategy (mission §3-§22) ----------

    def _sync_phase_params(self) -> None:
        """Mirror phase extras into benchmark_params (no DB schema migration)."""
        try:
            params = self.state.run.benchmark_params or {}
            prev_pause = (params.get("phase_ab") or {}).get("pause")
            params["phase_ab"] = {
                "phase": self.state.phase,
                "speed_context": self.state.speed_context,
                "quality_context": self.state.quality_context,
                "budget_class": self.state.budget_class,
                "speed_finalist_ids": list(self.state.speed_finalists),
                "raw_fastest_id": str(self.state.raw_fastest.id)
                if self.state.raw_fastest is not None
                else None,
                "recovery_log": list(self.state.recovery_log),
                "recovery_cursor": dict(self.state.recovery_cursor or {}),
                "probe_attempts": dict(self.state.probe_attempts or {}),
                "checkpoint_seq": int(getattr(self.state, "checkpoint_seq", 0) or 0),
                "pause": prev_pause,
                "excluded_notes": list(self.state.excluded_notes),
                "phase_b_enabled": self.state.phase_b_enabled,
                "phase_b_result": self.state.phase_b_result,
            }
            self.state.run.benchmark_params = params
        except Exception:
            pass

    def _speed_caps(self):
        """Capability gates for the speed plan (registry-backed, R1)."""
        from lm_optimizer.services.speed_search import SpeedCaps

        caps = getattr(self.client, "capabilities", None)
        rest: set[str] = set()
        if caps is not None:
            if getattr(caps, "supports_flash_attention", False):
                rest.add("flash_attention")
            if getattr(caps, "supports_kv_cache_placement", False):
                rest.add("offload_kv_cache_to_gpu")
            if getattr(caps, "supports_eval_batch_size", False):
                rest.add("eval_batch_size")
            if getattr(caps, "supports_physical_batch_size", False):
                rest.add("physical_batch_size")
            if getattr(caps, "supports_parallel", False):
                rest.add("parallel")
            if getattr(caps, "supports_context_checkpoints", False):
                rest.add("context_checkpoints")
            if getattr(caps, "supports_num_experts", False):
                rest.add("num_experts")
        rest.add("context_length")
        advanced = {}
        try:
            advanced = (self.state.run.benchmark_params or {})
        except Exception:
            pass
        return SpeedCaps(
            rest_keys=frozenset(rest),
            gpu_via_cli=bool(getattr(self.benchmark, "gpu_via_cli", False)),
            is_moe=bool(getattr(self.state.run.model, "is_moe", False)),
            has_draft_model=bool(advanced.get("speculative_draft_model")),
            flash_required=bool(self.state.flash_required),
        )

    async def _granular_baseline_or_none(
        self, model: ModelIdentity, baseline_config, skips: set[str]
    ):
        """Anchor config for the run; granular fallback keeps quality modes alive.

        A cold server (nothing loaded) yields no capturable baseline; plain
        campaigns survive because the speed plan seeds from defaults. Granular
        quality modes (quality-only / reasoning-only) have no speed phase, so
        with no anchor they would end instantly with an empty frontier
        (n=0, Run FAILED). Fail-safe: use the S0 default anchor as baseline.
        """
        if baseline_config is not None:
            return baseline_config
        captured = await self._capture_baseline_config(model)
        if captured is not None:
            return captured
        if "speed" in skips:
            self._log_event(
                "Baseline fallback",
                "model not loaded; S0 default anchor used as quality baseline",
            )
            return self._default_anchor()
        return None

    def _default_anchor(self) -> LoadConfiguration:
        """S0 anchor: captured as-loaded config, else server defaults."""
        ctx = self.state.speed_context or 2048
        base = None
        if self.state.baseline_result is not None:
            base = self.state.baseline_result.config
        if base is None:
            return LoadConfiguration(
                context_length=ctx,
                flash_attention=True,
                offload_kv_cache_to_gpu=True,
                eval_batch_size=2048,
                physical_batch_size=512,
                parallel=4,
                context_checkpoints=32,
            )
        data = {k: v for k, v in base.to_dict().items()}
        data["context_length"] = ctx
        cfg = LoadConfiguration(**{k: v for k, v in data.items() if hasattr(LoadConfiguration, k)})
        cfg.context_length = ctx
        return cfg

    @staticmethod
    def _candidate_key(cfg: LoadConfiguration) -> tuple:
        return tuple(sorted((k, v) for k, v in cfg.to_dict().items()))

    def _unverified_unload_block(self) -> bool:
        """Cancel BEFORE the next load when the previous unload is unverified.

        Fail-closed: a leftover model that could not be verified unloaded is
        still holding memory (or the endpoint state is unverifiable). Loading
        another memory-heavy candidate would repeat the resource-guardrail
        cascade (observed as load-failed storm in the Fable 27B campaign).
        ``cancel()`` checkpoints everything measured so far; results, errors
        and the checkpoint file are preserved for resume/inspection.
        """
        flag = getattr(self.benchmark, "last_unverified_unload", None)
        if not (isinstance(flag, str) and flag):
            return False
        model = flag
        try:
            self.benchmark.last_unverified_unload = None
        except Exception:
            pass
        self._log_event(
            "UNLOAD_UNVERIFIED",
            f"model={model} unload could not be verified; stopping before "
            "next memory-heavy load (results checkpointed)",
        )
        try:
            self.state.errors.append(
                f"unload_not_verified: {model} still occupying memory; "
                "refusing further memory-heavy loads"
            )
        except Exception:
            pass
        self.cancel()
        return True

    async def _record_probe_result(
        self, result: ConfigurationResult, context_length: int
    ) -> ConfigurationResult:
        """Gate + persist one speed probe (failures: classified, scoreless)."""
        from lm_optimizer.services.failure_states import classify_error

        result.run_id = self.state.run.id
        try:
            self.state.current_config_id = result.id
        except Exception:
            pass
        gen = dict(result.generation or {})
        gen["stage"] = self.state.run.stage.value
        gen["phase"] = OptimizationPhase.SPEED.value
        result.generation = gen
        if result.status != ConfigurationStatus.PASSED and result.status != "passed":
            result.status = classify_error(result.error)
            result.score = None
            result.quality_score = None
            result.score_breakdown = None
            try:
                config_repo.save(result)
            except Exception as e:
                logger.error("Failure record save failed", error=str(e))
            self._log_event(str(result.status.value).upper(), (result.error or "")[:120])
        else:
            result.status = ConfigurationStatus.PASSED
            try:
                config_repo.save(result)
            except Exception as e:
                logger.error("Probe record save failed", error=str(e))
        if result.id not in [c.id for c in self.state.tested_configs]:
            self.state.tested_configs.append(result)
        if result.status != ConfigurationStatus.PASSED:
            key = self._candidate_key(result.config)
            self.state.probe_attempts[repr(key)] = (
                self.state.probe_attempts.get(repr(key), 0) + 1
            )
        return result

    async def _probe_speed_item(self, item, speed_ctx: int, tested_keys: set):
        """Probe one staged plan item (throughput items also measure aggregate)."""
        if self._candidate_key(item.config) in tested_keys:
            return None
        if self._probe_completed(self._candidate_key(item.config)):
            self._log_event(
                "RESUME_SKIP probe_completed", f"key attempts exhausted for {item.stage}"
            )
            return None
        if self._unverified_unload_block():
            return None
        # Same host guard as the full-suite path: a foreign load poisons a
        # speed probe just as badly as a quality run.
        await self.guard_host()
        res = await self.benchmark.run_speed_probe(
            self.state.run.model.id,
            item.config,
            speed_ctx,
            repetitions=self.state.probe_repetitions or 1,
        )
        if res is None:
            return None
        if item.kind == "throughput" and res.status in (
            ConfigurationStatus.PASSED,
            "passed",
        ):
            try:
                tp = await self.benchmark.measure_throughput(
                    self.state.run.model.id,
                    item.config,
                    item.config.parallel or 2,
                )
                # Standard key: feeds workload_tiebreak_key/effective_throughput.
                # Single-stream generation_tok_s stays the primary speed metric.
                res.generation = {**(res.generation or {}), "throughput_measured": tp}
            except Exception as e:
                logger.error("Throughput probe failed", error=str(e))
        rec = await self._record_probe_result(res, speed_ctx)
        tested_keys.add(self._candidate_key(item.config))
        return rec

    async def _safe_probe(self, item, speed_ctx: int, tested_keys: set):
        """One probe that can never kill the phase (crash -> event + skip)."""
        try:
            return await self._probe_speed_item(item, speed_ctx, tested_keys)
        except Exception as e:
            logger.error("Speed probe crashed", error=str(e))
            self._log_event("Probe error", str(e)[:120])
            return None

    def _find_tested_probe(self, item) -> ConfigurationResult | None:
        """Already-measured equivalent of a skipped plan item (resume path)."""
        key = self._candidate_key(item.config)
        for c in self.state.tested_configs:
            if (
                self._candidate_key(c.config) == key
                and (c.generation or {}).get("phase") == OptimizationPhase.SPEED.value
                and c.status == ConfigurationStatus.PASSED
            ):
                return c
        return None

    def _classify_budget(self, s0_result: ConfigurationResult | None) -> str:
        """Budget class from S0 (or its resume equivalent); records baseline."""
        from lm_optimizer.services.speed_probe import (
            budget_class_for,
            probe_repetitions_for,
        )

        s0_speed = (
            s0_result.get_avg_generation_tok_s()
            if s0_result is not None and s0_result.status == ConfigurationStatus.PASSED
            else 0.0
        )
        budget = budget_class_for(s0_speed)
        self.state.budget_class = budget
        try:
            user_reps = (self.state.run.benchmark_params or {}).get("speed_probe_repetitions")
        except Exception:
            user_reps = None
        self.state.probe_repetitions = probe_repetitions_for(budget, user_reps)
        if s0_result is not None and s0_result.status == ConfigurationStatus.PASSED:
            self.state.baseline_result = s0_result
            self.state.run.baseline_config_id = s0_result.id
            self.state.run.baseline_metrics = {
                "generation_tok_s": s0_result.get_avg_generation_tok_s(),
                "prompt_tok_s": s0_result.get_avg_prompt_tok_s(),
                "estimated_ttft_ms": s0_result.get_avg_estimated_ttft_ms(),
                "context_length": s0_result.context_length,
                "peak_vram_gb": s0_result.peak_vram_gb,
                "peak_ram_gb": s0_result.peak_ram_gb,
                "quality_overall": None,
                "stability": s0_result.stability_score,
            }
        return budget

    async def _phase_speed(self) -> dict:
        """SPEED: staged cheap probes at the frozen speed context."""
        from lm_optimizer.services.speed_probe import finalist_limit_for
        from lm_optimizer.services.speed_search import build_speed_plan, select_frontier

        self.state.phase = OptimizationPhase.SPEED.value
        self.state.run.phase = OptimizationPhase.SPEED.value
        self.state.run.stage = OptimizationStage.SPEED_DISCOVERY
        speed_ctx = self.state.speed_context or 2048
        anchor = self._default_anchor()
        caps = self._speed_caps()
        plan, notes = build_speed_plan(anchor, speed_ctx, caps)
        if getattr(self.benchmark, "mtp_test", None) is True:
            # MTP-only mode: keep S0 (budget classification) plus the
            # speculative/MTP probes; everything else is out of scope.
            plan = plan[:1] + [
                p for p in plan[1:] if str(p.stage).upper().startswith(("S1", "S2"))
            ]
            notes = list(notes)
            spec_n = len(plan) - 1
            notes.append(
                f"MTP-only mode: {spec_n} speculative probe(s) retained"
                if spec_n
                else "MTP-only mode: no draft model discovered - S0 only, "
                "speculative path not auto-probed"
            )

        # Preset/user probe cap (spec §4.2): bound the number of model loads.
        # The plan is priority-ordered (S0, high-impact toggles first), so the
        # cap drops the lowest-value tail rather than random probes.
        probe_cap = (self.state.run.benchmark_params or {}).get("max_speed_probes")
        try:
            probe_cap = int(probe_cap) if probe_cap is not None else None
        except (TypeError, ValueError):
            probe_cap = None
        if probe_cap is not None:
            probe_cap = max(1, probe_cap)
            if len(plan) > probe_cap:
                notes = list(notes)
                notes.append(
                    f"probe cap {probe_cap}: {len(plan) - probe_cap} "
                    "lower-priority probe(s) skipped"
                )
                plan = plan[:probe_cap]

        self.state.excluded_notes = notes
        self.state.planned_total = len(plan) + 2
        # Checkpoint BEFORE DB save so checkpoint phase >= DB stage on crash
        self._checkpoint("phase:speed:start")
        run_repo.save(self.state.run)
        self._log_event("SPEED discovery started", f"ctx={speed_ctx} probes~{len(plan)}")
        await self._pause_point()

        # Resume skips completed PASSED probes; FAILED probes are retried
        # (transient load failures, e.g. aborted runs, must not poison resume).
        tested_keys = {
            self._candidate_key(c.config)
            for c in self.state.tested_configs
            if (c.generation or {}).get("phase") == OptimizationPhase.SPEED.value
            and c.status == ConfigurationStatus.PASSED
        }

        # S0 first: classifies the budget for everything else.
        s0 = plan[0] if plan else None
        s0_result = await self._safe_probe(s0, speed_ctx, tested_keys) if s0 else None
        if s0_result is None and s0 is not None:
            # Resume-skip or crash: reuse the already-measured S0 equivalent
            # instead of degrading to VERY_SLOW on missing data.
            s0_result = self._find_tested_probe(s0)
        budget = self._classify_budget(s0_result)
        self._log_event("Speed budget", f"{budget} reps={self.state.probe_repetitions}")

        for item in plan[1:]:
            if self.state.should_cancel:
                break
            await self._pause_point()
            await self._safe_probe(item, speed_ctx, tested_keys)

        # S5: interactions of the strongest settings only, within the probe cap.
        passed = [c for c in self.state.tested_configs if c.status == ConfigurationStatus.PASSED]
        if passed:
            from lm_optimizer.services.speed_search import build_interaction_probes

            top = sorted(passed, key=lambda r: r.get_avg_generation_tok_s(), reverse=True)[:2]
            s5_items = build_interaction_probes([t.config for t in top], speed_ctx)
            s5_budget = None if probe_cap is None else max(0, probe_cap - len(plan))
            s5_new = 0
            for item in s5_items:
                if self.state.should_cancel:
                    break
                if s5_budget is not None and s5_new >= s5_budget:
                    break
                await self._pause_point()
                if self._candidate_key(item.config) in tested_keys:
                    continue
                s5_new += 1
                await self._safe_probe(item, speed_ctx, tested_keys)
            self._log_event(
                "S5 interactions",
                f"generated={len(s5_items)} already-tested={len(s5_items) - s5_new} "
                f"new={s5_new} (duplicates never re-run"
                + (f"; cap={probe_cap}" if probe_cap is not None else "")
                + ")",
            )

        probed = [
            c
            for c in self.state.tested_configs
            if (c.generation or {}).get("phase") == OptimizationPhase.SPEED.value
            and c.status == ConfigurationStatus.PASSED
        ]
        limit = finalist_limit_for(budget)
        frontier = select_frontier(probed, max_band=limit)
        if frontier["winner"] is not None:
            self.state.raw_fastest = frontier["winner"]
            self.state.run.raw_fastest_config_id = frontier["winner"].id
        self.state.speed_finalists = [
            str(c.id) for c in (frontier["finalists"] or [])
        ]
        if frontier["contrarian"] is not None and str(frontier["contrarian"].id) not in (
            self.state.speed_finalists
        ):
            self.state.speed_finalists.append(str(frontier["contrarian"].id))
        self._sync_phase_params()
        run_repo.save(self.state.run)
        self._checkpoint("phase:speed:complete")
        await self._broadcast({"stage": "speed_discovery", "event": "speed_complete"})
        return frontier

    def _tag_phase(self, result: ConfigurationResult, phase: str) -> None:
        """Stamp the measurement phase (SPEED/QUALITY/RECOVERY/FINAL_VALIDATION).

        Candidate identity (candidate key) is unchanged; only provenance is
        recorded so later stages can select same-kind measurements.
        """
        result.generation = {**(result.generation or {}), "phase": phase}
        try:
            config_repo.save(result)
        except Exception as e:
            logger.error("Phase tag save failed", error=str(e))

    async def _quality_finalist(
        self, finalist: ConfigurationResult, quality_ctx: int, quality_done: dict
    ) -> ConfigurationResult | None:
        """Full-suite quality test of one finalist (resume reuses tested)."""
        data = {k: v for k, v in finalist.config.to_dict().items()}
        data["context_length"] = quality_ctx
        qcfg = LoadConfiguration(
            **{k: v for k, v in data.items() if hasattr(LoadConfiguration, k)}
        )
        qcfg.context_length = quality_ctx
        existing = quality_done.get(self._candidate_key(qcfg))
        result = (
            existing
            if existing is not None
            else await self._test_config(qcfg, quality_ctx, score_context=False)
        )
        if result is None:
            return None
        if existing is None:
            # Fresh full-suite measurement at the frozen quality context.
            self._tag_phase(result, OptimizationPhase.QUALITY.value)
        else:
            _st = getattr(result, "status", "")
            status = getattr(_st, "value", str(_st))
            self._log_event(
                "RESUME_SKIP quality_completed",
                f"config={str(result.id)[:8]} status={status}",
            )
        # P3: deterministic Quality Lab on every PASSING finalist, with hard
        # gates that can veto the winner (a broken output never wins).
        if (
            getattr(self.benchmark, "quality_lab", False)
            and result.status == ConfigurationStatus.PASSED
            and existing is None
        ):
            lab = await self._run_quality_lab(self.state.run.model.id)
            if result.quality_score is not None:
                try:
                    result.quality_score.details["quality_lab"] = lab
                except Exception:
                    pass
            if lab.get("hard_gate_failures"):
                result.status = ConfigurationStatus.QUALITY_FAILED
                self._log_event(
                    "QUALITY_LAB hard gate failed",
                    ",".join(lab["hard_gate_failures"])[:180],
                )
                try:
                    config_repo.save(result)
                except Exception as e:
                    logger.error("Quality Lab result save failed", error=str(e))
            else:
                self._log_event(
                    "QUALITY_LAB passed",
                    f"mean={lab.get('mean', 0):.3f} tasks={lab.get('tasks', 0)}",
                )
        if result.id not in [c.id for c in self.state.tested_configs]:
            self.state.tested_configs.append(result)
        return result

    async def _run_quality_lab(self, model_id: str) -> dict:
        """Run the deterministic Quality Lab tasks against the loaded model.

        Returns a compact summary merged into the configuration's quality
        details: mean score, task count and the hard gates that failed.
        Every task is a real generation on the currently loaded config.
        """
        import statistics as _stats

        from lm_optimizer.services.quality_lab import default_tasks, run_task

        async def _chat(prompt: str, max_tokens: int) -> str:
            response = await self.client.chat_completion(
                model=model_id,
                input_text=prompt,
                temperature=0.0,
                max_output_tokens=max_tokens,
            )
            choices = response.get("choices") or []
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content", "") or ""

        outcomes = []
        hard_gate_failures: list[str] = []
        for task in default_tasks():
            try:
                outcome = await run_task(task, _chat)
            except Exception as e:  # noqa: BLE001 - one task must not abort the suite
                logger.warning("Quality Lab task failed", task=task.id, error=str(e)[:160])
                continue
            outcomes.append(outcome.as_dict())
            if not outcome.gate_passed:
                hard_gate_failures.append(task.id)
        scores = [o["score"] for o in outcomes]
        return {
            "mean": _stats.fmean(scores) if scores else 0.0,
            "tasks": len(outcomes),
            "hard_gate_failures": hard_gate_failures,
            "outcomes": outcomes,
        }

    def _quality_result_exists(
        self, candidate_key: tuple, context_length: int
    ) -> ConfigurationResult | None:
        """Completed quality suite for a candidate, regardless of pass/fail.

        A suite that ran to a recorded quality result is completed work:
        PASSED, QUALITY_FAILED, or an explicit evaluator failure all count,
        provided the suite actually ran (recorded per-test evidence). No
        synthetic score is used to mark completion.
        """
        for c in self.state.tested_configs:
            if self._candidate_key(c.config) != candidate_key:
                continue
            if c.context_length != context_length:
                continue
            gen = c.generation or {}
            if gen.get("quality_completed") is True:
                return c
            # Backfill for rows written before the marker existed: a recorded
            # per-test evidence block means the suite ran to completion.
            status = getattr(getattr(c, "status", ""), "value", str(getattr(c, "status", "")))
            if status in ("passed", "quality_failed") and gen.get("quality_by_test"):
                return c
        return None

    def _probe_completed(self, candidate_key: tuple) -> bool:
        """A probe key is completed when passed, or failed twice (no blind repeats)."""
        for c in self.state.tested_configs:
            if self._candidate_key(c.config) != candidate_key:
                continue
            gen = c.generation or {}
            if gen.get("phase") != OptimizationPhase.SPEED.value:
                continue
            _st = getattr(c, "status", "")
            if getattr(_st, "value", str(_st)) == "passed":
                return True
        return self.state.probe_attempts.get(repr(candidate_key), 0) >= 2

    def _file_quality_result(
        self, result: ConfigurationResult | None, safe: list, failed: list
    ) -> None:
        """Track raw-fastest (even when failed) and sort safe vs failed."""
        if result is None:
            return
        speed = result.get_avg_generation_tok_s()
        raw = self.state.raw_fastest
        if raw is None or speed > raw.get_avg_generation_tok_s():
            self.state.raw_fastest = result
            self.state.run.raw_fastest_config_id = result.id
        if result.status == ConfigurationStatus.PASSED:
            safe.append(result)
            self._update_best(result)
        elif result.status == ConfigurationStatus.QUALITY_FAILED:
            failed.append(result)

    def _finalist_limit(self) -> int:
        """Quality finalist cap: explicit int wins; missing/None/invalid safe.

        None means "use the budget default" (dict.get's fallback does NOT
        apply when the key exists with None — the BUG-7 silent drop).
        """
        from lm_optimizer.services.speed_probe import finalist_limit_for

        raw = (self.state.run.benchmark_params or {}).get("quality_finalist_limit")
        if raw is None:
            return finalist_limit_for(self.state.budget_class or "NORMAL")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 4

    @staticmethod
    def rotate_order(items: list, offset: int) -> list:
        """Rotate a list so a different candidate leads each round.

        Master spec §7.3: finalists must be rotated (ABC/BCA/CAB) so a later
        candidate is not systematically penalized by thermal throttling -
        the first-measured finalist always runs on the coolest machine.
        """
        if not items or offset <= 0:
            return list(items)
        k = int(offset) % len(items)
        return list(items[k:]) + list(items[:k])

    async def _phase_quality(self, frontier: dict) -> tuple[list, list]:
        """QUALITY: full suite on frozen finalists at the frozen quality ctx."""
        self.state.phase = OptimizationPhase.QUALITY.value
        self.state.run.phase = OptimizationPhase.QUALITY.value
        self.state.run.stage = OptimizationStage.QUALITY_CHECK
        quality_ctx = self.state.quality_context or 8192
        # Checkpoint BEFORE DB save so checkpoint phase >= DB stage on crash
        self._checkpoint("phase:quality:start")
        run_repo.save(self.state.run)
        ordered = list(frontier.get("finalists") or [])
        contrarian = frontier.get("contrarian")
        if contrarian is not None and all(c.id != contrarian.id for c in ordered):
            ordered.append(contrarian)
        limit = self._finalist_limit()
        ordered = ordered[: max(limit + 1, 1)]
        # Rotate the finalist order per run so the coolest first slot is not
        # always the same candidate (spec §7.3: ABC/BCA/CAB rotation).
        if len(ordered) > 1:
            seed = str(self.state.run.id).replace("-", "")
            try:
                offset = int(seed[:8], 16) % len(ordered)
            except ValueError:
                offset = 0
            ordered = self.rotate_order(ordered, offset)
        self.state.planned_total += len(ordered)  # one suite per finalist
        self._log_event("QUALITY validation started", f"ctx={quality_ctx} n={len(ordered)}")

        # Resume: reuse already quality-tested configs (never re-run).
        # Completed = marker OR backfill (status + stage + recorded evidence).
        quality_done = {}
        for c in self.state.tested_configs:
            if c.context_length != quality_ctx:
                continue
            found = self._quality_result_exists(self._candidate_key(c.config), quality_ctx)
            if found is not None:
                quality_done[self._candidate_key(c.config)] = found

        safe: list[ConfigurationResult] = []
        failed: list[ConfigurationResult] = []
        for finalist in ordered:
            if self.state.should_cancel:
                break
            await self._pause_point()
            result = await self._quality_finalist(finalist, quality_ctx, quality_done)
            self._file_quality_result(result, safe, failed)
        # Phase A winner: fastest quality-passing config (R14).
        if safe:
            fastest = max(safe, key=lambda r: r.get_avg_generation_tok_s())
            self.state.best_config = fastest
            self.state.run.best_config_id = fastest.id
            self._log_event(
                "QUALITY_SAFE winner",
                f"{fastest.get_avg_generation_tok_s():.1f} tok/s",
            )
        self._sync_phase_params()
        run_repo.save(self.state.run)
        self._checkpoint("phase:quality:complete")
        await self._broadcast({"stage": "quality_check", "event": "quality_complete"})
        return safe, failed

    async def _recheck_quality(
        self, cfg: LoadConfiguration, ctx: int, test_names: list[str]
    ) -> tuple[bool, object, ConfigurationResult | None]:
        """Rerun ONLY named quality tests (recovery subset, never full suite).

        The measurement context comes from `cfg`, not from `ctx`. A rollback
        changes cfg (e.g. context_length 32768 -> 4096) while `ctx` still holds
        the FAILED result's context; measuring at `ctx` would silently re-run
        the exact configuration recovery just decided to abandon, at 8x the
        cost, and record a quality_after that describes a run that never
        happened. `ctx` is only the fallback when cfg carries no context.
        """
        effective_ctx = int(getattr(cfg, "context_length", None) or ctx)
        cases = [
            c
            for c in self.benchmark.create_cases_for_context(effective_ctx)
            if c.name in set(test_names)
        ]
        if not cases:
            return False, None, None
        res = await self.benchmark.run_cases(
            self.state.run.model.id, cfg, effective_ctx, cases
        )
        try:
            scores = self.quality.evaluate_all(self.state.run.model.id, res.metrics)
            agg = self.quality.aggregate_quality(scores)
            return bool(self.quality.passes_threshold(agg)), agg, res
        except Exception as e:
            # An evaluator crash fails the recheck — it must never kill recovery.
            logger.error("Quality recheck crashed", error=str(e))
            self._log_event("Recheck evaluator crashed", str(e)[:120])
            return False, None, res

    async def _run_recovery(
        self,
        failed_result: ConfigurationResult,
        last_good: LoadConfiguration,
        ref_kind: str = "safe",
    ) -> tuple[ConfigurationResult | None, str | None]:
        """Bounded causal rollback: reconfirm, then single-param rollbacks."""
        from lm_optimizer.services.config_delta import config_delta, rank_delta
        from lm_optimizer.services.recovery import (
            RECOVERY_BUDGET_DEFAULT,
            pairwise_allowed,
            plan_recovery,
        )

        self.state.phase = OptimizationPhase.RECOVERY.value
        self.state.run.phase = OptimizationPhase.RECOVERY.value
        self.state.run.stage = OptimizationStage.RECOVERY
        # Checkpoint BEFORE DB save so checkpoint phase >= DB stage on crash
        self._checkpoint("phase:recovery:start")
        run_repo.save(self.state.run)
        from lm_optimizer.services.recovery import rank_failed_tests

        qbt = (failed_result.generation or {}).get("quality_by_test") or {}
        # Explicit severity ordering; None-safe (null checks = unavailable
        # evidence, rechecked first, never compared numerically).
        failed_names = rank_failed_tests(qbt) or sorted(qbt)
        ctx = failed_result.context_length
        try:
            budget = int(
                (self.state.run.benchmark_params or {}).get(
                    "recovery_budget", RECOVERY_BUDGET_DEFAULT
                )
            )
        except (TypeError, ValueError):
            budget = RECOVERY_BUDGET_DEFAULT

        # Step 1: reconfirm failed tests only (noise vs deterministic).
        ok, agg, _res = await self._recheck_quality(failed_result.config, ctx, failed_names)
        self.state.recovery_log.append(
            {
                "event": "reconfirm",
                "config_id": str(failed_result.id),
                "tests": failed_names,
                "passed": ok,
                "reference": ref_kind,
            }
        )
        self.state.recovery_cursor = {
            "finalist_id": str(failed_result.id),
            "step": len(self.state.recovery_log),
        }
        self._persist_recovery_step("reconfirm")
        if ok:
            # Noisy failure: the config passes NOW — admit it through the full
            # suite so it can compete (never silently dropped as FAILED).
            self._log_event("Recovery: noisy failure (reconfirm passed)", "")
            admitted = await self._admit_recovered(failed_result.config, ctx)
            self._sync_phase_params()
            return admitted, None

        # Steps 2-4: single-param rollbacks in suspect order.
        delta = config_delta(last_good, failed_result.config)
        ranked = rank_delta(delta)
        for suspect in plan_recovery(ranked, budget):
            if self.state.should_cancel:
                break
            await self._pause_point()
            admitted = await self._try_rollback_suspect(
                suspect, failed_result, last_good, ctx, failed_names
            )
            if admitted is not None:
                return admitted, suspect

        # Step 5: bounded pairwise rollback for tiny deltas only.
        if pairwise_allowed(delta):
            top2 = rank_delta(delta)[:2]
            if len(top2) == 2:
                admitted = await self._try_pairwise(
                    top2, failed_result, last_good, ctx, failed_names
                )
                if admitted is not None:
                    return admitted, "+".join(top2)
        self._log_event("Recovery: unrecovered", f"delta={sorted(delta)}")
        self.state.recovery_cursor = {
            "finalist_id": str(failed_result.id),
            "step": len(self.state.recovery_log),
            "decision": "unrecovered",
        }
        self._persist_recovery_step("unrecovered")
        return None, None

    def _persist_recovery_step(self, reason: str) -> None:
        """Durable recovery progress: sync + DB save + atomic checkpoint.

        Called after EVERY recovery state transition (reconfirm, rollback,
        pairwise, admission, unrecovered) so a killed process never loses
        more than the single in-flight operation.
        """
        n = len(self.state.recovery_log)
        cur = self.state.recovery_cursor or {}
        cur["step"] = n
        try:
            from datetime import datetime as _dt

            cur["updated_at"] = _dt.now().isoformat(timespec="seconds")
        except Exception:
            pass
        self.state.recovery_cursor = cur
        self._sync_phase_params()
        try:
            run_repo.save(self.state.run)
        except Exception as e:
            logger.error("Recovery persist save failed", error=str(e))
        self._checkpoint(f"recovery:{reason}:{n}")
        self._log_event("CHECKPOINT_PERSIST", f"recovery_step={n} reason={reason}")

    def _rolled_config(
        self, failed_result: ConfigurationResult, overrides: dict
    ) -> LoadConfiguration:
        """Failed config with `overrides` applied (None clears back to unset)."""
        data = {k: v for k, v in failed_result.config.to_dict().items()}
        data.update(overrides)
        rolled = LoadConfiguration(
            **{k: v for k, v in data.items() if hasattr(LoadConfiguration, k)}
        )
        for key, value in overrides.items():
            if value is None:
                setattr(rolled, key, None)
        return rolled

    async def _admit_recovered(
        self, rolled: LoadConfiguration, ctx: int
    ) -> ConfigurationResult | None:
        """Full-suite admission of a subset-passing rollback (PASSED + scored).

        Admission must measure `rolled`, not the failed result's context, for
        the same reason `_recheck_quality` does - otherwise a recovered config
        is validated under settings it will never run with.
        """
        effective_ctx = int(getattr(rolled, "context_length", None) or ctx)
        admitted = await self._test_config(rolled, effective_ctx, score_context=False, min_speed=self.state.run.advanced_settings.get("min_speed", 0.0))
        if admitted is not None and admitted.status == ConfigurationStatus.PASSED:
            self._tag_phase(admitted, OptimizationPhase.RECOVERY.value)
            if admitted.id not in [c.id for c in self.state.tested_configs]:
                self.state.tested_configs.append(admitted)
            self._update_best(admitted)
            self._persist_recovery_step("admitted")
            return admitted
        return None

    async def _try_rollback_suspect(
        self,
        suspect: str,
        failed_result: ConfigurationResult,
        last_good: LoadConfiguration,
        ctx: int,
        failed_names: list[str],
    ) -> ConfigurationResult | None:
        """One rollback probe: recheck failed tests, admit on recovery."""
        from dataclasses import asdict

        from lm_optimizer.services.recovery import RecoveryAttempt

        old_value = getattr(last_good, suspect, None)
        rolled = self._rolled_config(failed_result, {suspect: old_value})
        ok, agg, _res = await self._recheck_quality(rolled, ctx, failed_names)
        attempt = RecoveryAttempt(
            suspect=suspect,
            old_value=getattr(failed_result.config, suspect, None),
            new_value=old_value,
            recheck_passed=ok,
            quality_after=getattr(agg, "overall", None),
        )
        self.state.recovery_log.append(asdict(attempt))
        self._persist_recovery_step(f"rollback:{suspect}")
        if not ok:
            return None
        admitted = await self._admit_recovered(rolled, ctx)
        if admitted is not None:
            self._log_event("Recovery: culprit found", suspect)
        return admitted

    async def _try_pairwise(
        self,
        top2: list[str],
        failed_result: ConfigurationResult,
        last_good: LoadConfiguration,
        ctx: int,
        failed_names: list[str],
    ) -> ConfigurationResult | None:
        """One bounded two-param rollback (tiny deltas only)."""
        rolled = self._rolled_config(
            failed_result, {s: getattr(last_good, s, None) for s in top2}
        )
        ok, _agg, _res = await self._recheck_quality(rolled, ctx, failed_names)
        self.state.recovery_log.append({"event": "pairwise", "suspects": top2, "passed": ok})
        self._persist_recovery_step("pairwise")
        if not ok:
            return None
        return await self._admit_recovered(rolled, ctx)

    async def _phase_final_validation(self) -> None:
        """Adaptive validation of the final runtime config (min 2, max 5)."""
        self.state.phase = OptimizationPhase.FINAL_VALIDATION.value
        self.state.run.phase = OptimizationPhase.FINAL_VALIDATION.value
        self.state.run.stage = OptimizationStage.FINAL_VALIDATION
        # Checkpoint BEFORE DB save so checkpoint phase >= DB stage on crash
        self._checkpoint("phase:final_validation:start")
        run_repo.save(self.state.run)
        best = self.state.best_config
        if best is None:
            return
        await self._pause_point()
        key = self._candidate_key(best.config)
        # Same-kind measurements only: a 64-token SPEED probe must never
        # become a validation repetition or skew final stability.
        speeds = [
            c.get_avg_generation_tok_s()
            for c in self.state.tested_configs
            if self._candidate_key(c.config) == key
            and (c.generation or {}).get("phase") != OptimizationPhase.SPEED.value
            and c.get_avg_generation_tok_s() > 0
        ]
        reps = self.state.run.validation_repetitions or 5
        try:
            vmin = max(1, int(getattr(self.state.run, "validation_repetitions_min", 2) or 2))
            vmax = max(vmin, int(getattr(self.state.run, "validation_repetitions_max", 5) or 5))
        except (TypeError, ValueError):
            vmin, vmax = 2, 5
        if len(speeds) >= 2:
            import statistics as _stats

            mean = _stats.mean(speeds)
            cv = (_stats.pstdev(speeds) / mean) if mean > 0 else 1.0
            stability = max(0.0, 1.0 - cv)
        else:
            # Few samples: trust the existing stability_score semantics.
            stability = getattr(best, "stability_score", 0.0) or 0.0
        if len(speeds) >= 2 or getattr(best, "stability_score", None):
            reps = 2 if stability >= 0.95 else (3 if stability >= 0.85 else reps)
        # Preset bounds (spec §2): never fewer than min, never more than max.
        reps = max(vmin, min(reps, vmax))
        self.state.planned_total += reps  # one unit per validation repetition
        self._log_event(
            "Final validation", f"reps={reps} bounds={vmin}..{vmax}"
        )
        validated: list[ConfigurationResult] = []
        for _i in range(reps):
            if self.state.should_cancel:
                break
            data = {k: v for k, v in best.config.to_dict().items()}
            vcfg = LoadConfiguration(
                **{k: v for k, v in data.items() if hasattr(LoadConfiguration, k)}
            )
            # Winner invariant: validation must run at the OUTER context
            # (best.context_length). The nested config copy can lag behind
            # (e.g. speed 4096 vs quality 32768 winner) — never validate stale ctx.
            vcfg.context_length = best.context_length
            result = await self._test_config(vcfg, best.context_length, score_context=False, min_speed=self.state.run.advanced_settings.get("min_speed", 0.0))
            if result is not None and result.status == ConfigurationStatus.PASSED:
                self._tag_phase(result, OptimizationPhase.FINAL_VALIDATION.value)
                validated.append(result)
                if result.id not in [c.id for c in self.state.tested_configs]:
                    self.state.tested_configs.append(result)
        if validated:
            self.state.best_config = self._median_result(validated)
            self.state.run.best_config_id = self.state.best_config.id
        self.state.pareto_frontier = self._compute_pareto_frontier(self.state.tested_configs)
        self.state.run.pareto_config_ids = [c.id for c in self.state.pareto_frontier]
        run_repo.save(self.state.run)
        self._checkpoint("phase:final_validation:complete")

    def _phase_b_failed(self, error: str) -> dict:
        """Contained Phase B failure (optional phase never fails the run)."""
        result = {"error": error, "gpu_only_max": None, "system_max": None}
        self.state.phase_b_result = result
        self.state.run.phase_b_result = result
        self._sync_phase_params()
        run_repo.save(self.state.run)
        self._log_event("Phase B failed (contained)", error[:120])
        return result

    async def _phase_context_optional(
        self, winner: ConfigurationResult, sweep_fn=None
    ) -> dict | None:
        """Phase B (opt-in): max context on the FROZEN runtime config."""
        if not self.state.phase_b_enabled:
            return None
        from lm_optimizer.services import context_sweep as _sweep_mod

        sweep = sweep_fn or _sweep_mod.sweep
        self.state.phase = OptimizationPhase.CONTEXT_OPTIONAL.value
        self.state.run.phase = OptimizationPhase.CONTEXT_OPTIONAL.value
        self.state.run.stage = OptimizationStage.CONTEXT_OPTIONAL
        # Checkpoint BEFORE DB save so checkpoint phase >= DB stage on crash
        self._checkpoint("phase:context_optional:start")
        run_repo.save(self.state.run)
        model = self.state.run.model
        model_limit = model.context_limit or 131072
        min_ctx = self.state.quality_context or 8192

        async def _probe_at(cfg: LoadConfiguration, ctx: int) -> dict:
            data = {k: v for k, v in cfg.to_dict().items()}
            data["context_length"] = ctx
            pcfg = LoadConfiguration(
                **{k: v for k, v in data.items() if hasattr(LoadConfiguration, k)}
            )
            pcfg.context_length = ctx
            try:
                ok, tok_s, err = await self.benchmark.smoke_test(model.id, pcfg)
                return {"ok": bool(ok), "tok_s": float(tok_s or 0.0), "error": err or ""}
            except Exception as e:
                return {"ok": False, "tok_s": 0.0, "error": f"{type(e).__name__}: {e}"[:160]}

        frozen = winner.config
        try:
            gpu_only = await sweep(
                lambda ctx: _probe_at(frozen, ctx), min_ctx=min_ctx, max_ctx=model_limit
            )
        except Exception as e:
            # Phase B is optional: a sweep crash is contained, never fatal.
            logger.error("Phase B GPU-only sweep crashed", error=str(e))
            return self._phase_b_failed(f"gpu-only sweep: {type(e).__name__}: {e}"[:160])
        # System-max pass: same runtime, spillover variant ONLY when the
        # winner is full-GPU-resident (transparently labeled, never hidden).
        spill_cfg = frozen
        spill_note = "winner already spills; system pass reuses frozen config"
        kv_gpu = getattr(frozen, "offload_kv_cache_to_gpu", None)
        ratio = getattr(frozen, "gpu_ratio", None)
        if kv_gpu is True and (ratio is None or ratio >= 1.0):
            data = {k: v for k, v in frozen.to_dict().items()}
            data["gpu_ratio"] = 0.5
            spill_cfg = LoadConfiguration(
                **{k: v for k, v in data.items() if hasattr(LoadConfiguration, k)}
            )
            spill_note = "spillover variant: gpu_ratio 0.5 (labeled, Phase B only)"
        try:
            system = await sweep(
                lambda ctx, _c=spill_cfg: _probe_at(_c, ctx),
                min_ctx=min_ctx,
                max_ctx=model_limit,
            )
        except Exception as e:
            logger.error("Phase B system sweep crashed", error=str(e))
            return self._phase_b_failed(f"system sweep: {type(e).__name__}: {e}"[:160])
        gpu_max = gpu_only.get("maximum_stable_context")
        sys_max = system.get("maximum_stable_context")
        hw_limit = sys_max if sys_max is not None else gpu_max
        final_capacity = min(model_limit, hw_limit) if hw_limit is not None else model_limit
        result = {
            "gpu_only_max": gpu_max,
            "system_max": sys_max,
            "model_limit": model_limit,
            "hardware_stable_limit": hw_limit,
            "final_capacity": final_capacity,
            "spill_note": spill_note,
            "performance_optimal": gpu_only.get("performance_optimal"),
            "balanced_recommended": gpu_only.get("balanced_recommended"),
            "gpu_only_sweep": gpu_only,
            "system_sweep": system,
        }
        self.state.phase_b_result = result
        self.state.run.phase_b_result = result
        self._sync_phase_params()
        run_repo.save(self.state.run)
        self._checkpoint("phase:context_optional:complete")
        await self._broadcast({"stage": "context_optional", "event": "context_complete"})
        return result

    async def _recover_failed(
        self,
        failed: list,
        ref_config: LoadConfiguration | None,
        ref_kind: str = "safe",
    ) -> tuple[list, LoadConfiguration | None]:
        """Run bounded recovery for each failed finalist (skips without ref)."""
        from lm_optimizer.services.recovery import RECOVERY_BUDGET_DEFAULT

        try:
            _budget = int(
                (self.state.run.benchmark_params or {}).get(
                    "recovery_budget", RECOVERY_BUDGET_DEFAULT
                )
            )
        except (TypeError, ValueError):
            _budget = RECOVERY_BUDGET_DEFAULT
        recovered: list[ConfigurationResult] = []
        kind = ref_kind
        if ref_config is not None:
            for failed_result in failed:
                if self.state.should_cancel:
                    break
                # Upper-bound plan upfront (1 reconfirm + budget rollbacks);
                # finalize snaps any unused remainder away.
                self.state.planned_total += 1 + max(0, _budget)
                await self._pause_point()
                rec, _culprit = await self._run_recovery(failed_result, ref_config, kind)
                if rec is not None:
                    recovered.append(rec)
                    ref_config = rec.config
                    kind = "safe"  # admitted configs are quality-verified
        return recovered, ref_config

    async def _run_phase_ab(
        self, advanced_settings: dict, baseline_config: LoadConfiguration | None
    ) -> None:
        """Phase A/B orchestration: SPEED → QUALITY → RECOVERY → validation → B."""
        from lm_optimizer.services.speed_probe import (
            quality_context_for,
            speed_context_for,
        )

        model = self.state.run.model
        user_cap = (advanced_settings or {}).get("max_context")
        # Granular modes may pin the speed context explicitly (--speed-only /
        # --mtp-only --context N). An explicit value wins over the derived one
        # and is clamped to the model's advertised context limit.
        explicit_speed = (advanced_settings or {}).get("speed_context")
        if explicit_speed:
            try:
                limit = int(getattr(model, "context_limit", 0) or 0)
                speed_ctx = min(max(256, int(explicit_speed)), limit or int(explicit_speed))
            except (TypeError, ValueError):
                speed_ctx = speed_context_for(user_cap, model.context_limit)
        else:
            speed_ctx = speed_context_for(user_cap, model.context_limit)
        quality_ctx = quality_context_for(user_cap, model.context_limit, speed_ctx)
        self.state.speed_context = speed_ctx
        self.state.quality_context = quality_ctx
        self.state.run.speed_context = speed_ctx
        self.state.run.quality_context = quality_ctx
        self.state.score_context = False
        self.state.phase_b_enabled = bool((advanced_settings or {}).get("optimize_context", False))
        self.state.run.phase_b_enabled = self.state.phase_b_enabled
        self._sync_phase_params()
        run_repo.save(self.state.run)

        # Anchor source only (no expensive baseline benchmark; S0 is the anchor).
        skips = phase_skips_of(advanced_settings)
        captured = await self._granular_baseline_or_none(model, baseline_config, skips)
        if captured is not None and self.state.baseline_result is None:
            self.state.baseline_result = ConfigurationResult(
                config=captured,
                context_length=speed_ctx,
                status=ConfigurationStatus.SKIPPED,
            )
        self._log_event(
            "Phase A contexts frozen", f"speed={speed_ctx} quality={quality_ctx}"
        )

        if skips:
            self._log_event(
                "Phase skips active", f"skipped: {', '.join(sorted(skips))}"
            )

        if "speed" in skips:
            self._log_event("Phase SPEED skipped", "--skip-phases")
            frontier = granular_quality_frontier(
                skips, advanced_settings, self.state.baseline_result
            )
            if frontier["finalists"]:
                self._log_event(
                    "QUALITY candidates from baseline",
                    "granular mode without speed phase: baseline config only",
                )
        else:
            frontier = await self._phase_speed()
        if self.state.should_cancel:
            raise asyncio.CancelledError()
        if "quality" in skips:
            self._log_event("Phase QUALITY skipped", "--skip-phases (no winner path)")
            safe, failed = [], []
        else:
            safe, failed = await self._phase_quality(frontier)
        if self.state.should_cancel:
            raise asyncio.CancelledError()

        # Recovery reference: fastest safe config, else passed S0 anchor.
        ref_config, ref_kind = self._recovery_reference(safe)
        if "recovery" in skips:
            self._log_event("Phase RECOVERY skipped", "--skip-phases")
            recovered, _ref = [], None
        else:
            recovered, _ref = await self._recover_failed(failed, ref_config, ref_kind)
        if self.state.should_cancel:
            raise asyncio.CancelledError()

        all_safe = safe + [r for r in recovered if r.status == ConfigurationStatus.PASSED]
        if all_safe:
            fastest = max(all_safe, key=lambda r: r.get_avg_generation_tok_s())
            self.state.best_config = fastest
            self.state.run.best_config_id = fastest.id
        await self._pause_point()
        if "validation" in skips:
            self._log_event("Phase FINAL_VALIDATION skipped", "--skip-phases")
        else:
            await self._phase_final_validation()
        if self.state.should_cancel:
            raise asyncio.CancelledError()
        await self._pause_point()
        if "context" in skips and self.state.phase_b_enabled:
            self._log_event("Phase CONTEXT skipped", "--skip-phases (Phase B off)")
            self.state.phase_b_enabled = False
            self.state.run.phase_b_enabled = False
        if self.state.phase_b_enabled and self.state.best_config is not None:
            await self._phase_context_optional(self.state.best_config)

        self._recompute_all_scores()
        await self._finalize()

    async def _finalize(self) -> None:
        """Finalize optimization run with comparison to baseline."""
        from lm_optimizer.services.run_summary import classify_run, final_status_for_run

        self.state.run.configurations = self.state.tested_configs
        # Only validated PASSED configs may win.
        if self.state.best_config and self.state.best_config.status != ConfigurationStatus.PASSED:
            self.state.best_config = None
            self.state.run.best_config_id = None

        # --- Speed floor enforcement (spec §3) ---
        speed_floor = getattr(self.state.run, "min_generation_tok_s", 2.0)
        enforce_floor = getattr(self.state.run, "enforce_min_generation_speed", True)
        speed_floor_met = False
        diagnostic_candidate = None
        diagnostic_speed = None
        if self.state.best_config:
            diagnostic_candidate = self.state.best_config
            diagnostic_speed = self.state.best_config.get_avg_generation_tok_s()
            speed_floor_met = diagnostic_speed >= speed_floor
            if enforce_floor and not speed_floor_met:
                # The fastest candidate stays visible as a diagnostic result;
                # it is simply not eligible to be the recommended winner.
                self.state.best_config = None
                self.state.run.best_config_id = None
                self.state.run.no_winner_reason = "NO_CONFIG_MEETS_SPEED_FLOOR"
                self._log_event(
                    "Speed floor not met",
                    f"best={diagnostic_speed:.2f} tok/s < floor={speed_floor:.2f} tok/s (enforced)",
                )
            elif not enforce_floor and not speed_floor_met:
                self._log_event(
                    "Speed floor not met (informational)",
                    f"best={diagnostic_speed:.2f} tok/s < floor={speed_floor:.2f} tok/s (not enforced)",
                )

        if self.state.best_config:
            self.state.run.best_config_id = self.state.best_config.id

        # Record run-level pruning for reports (no schema change: search_space JSON)
        if self.state.flash_required:
            self.state.run.search_space["flash_pruned"] = True
        # Record speed floor status
        self.state.run.search_space["speed_floor"] = {
            "value": speed_floor,
            "enforced": enforce_floor,
            "met": speed_floor_met,
            # Diagnostic fastest candidate stays visible even when it is not
            # eligible as a winner (spec §3.3 "never fabricate a passing winner").
            "fastest_candidate_id": (
                str(diagnostic_candidate.id) if diagnostic_candidate is not None else None
            ),
            "fastest_tok_s": (
                round(float(diagnostic_speed), 3)
                if diagnostic_speed is not None
                else None
            ),
        }
        # Persist the explicit no-winner reason (no DB column: benchmark_params
        # JSON is the durable carrier, so resume/reports keep the same meaning).
        if self.state.run.no_winner_reason:
            self.state.run.benchmark_params["no_winner_reason"] = (
                self.state.run.no_winner_reason
            )

        # Calculate percentage changes vs baseline if available
        if self.state.run.baseline_metrics and self.state.best_config:
            baseline = self.state.run.baseline_metrics
            best = self.state.best_config
            changes = {}
            try:
                if baseline.get("generation_tok_s") and best.get_avg_generation_tok_s():
                    changes["generation_speed_change"] = (
                        (best.get_avg_generation_tok_s() - baseline["generation_tok_s"])
                        / baseline["generation_tok_s"]
                        * 100
                    )
                if baseline.get("prompt_tok_s") and best.get_avg_prompt_tok_s():
                    changes["prompt_speed_change"] = (
                        (best.get_avg_prompt_tok_s() - baseline["prompt_tok_s"])
                        / baseline["prompt_tok_s"]
                        * 100
                    )
                if baseline.get("context_length") and best.context_length:
                    changes["context_change"] = (
                        (best.context_length - baseline["context_length"])
                        / baseline["context_length"]
                        * 100
                    )
                if baseline.get("peak_vram_gb") and best.peak_vram_gb:
                    changes["vram_change"] = (
                        (best.peak_vram_gb - baseline["peak_vram_gb"])
                        / baseline["peak_vram_gb"]
                        * 100
                    )
                if baseline.get("quality_overall") and best.quality_score:
                    changes["quality_change"] = (
                        (best.quality_score.overall - baseline["quality_overall"])
                        / baseline["quality_overall"]
                        * 100
                        if baseline["quality_overall"]
                        else 0
                    )
                self.state.run.baseline_metrics["changes_vs_baseline"] = changes
            except Exception:
                pass

        # Explicit terminal state: SUCCESS / PARTIAL_SUCCESS / FAILED.
        # If speed floor enforced and not met, use a special status
        if enforce_floor and not speed_floor_met:
            self.state.run.status = RunStatus.COMPLETED_NO_WINNER
            self._log_event(
                "Run completed: no config meets speed floor",
                f"floor={speed_floor:.2f} tok/s enforced",
            )
        else:
            final_status = final_status_for_run(self.state.run)
            self.state.run.status = final_status

        # Keep COMPLETED readable in legacy UIs: stage is always COMPLETE here.
        self.state.run.stage = OptimizationStage.COMPLETE
        state_name, info = classify_run(list(self.state.tested_configs), self.state.run)
        self._log_event(
            f"Run {state_name}",
            f"successful={info['successful']} failed={info['failed']} "
            f"validated_winner={info['validated_winner']}",
        )
        run_repo.save(self.state.run)
        self._checkpoint(f"final:{state_name}")
        await self._broadcast({"stage": "complete", "event": "complete", "final": state_name})

        # --- Context strategies (spec §5) ---
        # Gated by the run's skip plan: granular modes (--reasoning-only,
        # --speed-only, --quality-only, --skip-phases context) must not run
        # extra context sweeps after the run (observed 2026-10-26: a
        # reasoning-only resume spent ~2x40 min on unintended strategies).
        _skips = run_skip_phases(self.state.run)
        if (
            self.state.best_config
            and self.state.run.context_strategy_modes
            and "context" not in _skips
        ):
            await self._run_context_strategies(self.state.best_config)
        elif self.state.run.context_strategy_modes and "context" in _skips:
            self._log_event(
                "Context strategies skipped",
                "context phase skipped by the run plan",
            )

        # --- Final sandbox (spec §6) ---
        if self.state.best_config and self.state.run.final_sandbox_mode != "off":
            await self._run_final_sandbox(self.state.best_config)

    def pause(self) -> None:
        """Pause optimization."""
        if self.state:
            self.state.should_pause = True

    def _pause_flag_path(self, run_id: Any = None) -> Any:
        """Pause-flag file for cross-process pause requests (same machine)."""
        try:
            from pathlib import Path as _Path

            from lm_optimizer.storage.run_checkpoint import checkpoint_dir

            rid = str(run_id or "")
            if not rid and self.state is not None and self.state.run is not None:
                rid = str(self.state.run.id)
            if not rid:
                return None
            return _Path(str(checkpoint_dir())) / f"pause_{rid}.flag"
        except Exception:
            return None

    def _should_pause(self) -> bool:
        """In-process flag or pause-flag file (exception-proof polling)."""
        try:
            if self.state is not None and self.state.should_pause:
                return True
            path = self._pause_flag_path()
            return bool(path is not None and path.exists())
        except Exception:
            return False

    async def _pause_point(self) -> None:
        """Safe-boundary pause check; raises PauseRequested when asked."""
        if self._should_pause():
            await self._pause_now()

    async def _pause_now(self, reason: str = "user_requested") -> None:
        """Finish-at-boundary pause: persist everything, unload, mark PAUSED."""
        from datetime import datetime as _dt

        run = self.state.run
        self._sync_phase_params()
        params = run.benchmark_params or {}
        pab = dict(params.get("phase_ab") or {})
        pab["pause"] = {
            "paused_at": _dt.now().isoformat(timespec="seconds"),
            "reason": reason,
            "phase": self.state.phase,
            "stage": run.stage.value,
            "current_config_id": str(self.state.current_config_id or ""),
            "recovery_cursor": dict(self.state.recovery_cursor or {}),
            "completed": len(self.state.tested_configs),
        }
        params["phase_ab"] = pab
        run.benchmark_params = params
        self._checkpoint(f"paused:{reason}")
        try:
            run_repo.save(run)
        except Exception as e:
            logger.error("Pause persist save failed", error=str(e))
        try:
            if not await self.client.ensure_unloaded(run.model.id):
                logger.warning("Model may still be loaded after pause", model=run.model.id)
        except Exception:
            pass
        run.status = RunStatus.PAUSED
        run.stage = OptimizationStage.PAUSED
        try:
            run_repo.save(run)
        except Exception as e:
            logger.error("Pause status save failed", error=str(e))
        try:
            await self._broadcast({"stage": "paused", "event": "paused"})
        except Exception:
            pass
        self.state.should_pause = False
        try:
            flag = self._pause_flag_path()
            if flag is not None and flag.exists():
                flag.unlink()
        except Exception:
            pass
        self._log_event("Paused", reason)
        raise PauseRequested(reason)

    def _sigint_pause_handler(self):
        """First SIGINT arms graceful pause; second falls through to abort."""

        def _handler(signum, frame):
            import signal as _signal

            if getattr(self, "_pause_second_signal", False):
                try:
                    _signal.signal(
                        _signal.SIGINT,
                        getattr(self, "_pause_prev_handler", _signal.default_int_handler),
                    )
                except Exception:
                    pass
                raise KeyboardInterrupt()
            self._pause_second_signal = True
            try:
                if self.state is not None:
                    self.state.should_pause = True
            except Exception:
                pass
            try:
                self._log_event("Pause requested (SIGINT)", "finishing current operation")
            except Exception:
                pass

        return _handler

    def _install_pause_handler(self) -> None:
        """Route first Ctrl+C to graceful pause (restored by caller)."""
        try:
            import signal as _signal

            self._pause_prev_handler = _signal.getsignal(_signal.SIGINT)
            self._pause_second_signal = False
            _signal.signal(_signal.SIGINT, self._sigint_pause_handler())
        except Exception as e:
            logger.error("Pause handler install failed", error=str(e))

    async def _return_paused(self, run: OptimizationRun) -> OptimizationRun:
        """Return path for graceful pause (state already persisted)."""
        try:
            run_repo.save(run)
        except Exception:
            pass
        await self._broadcast({"stage": "paused", "event": "paused"})
        return run

    def _restore_pause_handler(self) -> None:
        """Restore the pre-run SIGINT handler."""
        try:
            import signal as _signal

            prev = getattr(self, "_pause_prev_handler", None)
            if prev is not None:
                _signal.signal(_signal.SIGINT, prev)
        except Exception:
            pass
        try:
            self._pause_second_signal = False
        except Exception:
            pass

    def resume(self) -> None:
        """Resume optimization."""
        if self.state:
            self.state.should_pause = False

    def cancel(self) -> None:
        """Cancel optimization — checkpoint is written before ack by callers."""
        if self.state:
            self.state.should_cancel = True
            self.state.should_pause = False
            self._log_event("Cancel requested", "")
            self.checkpoint_now(reason="cancel-requested")

    async def resume_from_checkpoint(
        self,
        run_id: Any,
        revalidate: bool = False,
        revalidate_fingerprint: bool = False,
        fingerprint_diff_fields: list[str] | None = None,
        fingerprint_components: dict | None = None,
        style: str = "balanced",
    ) -> OptimizationRun:
        """Resume a run from its interrupt checkpoint + DB state.

        Completed candidates are NOT re-run unless:
        - revalidate=True (explicit re-run of all completed)
        - revalidate_fingerprint=True and fingerprint_diff indicates affected configs
        - their DB record is missing/corrupted
        - the model identity changed
        - the LM Studio config changed incompatibly (capability mismatch -> revalidate)
        """
        from lm_optimizer.storage.run_checkpoint import load_checkpoint

        run = run_repo.get(str(run_id))
        if not run:
            raise ValueError(f"Run not found: {run_id}")
        ckpt = load_checkpoint(run_id)
        if ckpt is None:
            raise ValueError(f"No checkpoint for run {run_id}")

        # Model identity guard.
        if ckpt.get("model") and ckpt["model"] != run.model.id:
            raise ValueError("Model identity changed since checkpoint; refuse resume")

        # Rebuild in-memory state from DB (source of truth for results).
        full = run_repo.get(str(run_id))
        assert full is not None
        # Restore the frozen settings before anything reads them (legacy runs
        # without the snapshot keep safe defaults).
        apply_persisted_run_settings(full)
        space = self._rebuild_resume_space(full, ckpt)
        # Restore model/hardware identity for scoring.
        model = full.model
        if not model.name:
            model = run.model
        try:
            hw = hardware_repo.get(full.hardware_id) if full.hardware_id else full.hardware
        except Exception:
            hw = full.hardware
        if hw is None:
            hw = full.hardware

        self.state = OptimizationState(run=full, search_space=space)
        self._restore_tested_state(full, ckpt)

        # Fingerprint-based revalidation: if fingerprint changed, revalidate affected configs
        if revalidate_fingerprint:
            from lm_optimizer.services import lms_cli as _lms_cli
            from lm_optimizer.benchmark.suite_version import BENCHMARK_SUITE_VERSION

            lms_version = "unknown"
            try:
                lms_version = _lms_cli.lms_version()
            except Exception:
                pass

            hw_fp = hardware_fingerprint(hw) if hw else "unknown"
            current_fp = fingerprint_components or {
                "model_id": full.model.id,
                "quantization": full.model.quantization,
                "architecture": full.model.architecture,
                "lm_studio_version": lms_version,
                "hardware_fingerprint": hw_fp,
                "benchmark_suite_version": BENCHMARK_SUITE_VERSION,
            }
            old_fp = full.benchmark_params.get("fingerprint_components", {})
            actual_diff = fingerprint_diff_fields or fingerprint_diff(old_fp, current_fp)
            if actual_diff:
                self._log_event(
                    "FINGERPRINT_REVALIDATE",
                    f"diff={actual_diff} revalidating affected configs",
                )
                keep, reval = filter_configs_for_revalidation(
                    self.state.tested_configs, actual_diff
                )
                self.state.tested_configs = keep
                self._log_event(
                    "FINGERPRINT_REVALIDATE",
                    f"kept={len(keep)} revalidate={len(reval)}",
                )
        else:
            # Backfill: a run created before fingerprinting has no stored
            # fingerprint. Adopt the current one so the NEXT auto-run can
            # compare honestly instead of treating absence as a mismatch.
            try:
                from lm_optimizer.benchmark.suite_version import BENCHMARK_SUITE_VERSION
                from lm_optimizer.services import lms_cli as _lms_cli

                try:
                    lms_version = _lms_cli.lms_version()
                except Exception:
                    lms_version = "unknown"
                comps = {
                    "model_id": full.model.id,
                    "quantization": full.model.quantization,
                    "architecture": full.model.architecture,
                    "lm_studio_version": lms_version,
                    "hardware_fingerprint": hardware_fingerprint(hw) if hw else "unknown",
                    "benchmark_suite_version": BENCHMARK_SUITE_VERSION,
                }
                if not (full.benchmark_params or {}).get("fingerprint"):
                    full.benchmark_params = dict(full.benchmark_params or {})
                    full.benchmark_params["fingerprint"] = compute_fingerprint(**comps)
                    full.benchmark_params["fingerprint_components"] = comps
                    self._log_event("FINGERPRINT_BACKFILL", f"fp={full.benchmark_params['fingerprint'][:16]}")
            except Exception as e:
                logger.debug("Fingerprint backfill skipped", error=str(e))

        # Reconcile phase/winner state if DB and checkpoint disagree
        self._reconcile_phase_and_winner(full, ckpt)
        # Resumed runs always continue under Phase A/B scoring rules.
        self.state.score_context = False
        # A resumed run is running again: drop any stale pause marker so
        # status surfaces and future checkpoints do not claim PAUSED.
        try:
            _params = self.state.run.benchmark_params or {}
            _pab = dict(_params.get("phase_ab") or {})
            if "pause" in _pab:
                del _pab["pause"]
                _params["phase_ab"] = _pab
                self.state.run.benchmark_params = _params
        except Exception:
            pass
        self.state.started_at = datetime.now()
        self.state.run.status = RunStatus.RESUMED
        self.state.run.stage = OptimizationStage.DISCOVERY
        self.benchmark.style = style
        run_repo.save(self.state.run)

        completed = len(self.state.tested_configs)
        remaining_est = max(0, space.estimate_size() - completed) if space else 0
        self._log_event(
            "Resuming optimization",
            f"completed={completed} remaining~{remaining_est}",
        )
        logger.info(
            "Resuming optimization",
            run_id=str(run_id),
            completed=completed,
            remaining_estimate=remaining_est,
        )
        if revalidate:
            logger.info("Revalidation requested: completed candidates will re-run")

        # If revalidate, drop tested cache so stages regenerate everything.
        if revalidate:
            self.state.tested_configs = []
            self.state.best_config = None
            self.state.failed_regions = set()
            self._reset_phase_for_revalidate()

        # A fresh resume consumes any stale pause flag so it cannot
        # instantly re-pause; Ctrl+C during resume still pauses gracefully.
        try:
            flag = self._pause_flag_path(run_id)
            if flag is not None and flag.exists():
                flag.unlink()
        except Exception:
            pass
        self._install_pause_handler()

        # Legacy runs (no phase_ab record in the persisted run) resume
        # through the legacy stage path; Phase A/B runs continue through
        # phases. Every path skips already tested candidate keys.
        # (The checkpoint FILE always carries a phase_ab block since the
        # builder is unconditional, so the DB run is the discriminator.)
        is_legacy = "phase_ab" not in (full.benchmark_params or {})
        if is_legacy:
            return await self._finish_resume(self._resume_legacy_pipeline())
        return await self._finish_resume(self._resume_phase_pipeline())

    async def _finish_resume(self, pipeline) -> OptimizationRun:
        """Run a resume pipeline with shared cancel/finalize handling."""
        try:
            await pipeline
        except PauseRequested:
            # Graceful pause: _pause_now already persisted everything.
            run_repo.save(self.state.run)
            return self.state.run
        except asyncio.CancelledError:
            self.state.run.status = RunStatus.CANCELLED
            self.state.run.stage = OptimizationStage.CANCELLED
            self.checkpoint_now(reason="cancelled-resumed")
            run_repo.save(self.state.run)
            raise
        finally:
            self._restore_pause_handler()
            self.state.run.completed_at = datetime.now()
            if self.state.run.started_at:
                self.state.run.duration_seconds = (
                    self.state.run.completed_at - self.state.run.started_at
                ).total_seconds()
            run_repo.save(self.state.run)
            self._checkpoint(f"final:{self.state.run.status.value}")
            if not await self.client.ensure_unloaded(self.state.run.model.id):
                logger.warning("Model may still be loaded after resume")
        return self.state.run

    @staticmethod
    def _rebuild_resume_space(full, ckpt: dict):
        """Search space object from the persisted run (resume scaffolding)."""
        from lm_optimizer.services.search_space import SearchSpace as _SS

        search_space_dict = full.search_space or ckpt.get("search_space", {})
        return _SS(
            context_lengths=search_space_dict.get("context_lengths", []),
            gpu_ratios=search_space_dict.get("gpu_ratios", []),
            flash_attention_options=search_space_dict.get("flash_attention_options", []),
            kv_cache_options=search_space_dict.get("kv_cache_options", []),
            batch_sizes=search_space_dict.get("batch_sizes", []),
            physical_batch_sizes=search_space_dict.get("physical_batch_sizes", []),
            parallels=search_space_dict.get("parallels", []),
            context_checkpoints_options=search_space_dict.get(
                "context_checkpoints_options", []
            ),
        )

    def _restore_tested_state(self, full, ckpt: dict) -> None:
        """Tested configs, best/pareto/failed regions, log and phase fields."""
        self.state.tested_configs = list(full.configurations)
        best_id = str(full.best_config_id) if full.best_config_id else None
        for c in self.state.tested_configs:
            if str(c.id) == best_id:
                self.state.best_config = c
        pareto_ids = {str(i) for i in (full.pareto_config_ids or [])}
        self.state.pareto_frontier = [
            c for c in self.state.tested_configs if str(c.id) in pareto_ids
        ]
        for c in self.state.tested_configs:
            if c.status in (ConfigurationStatus.FAILED, ConfigurationStatus.OOM):
                self.state.failed_regions.add(
                    (c.context_length, getattr(c.config, "gpu_ratio", None))
                )
        self.state.event_log = list(ckpt.get("event_log", []))
        self.state.errors = list(ckpt.get("errors", []))
        self.state.flash_required = bool(ckpt.get("flash_required", False))
        self._restore_phase_state(ckpt)
        # Reconcile recovery state: DB owns completed results, checkpoint
        # owns the cursor; never downgrade newer DB state with stale file
        # data (14b finding: resumed runs replayed recovery attempts).
        db_pab = ((full.benchmark_params or {}).get("phase_ab") or {})
        ckpt_pab = ((ckpt or {}).get("phase_ab") or {})
        if db_pab.get("recovery_log") or ckpt_pab.get("recovery_log"):
            self._reconcile_recovery_state(ckpt_pab, db_pab)

    def _restore_phase_state(self, ckpt: dict) -> None:
        """Restore Phase A/B resume fields (absent in legacy checkpoints)."""
        pab = (ckpt or {}).get("phase_ab") or {}
        self.state.phase = str(pab.get("phase") or "speed")
        self.state.run.phase = self.state.phase
        self.state.speed_context = pab.get("speed_context")
        self.state.quality_context = pab.get("quality_context")
        self.state.run.speed_context = self.state.speed_context
        self.state.run.quality_context = self.state.quality_context
        self.state.budget_class = pab.get("budget_class")
        try:
            self.state.probe_repetitions = max(1, int(pab.get("probe_repetitions", 1)))
        except (TypeError, ValueError):
            self.state.probe_repetitions = 1
        base_cfg = pab.get("baseline_config")
        if isinstance(base_cfg, dict) and base_cfg:
            try:
                cfg = LoadConfiguration(
                    **{k: v for k, v in base_cfg.items() if hasattr(LoadConfiguration, k)}
                )
                self.state.baseline_result = ConfigurationResult(
                    config=cfg,
                    context_length=int(
                        base_cfg.get("context_length") or self.state.speed_context or 2048
                    ),
                    status=ConfigurationStatus.SKIPPED,
                )
            except Exception:
                self.state.baseline_result = None
        self.state.speed_finalists = list(pab.get("speed_finalist_ids") or [])
        self.state.recovery_log = list(pab.get("recovery_log") or [])
        self.state.recovery_cursor = dict(pab.get("recovery_cursor") or {}) or None
        restored_attempts = pab.get("probe_attempts") or {}
        self.state.probe_attempts = (
            dict(restored_attempts) if isinstance(restored_attempts, dict) else {}
        )
        try:
            self.state.checkpoint_seq = int(pab.get("checkpoint_seq", 0) or 0)
        except (TypeError, ValueError):
            self.state.checkpoint_seq = 0
        self.state.excluded_notes = list(pab.get("excluded_notes") or [])
        self.state.phase_b_enabled = bool(pab.get("phase_b_enabled", False))
        self.state.run.phase_b_enabled = self.state.phase_b_enabled
        self.state.phase_b_result = pab.get("phase_b_result")
        self.state.run.phase_b_result = self.state.phase_b_result
        raw_id = pab.get("raw_fastest_id")
        if raw_id:
            for c in self.state.tested_configs:
                if str(c.id) == str(raw_id):
                    self.state.raw_fastest = c
                    self.state.run.raw_fastest_config_id = c.id
                    break

    def _reconcile_recovery_state(self, ckpt_pab: dict, db_pab: dict) -> dict:
        """Merge checkpoint + DB recovery state without downgrading.

        Source-of-truth rule: the DB row owns completed results, the
        checkpoint owns the execution cursor. For the recovery log both
        sides append monotonically, so the LONGER log wins; a stale
        (shorter/empty) checkpoint log never clears a newer DB log. The
        cursor keeps the greatest step seen on either side.
        Returns the merged phase_ab recovery section applied to state.
        """
        ckpt_pab = ckpt_pab or {}
        db_pab = db_pab or {}
        ckpt_log = list(ckpt_pab.get("recovery_log") or [])
        db_log = list(db_pab.get("recovery_log") or [])
        if len(db_log) >= len(ckpt_log):
            merged_log = db_log
            db_newer = len(db_log) > len(ckpt_log)
        else:
            merged_log = ckpt_log
            db_newer = False
        ckpt_step = (ckpt_pab.get("recovery_cursor") or {}).get("step", 0) or 0
        db_step = (db_pab.get("recovery_cursor") or {}).get("step", 0) or 0
        try:
            ckpt_step, db_step = int(ckpt_step), int(db_step)
        except (TypeError, ValueError):
            ckpt_step, db_step = 0, 0
        merged_cursor = dict(
            (db_pab.get("recovery_cursor") or {})
            if db_step >= ckpt_step
            else (ckpt_pab.get("recovery_cursor") or {})
        )
        ckpt_seq = ckpt_pab.get("checkpoint_seq", 0) or 0
        db_seq = db_pab.get("checkpoint_seq", 0) or 0
        self.state.recovery_log = merged_log
        self.state.recovery_cursor = merged_cursor or None
        try:
            self.state.checkpoint_seq = max(int(ckpt_seq), int(db_seq))
        except (TypeError, ValueError):
            pass
        self._log_event(
            "RESUME_RECONCILE",
            f"db_newer={db_newer} checkpoint_newer={not db_newer and len(ckpt_log) > len(db_log)} "
            f"log_len={len(merged_log)} cursor_step={merged_cursor.get('step', 0)}",
        )
        try:
            run_repo.save(self.state.run)
        except Exception:
            pass
        return {"recovery_log": merged_log, "recovery_cursor": merged_cursor}

    def _reconcile_phase_and_winner(self, full: Any, ckpt: dict) -> None:
        """Reconcile phase progression and winner state between DB and checkpoint.

        Authority rules:
        - Completed candidate results: DB is authoritative (source of truth)
        - Current phase/search state: checkpoint is authoritative for in-progress work
        - Winner selection: if DB has eligible quality-passed configs but no best_config_id,
          reconstruct winner from DB completed work.
        - If checkpoint phase is ahead of DB stage, trust checkpoint (in-progress phase started).
        - If DB stage is ahead of checkpoint phase, trust DB completed work and reconstruct.
        """
        try:
            db_pab = ((full.benchmark_params or {}).get("phase_ab") or {})
            ckpt_pab = ((ckpt or {}).get("phase_ab") or {})

            db_phase = db_pab.get("phase", "speed")
            ckpt_phase = ckpt_pab.get("phase", "speed")
            db_stage = getattr(full.stage, "value", str(full.stage))
            ckpt_stage = ckpt.get("stage", "speed_discovery")

            # Phase order for comparison
            phase_order = {
                "speed": 0,
                "quality": 1,
                "recovery": 2,
                "final_validation": 3,
                "context_optional": 4,
            }
            db_phase_ord = phase_order.get(db_phase, 0)
            ckpt_phase_ord = phase_order.get(ckpt_phase, 0)

            # Log reconciliation
            self._log_event(
                "PHASE_RECONCILE",
                f"db_phase={db_phase} ckpt_phase={ckpt_phase} db_stage={db_stage} ckpt_stage={ckpt_stage}",
            )

            # If DB has completed quality work but no winner, reconstruct from DB
            if (full.best_config_id is None
                and db_phase_ord >= phase_order["quality"]
                and self.state.tested_configs):

                # Find quality-passed configs at quality_context
                quality_ctx = self.state.quality_context or 8192
                safe_candidates = []
                for c in self.state.tested_configs:
                    if c.context_length != quality_ctx:
                        continue
                    if c.status != ConfigurationStatus.PASSED:
                        continue
                    gen = c.generation or {}
                    # Quality completed marker or backfill
                    if gen.get("quality_completed") is True:
                        safe_candidates.append(c)
                    elif gen.get("quality_by_test") and c.quality_score:
                        # Backfill for older records
                        safe_candidates.append(c)

                if safe_candidates:
                    # Select fastest quality-passing config (Phase A rule R14)
                    fastest = max(safe_candidates, key=lambda r: r.get_avg_generation_tok_s())
                    self.state.best_config = fastest
                    self.state.run.best_config_id = fastest.id
                    self._log_event(
                        "WINNER_RECONSTRUCTED",
                        f"config={str(fastest.id)[:8]} speed={fastest.get_avg_generation_tok_s():.1f} tok/s",
                    )
                    try:
                        run_repo.save(self.state.run)
                    except Exception as e:
                        logger.error("Winner reconstruction save failed", error=str(e))

            # Sync phase from the more advanced source
            if ckpt_phase_ord > db_phase_ord:
                # Checkpoint has started a newer phase; adopt it
                self.state.phase = ckpt_phase
                self.state.run.phase = ckpt_phase
                self._log_event("PHASE_ADOPTED_FROM_CKPT", f"phase={ckpt_phase}")
            elif db_phase_ord > ckpt_phase_ord:
                # DB has advanced further (completed work); keep DB phase but note mismatch
                self.state.phase = db_phase
                self.state.run.phase = db_phase
                self._log_event("PHASE_ADOPTED_FROM_DB", f"phase={db_phase}")

        except Exception as e:
            logger.error("Phase/winner reconciliation failed", error=str(e))
            # Non-fatal: resume will proceed with restored state

    async def _resume_legacy_pipeline(self) -> None:
        """Legacy stage sequence for pre-Phase-A/B runs."""
        await self._stage_coarse_search()
        if self.state.should_cancel:
            raise asyncio.CancelledError()
        await self._stage_refinement()
        if self.state.should_cancel:
            raise asyncio.CancelledError()
        await self._stage_batch_optimization()
        if self.state.should_cancel:
            raise asyncio.CancelledError()
        await self._stage_micro_refinement()
        await self._stage_validation()
        self._recompute_all_scores()
        await self._finalize()

    async def _resume_phase_pipeline(self) -> None:
        """Phase continuation for Phase A/B runs (all steps skip tested)."""
        skips = run_skip_phases(self.state.run)
        if skips:
            self._log_event(
                "Phase skips active", f"skipped: {', '.join(sorted(skips))}"
            )
        if "speed" in skips:
            self._log_event("Phase SPEED skipped", "--skip-phases")
            frontier = granular_quality_frontier(
                skips, self.state.run.advanced_settings, self.state.baseline_result
            )
            if frontier["finalists"]:
                self._log_event(
                    "QUALITY candidates from baseline",
                    "granular mode without speed phase: baseline config only",
                )
        else:
            frontier = await self._phase_speed()
        if self.state.should_cancel:
            raise asyncio.CancelledError()
        # Prefer checkpoint-recorded finalists that are still tested.
        tested_ids = {str(c.id) for c in self.state.tested_configs}
        kept = [i for i in self.state.speed_finalists if i in tested_ids]
        if kept and frontier.get("winner") is not None:
            by_id = {str(c.id): c for c in self.state.tested_configs}
            frontier = {
                "winner": frontier["winner"],
                "finalists": [by_id[i] for i in kept if i in by_id],
                "contrarian": frontier.get("contrarian"),
            }
        if "quality" in skips:
            self._log_event("Phase QUALITY skipped", "--skip-phases (no winner path)")
            safe, failed = [], []
        else:
            safe, failed = await self._phase_quality(frontier)
        if self.state.should_cancel:
            raise asyncio.CancelledError()
        if "recovery" in skips:
            self._log_event("Phase RECOVERY skipped", "--skip-phases")
            all_safe = [r for r in safe if r.status == ConfigurationStatus.PASSED]
            if all_safe:
                fastest = max(all_safe, key=lambda r: r.get_avg_generation_tok_s())
                self.state.best_config = fastest
                self.state.run.best_config_id = fastest.id
        else:
            await self._resume_recover(safe, failed)
        if self.state.should_cancel:
            raise asyncio.CancelledError()
        await self._pause_point()
        if "validation" in skips:
            self._log_event("Phase FINAL_VALIDATION skipped", "--skip-phases")
        else:
            await self._phase_final_validation()
        if self.state.should_cancel:
            raise asyncio.CancelledError()
        await self._pause_point()
        if "context" in skips and self.state.phase_b_enabled:
            self._log_event("Phase CONTEXT skipped", "--skip-phases (Phase B off)")
            self.state.phase_b_enabled = False
            self.state.run.phase_b_enabled = False
        if self.state.phase_b_enabled and self.state.best_config is not None:
            await self._phase_context_optional(self.state.best_config)
        self._recompute_all_scores()
        await self._finalize()

    def _recovery_reference(self, safe: list) -> tuple:
        """Rollback reference + kind: fastest safe, else S0 anchor, else none.

        Kinds: "safe" (quality-verified) | "baseline" (measured anchor only,
        NOT quality-verified) | "none".
        """
        if safe:
            return max(safe, key=lambda r: r.get_avg_generation_tok_s()).config, "safe"
        if self.state.baseline_result is not None:
            return self.state.baseline_result.config, "baseline"
        return None, "none"

    def _reset_phase_for_revalidate(self) -> None:
        """Clear Phase A/B progress so revalidate truly re-runs everything."""
        self.state.speed_finalists = []
        self.state.raw_fastest = None
        self.state.run.raw_fastest_config_id = None
        self.state.recovery_log = []
        self.state.excluded_notes = []
        self.state.budget_class = None
        self.state.probe_repetitions = 1
        self.state.phase_b_result = None
        self.state.run.phase_b_result = None
        self.state.baseline_result = None

    async def _resume_recover(self, safe: list, failed: list) -> None:
        """Recover failed finalists skipped by an earlier attempt log."""
        ref_config, ref_kind = self._recovery_reference(safe)
        recovered_ids = {
            str(e.get("config_id"))
            for e in self.state.recovery_log
            if isinstance(e, dict) and e.get("config_id")
        }
        if ref_config is not None:
            for failed_result in failed:
                if self.state.should_cancel:
                    break
                await self._pause_point()
                if str(failed_result.id) in recovered_ids:
                    continue
                rec, _culprit = await self._run_recovery(failed_result, ref_config, ref_kind)
                if rec is not None:
                    safe.append(rec)
                    ref_config = rec.config
                    ref_kind = "safe"
        all_safe = [r for r in safe if r.status == ConfigurationStatus.PASSED]
        if all_safe:
            fastest = max(all_safe, key=lambda r: r.get_avg_generation_tok_s())
            self.state.best_config = fastest
            self.state.run.best_config_id = fastest.id

    # ============================================================
    # Context Strategies (spec §5)
    # ============================================================

    async def _run_context_strategies(self, winner: ConfigurationResult) -> None:
        """Run multiple context strategies on the winner config (spec §5.1).

        Each strategy produces a separate configuration/result that is recorded
        independently. The user can then choose which strategy's recommendation
        to prefer.
        """
        from lm_optimizer.services.context_sweep import sweep, geometric_contexts
        from lm_optimizer.services.context_explorer import MIN_DECODE_TPS

        strategies = getattr(self.state.run, "context_strategy_modes", [])
        if not strategies:
            return

        model = self.state.run.model
        model_limit = getattr(model, "context_limit", 0) or 65536
        min_ctx = getattr(self.state.run, "min_context", 2048) or 2048
        max_ctx = getattr(self.state.run, "max_context", model_limit) or model_limit
        min_tps = getattr(self.state.run, "min_generation_tok_s", 2.0)

        base_config = winner.config
        results: dict[str, dict] = {}

        for strat_name in strategies:
            if self.state.should_cancel:
                break
            if strat_name not in VALID_CONTEXT_STRATEGIES:
                self._log_event("Context strategy skipped", f"unknown: {strat_name}")
                continue

            self._log_event("Context strategy started", f"strategy={strat_name}")
            await self._pause_point()

            # Build strategy-specific config
            strat_config = await self._build_strategy_config(base_config, strat_name, model)
            if strat_config is None:
                self._log_event("Context strategy unavailable", f"{strat_name} (capability gate)")
                results[strat_name] = {"available": False, "reason": "capability not supported"}
                continue

            # Probe function for this strategy
            async def _probe_at(ctx: int, _cfg=strat_config) -> dict:
                try:
                    res = await self._test_config(_cfg, ctx, min_speed=min_tps)
                    if res is None or res.status != ConfigurationStatus.PASSED:
                        return {"ok": False, "tok_s": 0.0, "error": res.error if res else "load failed"}
                    return {
                        "ok": True,
                        "tok_s": res.get_avg_generation_tok_s(),
                        "retrieval_passed": getattr(res, "retrieval_passed", False),
                    }
                except Exception as e:
                    return {"ok": False, "tok_s": 0.0, "error": str(e)[:160]}

            try:
                sweep_result = await sweep(
                    _probe_at,
                    min_ctx=min_ctx,
                    max_ctx=max_ctx,
                    min_tps=min_tps,
                )
                results[strat_name] = {
                    "available": True,
                    "config": strat_config.to_dict(),
                    **sweep_result,
                }
                self._log_event(
                    "Context strategy complete",
                    f"{strat_name}: max_stable={sweep_result.get('max_stable_context')}, "
                    f"max_usable={sweep_result.get('max_usable_context')}",
                )
            except Exception as e:
                logger.error("Context strategy sweep failed", strategy=strat_name, error=str(e))
                results[strat_name] = {"available": False, "reason": f"sweep failed: {e}"}

        # Store all strategy results
        self.state.run.phase_b_result = {
            **(self.state.run.phase_b_result or {}),
            "context_strategies": results,
        }
        self._sync_phase_params()
        run_repo.save(self.state.run)
        self._checkpoint("phase:context_strategies:complete")
        await self._broadcast({"stage": "context_strategies", "event": "context_strategies_complete"})

    async def _build_strategy_config(
        self, base: LoadConfiguration, strategy: str, model: ModelIdentity
    ) -> LoadConfiguration | None:
        """Build a strategy-specific load configuration (spec §5.1).

        Gates modes through live LM Studio capability registry.
        """
        from lm_optimizer.services.lm_studio import LMStudioClient

        # Check capabilities
        caps = getattr(self.client, "_capabilities", None)
        if caps is None:
            # Best effort - assume supported
            pass

        data = {k: v for k, v in base.to_dict().items()}

        if strategy == "winner":
            # Frozen runtime from Phase A - no changes
            pass

        elif strategy == "gpu_resident":
            # GPU-resident weights + KV on GPU
            data["offload_kv_cache_to_gpu"] = True
            data["gpu_ratio"] = 1.0
            # Verify capability
            if caps and not caps.supports_kv_cache_placement:
                return None

        elif strategy == "model_first_kv_cpu":
            # Model on GPU, KV on CPU
            data["offload_kv_cache_to_gpu"] = False
            # Keep model layers on GPU as much as possible
            if data.get("gpu_ratio") is None or data.get("gpu_ratio", 0) < 1.0:
                data["gpu_ratio"] = 1.0
            if caps and not caps.supports_kv_cache_placement:
                return None

        elif strategy == "system_spillover":
            # Partial GPU offload requires the verified CLI channel (REST
            # cannot set gpu_ratio).
            try:
                from lm_optimizer.services import lms_cli as _lms_cli

                cli_channel = bool(
                    getattr(self.benchmark, "gpu_via_cli", False)
                    and _lms_cli.lms_available()
                )
            except Exception:
                cli_channel = False
            if not cli_channel:
                return None
            data["gpu_ratio"] = 0.5
            data["offload_kv_cache_to_gpu"] = False

        # Create LoadConfiguration with only valid fields
        valid_fields = {k: v for k, v in data.items() if hasattr(LoadConfiguration, k) and v is not None}
        return LoadConfiguration(**valid_fields)

    # ============================================================
    # Final Sandbox (spec §6)
    # ============================================================

    async def _run_final_sandbox(self, winner: ConfigurationResult) -> None:
        """Run sandbox tasks against the actual optimizer winner config (spec §6)."""
        from lm_optimizer.services.sandbox_lab import (
            SandboxTask,
            BUILTIN_TASKS,
            get_task_by_id,
            validate_custom_prompt,
        )
        from lm_optimizer.services.sandbox_runner import run_manual_prompt, run_sandbox_task

        mode = self.state.run.final_sandbox_mode
        preset_ids = self.state.run.final_sandbox_preset_ids
        custom_prompts = self.state.run.final_sandbox_custom_prompts
        compare_baseline = self.state.run.sandbox_compare_baseline

        if mode == "off":
            return

        tasks_to_run: list[SandboxTask] = []
        invalid_custom: list[dict] = []
        manual_specs: dict[str, dict] = {}

        if mode == "one":
            for pid in preset_ids[:1]:
                task = get_task_by_id(pid)
                if task is None:
                    self._log_event("Final sandbox: unknown preset", str(pid))
                else:
                    tasks_to_run.append(task)
        elif mode == "all":
            # "all" means every registered compatible built-in (single source
            # of truth: SANDBOX_TASKS / BUILTIN_TASKS).
            tasks_to_run.extend(BUILTIN_TASKS.values())
        elif mode == "custom":
            for raw in custom_prompts:
                try:
                    spec = validate_custom_prompt(raw)
                except ValueError as e:
                    invalid_custom.append({"spec": raw, "error": str(e)})
                    continue
                if spec["manual_review_only"]:
                    # Handled by the generation-only path below; no score.
                    manual_specs[spec["id"]] = spec
                    tasks_to_run.append(
                        SandboxTask(
                            id=spec["id"],
                            kind=spec["task_kind"],
                            title=spec["name"],
                            prompt=spec["prompt"],
                            checks=[],
                            max_repair_attempts=0,
                            timeout_s=spec["timeout_s"],
                        )
                    )
                else:
                    tasks_to_run.append(
                        SandboxTask(
                            id=spec["id"],
                            kind=spec["task_kind"],
                            title=spec["name"],
                            prompt=spec["prompt"],
                            checks=spec["checks"],
                            max_repair_attempts=0,
                            timeout_s=spec["timeout_s"],
                        )
                    )

        if not tasks_to_run:
            self._log_event("Final sandbox skipped", "no valid tasks to run")
            self.state.run.benchmark_params["final_sandbox"] = {
                "mode": mode,
                "tasks": [],
                "invalid_custom_prompts": invalid_custom,
            }
            run_repo.save(self.state.run)
            return

        self._log_event("Final sandbox started", f"mode={mode} tasks={len(tasks_to_run)}")
        await self._pause_point()

        model_id = self.state.run.model.id
        winner_config = winner.config

        sandbox_results: list[dict] = []

        async def _run_one(cfg, task: SandboxTask, slot: str) -> None:
            if task.kind.value == "text" and not task.checks:
                spec = manual_specs.get(task.id) or {}
                result = await run_manual_prompt(
                    self.client,
                    model_id,
                    cfg,
                    task.prompt,
                    run_id=str(self.state.run.id),
                    max_output_tokens=spec.get("max_output_tokens", 2048),
                    temperature=spec.get("temperature", 0.2),
                    timeout_s=task.timeout_s,
                )
            else:
                result = await run_sandbox_task(
                    self.client,
                    model_id,
                    cfg,
                    task,
                    run_id=str(self.state.run.id),
                )
            sandbox_results.append(
                {
                    "task_id": task.id,
                    "task_kind": task.kind.value,
                    "config": slot,
                    "result": result,
                }
            )

        for task in tasks_to_run:
            if self.state.should_cancel:
                break
            self._log_event("Sandbox task", f"{task.id} ({task.kind.value})")
            try:
                await _run_one(winner_config, task, "winner")
            except Exception as e:
                logger.error("Sandbox task failed", task=task.id, error=str(e))
                sandbox_results.append(
                    {
                        "task_id": task.id,
                        "task_kind": task.kind.value,
                        "config": "winner",
                        "error": str(e)[:200],
                    }
                )

        # Optional baseline comparison (separate A/B, never mixed into scores).
        if compare_baseline and self.state.run.baseline_config_id:
            baseline_cfg = self.state.run.get_baseline_config()
            if baseline_cfg:
                self._log_event("Sandbox baseline comparison", "running")
                for task in tasks_to_run:
                    if self.state.should_cancel:
                        break
                    try:
                        await _run_one(baseline_cfg.config, task, "baseline")
                    except Exception as e:
                        sandbox_results.append(
                            {
                                "task_id": task.id,
                                "task_kind": task.kind.value,
                                "config": "baseline",
                                "error": str(e)[:200],
                            }
                        )

        # Store sandbox results (kept OUT of optimizer scores/Pareto).
        self.state.run.benchmark_params["final_sandbox"] = {
            "mode": mode,
            "tasks": sandbox_results,
            "invalid_custom_prompts": invalid_custom,
            "winner_config_id": str(winner.id),
            "winner_config": winner_config.to_dict(),
            "affects_optimizer_score": False,
        }
        self._sync_phase_params()
        run_repo.save(self.state.run)
        self._checkpoint("phase:final_sandbox:complete")
        await self._broadcast({"stage": "final_sandbox", "event": "final_sandbox_complete"})


# Type alias for LoadConfig
LoadConfig = LoadConfiguration
