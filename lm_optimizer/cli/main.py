"""CLI commands for LM Studio Optimizer."""

import asyncio
import dataclasses
import os
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from lm_optimizer.config import config
from lm_optimizer.database.repositories import (
    capability_repo,
    preset_repo,
    run_repo,
)
from lm_optimizer.domain.models import (
    ConfigurationStatus,
    LoadConfiguration,
    OptimizationProfile,
    OptimizationStage,
)
from lm_optimizer.logging_config import get_logger, setup_logging
from lm_optimizer.services.benchmark import VALID_STYLES, BenchmarkService
from lm_optimizer.services.context_sweep import format_report as _ctx_format
from lm_optimizer.services.context_sweep import geometric_contexts as _ctx_ladder
from lm_optimizer.services.context_sweep import sweep as _ctx_sweep
from lm_optimizer.services.hardware import hardware_detector
from lm_optimizer.services.fit import ladder as ctx_ladder
from lm_optimizer.services.generation_defaults import lookup as generation_lookup
from lm_optimizer.services.hostguard import (
    HostBusyTimeout,
    HostUnreachableError,
    format_snapshot,
    is_connection_error,
    prepare_host,
)
from lm_optimizer.services.joblock import JobBusyError, acquire_lock, release_lock
from lm_optimizer.services.matrix import build_matrix, recommendation_line, run_one
from lm_optimizer.services.model_recommendations import (
    diagnose_load_error,
    expert_advice,
    manual_checklist,
    recommend_for_vram,
    threads_advice,
)
from lm_optimizer.services.reporting import save_best_report, save_failed_report, save_fit_report
from lm_optimizer.services.unload_guard import UnloadNotClean, assert_unloaded
from lm_optimizer.services.optimizer import AdaptiveOptimizer
from lm_optimizer.services.quality import QualityConfig, QualityEvaluator
from lm_optimizer.services.search_space import SearchSpaceGenerator

app = typer.Typer(
    name="lm-optimizer",
    help="LM Studio Auto Optimizer - Automatically discover the best inference configuration",
    add_completion=False,
)

# Lab workflows (Context Explorer, Quality Lab, Recommender, Duel) live in
# their own module but are exposed under the same CLI.
from lm_optimizer.cli import lab as _lab  # noqa: E402

app.add_typer(_lab.app, name="lab")

console = Console()
logger = get_logger(__name__)

if TYPE_CHECKING:
    from lm_optimizer.backends.ollama.client import OllamaClient
    from lm_optimizer.services.lm_studio import LMStudioClient


def _lm_studio_client(base_url: str | None = None) -> "LMStudioClient":
    """Construct an LM Studio client (legacy behavior: validate + build)."""
    from lm_optimizer.services.lm_studio import LMStudioClient

    # Validate URL if provided
    if base_url:
        base_url = base_url.strip().rstrip("/")
        if not (base_url.startswith("http://") or base_url.startswith("https://")):
            console.print(f"[red]Invalid URL: {base_url}[/red]")
            console.print("URL must start with http:// or https://")
            console.print("Example: http://127.0.0.1:1234 or http://192.168.1.100:1234")
            sys.exit(1)
    client = LMStudioClient(base_url=base_url)
    return client


def get_client(base_url: str | None = None) -> "LMStudioClient | OllamaClient":
    """Create backend client, optionally overriding URL (without persisting).

    Default backend is lm-studio; that path is byte-identical to the legacy
    behavior. Select another backend via config or the ``--backend`` option.
    """
    if (config.backend or "lm-studio") != "lm-studio":
        return get_backend_client(config.backend, base_url)
    return _lm_studio_client(base_url)


_VALID_BACKENDS = ("lm-studio", "ollama", "llama-cpp")


def _require_backend(backend: str) -> str:
    """Validate a backend name, exiting with usage hint on error."""
    normalized = (backend or "").strip().lower()
    if normalized not in _VALID_BACKENDS:
        console.print(f"[red]Invalid --backend: {backend}[/red]")
        console.print(f"Expected one of: {', '.join(_VALID_BACKENDS)}")
        sys.exit(2)
    return normalized


def get_backend_client(
    backend: str | None = None, base_url: str | None = None
) -> "LMStudioClient | OllamaClient":
    """Create a backend client by name (default: configured backend, lm-studio).

    The ``lm-studio`` path uses the legacy construction byte-for-byte.
    An explicit ``backend`` argument always wins over global config.
    ``llama-cpp`` is phase 2 and raises explicitly.
    """
    from lm_optimizer.backends.ollama.client import OllamaClient

    resolved = _require_backend(backend or config.backend or "lm-studio")
    if resolved == "lm-studio":
        return _lm_studio_client(base_url=base_url)
    if resolved == "ollama":
        return OllamaClient(base_url=base_url or config.ollama_base_url)
    raise NotImplementedError("llama.cpp backend is phase 2")


def _handle_connection_error(url: str, error: Exception) -> None:
    """User-friendly connection error without exposing secrets."""
    console.print("\n[red]Cannot connect to LM Studio at:[/red]")
    console.print(f"  {url}")
    console.print("\n[bold]Check:[/bold]")
    console.print("  - LM Studio is running")
    console.print("  - Developer API is enabled (LM Studio -> Settings -> Developer)")
    console.print("  - URL/port is correct")
    console.print("  - Network access is allowed (firewall/VPN)")
    console.print(f"\n[dim]Details: {type(error).__name__}[/dim]")
    # Do not log full URL with credentials or stack trace containing secrets


def _acquire_run_lock(purpose: str) -> None:
    """Claim the cross-process optimizer job lock for this run.

    A second concurrent job is refused with holder info (exit 1); the
    existing lock file is never disturbed. Callers release in `finally`.
    """
    try:
        acquire_lock(purpose)
    except JobBusyError as e:
        holder = e.holder or {}
        console.print("[red]Another optimizer job is already running:[/red]")
        console.print(f"  pid {holder.get('pid')} purpose {holder.get('purpose')!r}")
        console.print("Run one job at a time; retry when it finishes.")
        sys.exit(1)


def _client_base_url(client) -> str:
    """Backend URL for messages (never raises)."""
    return getattr(client, "base_url", None) or config.lm_studio.base_url


def _busy_blockers(exc: HostBusyTimeout) -> str:
    """Blocker model ids for messages (never empty)."""
    return ", ".join(str(b) for b in (exc.blocker or [])) or "unknown model"


def _print_host_busy_abort(exc: HostBusyTimeout) -> None:
    """Clear abort message for a persistently busy host."""
    console.print(f"[red]Host busy: {_busy_blockers(exc)} still loaded; aborting.[/red]")
    console.print(f"  {exc}")
    console.print("Unload the model in LM Studio (disable Keep Model in Memory) and retry.")


def _print_host_busy_paused(model: str, exc: HostBusyTimeout) -> None:
    """Pause-framed busy message for optimize, with a resume hint.

    Nothing started (no run id, no checkpoint), so "pause" here means the
    pre-start wait budget ran out: free the host and re-run; in-progress
    runs resume via the normal checkpoint flow.
    """
    console.print(
        f"[yellow]Host busy: {_busy_blockers(exc)} still loaded "
        "- pausing before start.[/yellow]"
    )
    console.print(f"  {exc}")
    console.print("Nothing was started, so no checkpoint was needed.")
    console.print("Unload the model in LM Studio (disable Keep Model in Memory),")
    console.print(f"then re-run: lm-optimizer optimize {model}")
    console.print("In-progress runs resume with: lm-optimizer resume <run-id>")
    console.print("  (list them with: lm-optimizer checkpoints)")


async def _connect_guarded(
    client, purpose: str, wait_interval_s: float = 60.0, max_waits: int = 30
):
    """Connect + exclusive-access guard with typed failures.

    Raises HostUnreachableError when the backend cannot be reached (caller
    shows the friendly connection message) or HostBusyTimeout when a foreign
    model never leaves (caller pauses or aborts). Other errors propagate.
    """
    try:
        await client.connect()
    except Exception as e:
        if is_connection_error(e):
            raise HostUnreachableError(
                f"Cannot reach backend at {_client_base_url(client)}", cause=e
            ) from e
        raise
    return await prepare_host(
        client,
        purpose=purpose,
        wait_interval_s=wait_interval_s,
        max_waits=max_waits,
    )


async def _connect_guarded_or_abort(
    client, purpose: str, wait_interval_s: float = 60.0, max_waits: int = 30
):
    """_connect_guarded that aborts (exit 1) with a clear message.

    Shared by every command except optimize (which pauses with a resume
    hint instead). Unreachable shows the friendly connection message, busy
    shows the abort message; never the twain.
    """
    try:
        return await _connect_guarded(client, purpose, wait_interval_s, max_waits)
    except HostBusyTimeout as e:
        _print_host_busy_abort(e)
        sys.exit(1)
    except HostUnreachableError as e:
        _handle_connection_error(_client_base_url(client), e.cause or e)
        sys.exit(1)


def _require_style(style: str) -> str:
    """Validate benchmark style, exiting with usage hint on error."""
    if style not in VALID_STYLES:
        console.print(f"[red]Invalid style: {style}[/red]")
        console.print(f"Expected one of: {', '.join(VALID_STYLES)}")
        console.print("  precise: code/math (deterministic temperatures)")
        console.print("  balanced: suite defaults")
        console.print("  creative: summarization (higher temperature)")
        sys.exit(2)
    return style


def _benchmark_config_obj(repetitions: int, max_tokens_scale: float = 1.0, warmup_runs: int = 1):
    """Duck-typed benchmark config (repetitions/warmup/timeouts/scale)."""
    try:
        scale = min(1.0, max(0.25, float(max_tokens_scale)))
    except (TypeError, ValueError):
        scale = 1.0
    try:
        warmups = int(warmup_runs)
    except (TypeError, ValueError):
        warmups = 1
    warmups = min(5, max(0, warmups))
    return type(
        "obj",
        (object,),
        {
            "repetitions": repetitions,
            "warmup_repetitions": warmups,
            "timeout_seconds": 300,
            "load_timeout_seconds": 60,
            "generation_timeout_seconds": 120,
            "max_tokens_scale": scale,
        },
    )()


async def _run_optimization(
    client,
    model_info,
    hardware,
    profile_name: str,
    quality_threshold: float,
    repetitions: int,
    style: str,
    advanced: dict,
    max_tokens_scale: float = 1.0,
):
    """Shared optimize flow for the optimize/auto commands. Returns the run."""
    from lm_optimizer.services import lms_cli as _lms

    gpu_via_cli = bool(advanced.get("gpu_via_cli", False)) and _lms.lms_available()
    if advanced.get("gpu_via_cli") and not _lms.lms_available():
        console.print(
            "[yellow]lms CLI not found: GPU offload stays REST-only "
            "(ratio recorded, not applied).[/yellow]"
        )
    if max_tokens_scale < 1.0:
        console.print(
            "[yellow]max-tokens scaled to "
            f"{max_tokens_scale}: faster, but truncation may fail quality gates.[/yellow]"
        )
    benchmark_service = BenchmarkService(
        client,
        benchmark_config=_benchmark_config_obj(
            repetitions, max_tokens_scale, advanced.get("warmup_runs", 1)
        ),
        style=style,
        gpu_via_cli=gpu_via_cli,
        include_categories=advanced.get("include_categories"),
        test_only=advanced.get("test_only", False),
        speed_only=advanced.get("speed_only", False),
        quality_only=advanced.get("quality_only", False),
        speed_context=advanced.get("speed_context"),
        mtp_test=advanced.get("mtp_test", False),
        reasoning_sweep=advanced.get("reasoning_sweep", False),
        quality_lab=advanced.get("quality_lab", False),
        test_names=advanced.get("test_names") or None,
        min_tps=advanced.get("min_tps"),
    )
    quality_evaluator = QualityEvaluator(QualityConfig(minimum_score=quality_threshold))
    search_generator = SearchSpaceGenerator(client)
    optimizer = AdaptiveOptimizer(client, benchmark_service, quality_evaluator, search_generator)
    return await optimizer.optimize(
        model_info,
        hardware,
        OptimizationProfile(profile_name),
        quality_threshold=quality_threshold,
        advanced_settings=advanced,
        style=style,
        progress_cb=advanced.pop("progress_cb", None),
    )


def _host_info_dict(hardware, note: str | None = None) -> dict:
    """Host facts for reports (KV quant is a GUI preset, REST cannot read it)."""
    try:
        from lm_optimizer.services.hostguard import storage_info as _storage_info

        _storage = _storage_info()
    except Exception:
        _storage = {}
    info = {
        "gpu": ", ".join(f"{g.name} {g.vram_gb:.1f}GB" for g in hardware.gpus) or "none (CPU-only)",
        "cpu": f"{hardware.cpu_name} "
        f"({hardware.cpu_cores_physical}P/{hardware.cpu_cores_logical}L)",
        "threads": f"{hardware.cpu_cores_logical} recommended (REST cannot set n_threads)",
        "ram": f"{hardware.total_ram_gb:.1f}GB",
        "kv_quant": "Q4 starter (GUI preset, REST cannot set)",
        "storage": _storage,
    }
    try:
        ver = _version_info()
        info["version"] = ver["lms"]
        info["backend"] = ver["backend"]
    except Exception:
        pass
    if note:
        info["note"] = note
    return info


@lru_cache(maxsize=1)
def _version_info() -> dict:
    """LM Studio version + backend (one subprocess survey per process)."""
    from lm_optimizer.services import lms_cli

    backend = "unknown"
    try:
        backend = lms_cli.runtime_survey().get("engine") or "unknown"
    except Exception:
        pass
    return {"lms": lms_cli.lms_version(), "backend": backend}


def _warn_if_remote(base_url: str) -> str | None:
    """Warn when LM Studio is remote: local hardware does NOT score it.

    Returns a note for reports, or None when local.
    """
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except Exception:
        host = ""
    if host in ("127.0.0.1", "localhost", "::1", ""):
        return None
    console.print(
        "[bold yellow]Remote LM Studio detected:[/bold yellow] hardware below is the "
        "LOCAL CONTROLLER, not the server. Set HW_ overrides (.env) or treat as Unknown."
    )
    return "REMOTE server: hardware in this report is the local controller, not the inference host."


def _generation_for(model_id: str, architecture: str | None = None) -> dict:
    """Sourced generation settings dict for .md reports (settings + source)."""
    found = generation_lookup(model_id, architecture)
    return {**found["settings"], "source": found["source"]}


def _optimize_advanced(
    min_context: int,
    max_context: int,
    min_gpu: float,
    max_gpu: float,
    workload: str,
    gpu_via_cli: bool,
    optimize_context: bool,
    min_generation_tok_s: float = 2.0,
    enforce_min_generation_speed: bool = True,
    run_preset: str = "standard",
    selected_stages: list[str] | None = None,
    max_speed_probes: int | None = None,
    max_finalists: int | None = None,
    benchmark_repetitions: int | None = None,
    validation_repetitions_min: int | None = None,
    validation_repetitions_max: int | None = None,
    recovery_budget: int | None = None,
    quality_context: int = 8192,
    context_strategy_modes: list[str] | None = None,
    final_sandbox_mode: str = "off",
    final_sandbox_preset_ids: list[str] | None = None,
    final_sandbox_custom_prompts: list[str] | None = None,
    sandbox_compare_baseline: bool = False,
    artifact_test_id: str | None = None,
) -> dict:
    """Advanced settings for the optimize command (Phase A/B keys included)."""
    return {
        "min_context": min_context,
        "max_context": max_context,
        "min_gpu_ratio": min_gpu,
        "max_gpu_ratio": max_gpu,
        "test_flash_on": True,
        "test_flash_off": True,
        "test_kv_gpu": True,
        "test_kv_cpu": True,
        "auto_batch": True,
        "workload_type": workload,
        "selection_threshold": 0.05,
        "gpu_via_cli": gpu_via_cli,
        "optimize_context": optimize_context,
        # Speed floor
        "min_generation_tok_s": min_generation_tok_s,
        "enforce_min_generation_speed": enforce_min_generation_speed,
        # Preset
        "run_preset": run_preset,
        # Stage selection
        "selected_stages": selected_stages or [],
        # Budget controls
        "max_speed_probes": max_speed_probes,
        "max_finalists": max_finalists,
        "benchmark_repetitions": benchmark_repetitions,
        "validation_repetitions_min": validation_repetitions_min,
        "validation_repetitions_max": validation_repetitions_max,
        "recovery_budget": recovery_budget,
        # Quality context (explicit default 8192, independent of --max-context)
        "quality_context": quality_context,
        # Scheduled artifact test
        "artifact_test_id": artifact_test_id,
        # Context strategies
        "context_strategy_modes": context_strategy_modes or [],
        # Sandbox
        "final_sandbox_mode": final_sandbox_mode,
        "final_sandbox_preset_ids": final_sandbox_preset_ids or [],
        "final_sandbox_custom_prompts": final_sandbox_custom_prompts or [],
        "sandbox_compare_baseline": sandbox_compare_baseline,
    }


EXPERIMENTAL_PROMPT = "Test experimental features? (RoPE / CPU-MoE / Speculative)"


def _ask_experimental(question: str) -> bool:
    """Single experimental opt-in question; never raises, never blocks CI.

    EOF / closed stdin / non-TTY exceptions resolve to OFF (False).
    """
    try:
        return bool(typer.confirm(question, default=False))
    except Exception:
        return False


def _prompt_experimental_optins() -> dict:
    """Interactive experimental opt-in (default: all off). Never raises."""
    from lm_optimizer.services.optimizer import resolve_experimental_optins

    return resolve_experimental_optins(
        {"enable_rope": None, "enable_cpu_moe": None, "enable_speculative": None},
        ask=_ask_experimental,
    )


def _reject_bool_like_model_args(models: list[str] | None) -> None:
    """Fail fast when a boolean literal leaked into model selection.

    `bool` Typer flags never consume a value, so `--enable-rope False`
    parses as flag ON plus a positional model literally named "False".
    That must be a usage error (exit 2), never a model lookup.
    """
    for mid in models or []:
        if str(mid).strip().lower() in ("true", "false"):
            console.print(
                f"[red]Invalid model {mid!r}: boolean flags do not take values. "
                "Use --enable-rope/--no-enable-rope (and equivalents) instead of "
                "'True'/'False' arguments.[/red]"
            )
            sys.exit(2)


def _resolve_experimental_flags(
    enable_rope: bool | None,
    enable_cpu_moe: bool | None,
    enable_speculative: bool | None,
    non_interactive: bool = False,
) -> dict:
    """Merge CLI flags with the interactive opt-in prompt.

    Explicit CLI flags (True or False) always win and are never re-prompted.
    Unspecified (None) flags are asked one by one only on an interactive
    terminal; everywhere else (`--yes`, `--non-interactive`, pipes, EOF)
    they resolve to OFF. Defaults stay OFF everywhere.
    """
    from lm_optimizer.services.optimizer import resolve_experimental_optins

    requested = {
        "enable_rope": enable_rope,
        "enable_cpu_moe": enable_cpu_moe,
        "enable_speculative": enable_speculative,
    }
    if non_interactive or all(v is not None for v in requested.values()):
        return resolve_experimental_optins(requested, ask=None)
    try:
        interactive = sys.stdin.isatty()
    except Exception:
        interactive = False
    if not interactive:
        return resolve_experimental_optins(requested, ask=None)
    return resolve_experimental_optins(requested, ask=_ask_experimental)


def _phase_transparency_table(run, best):
    """PHASE A/B TRANSPARENCY table (None for legacy runs without phase_ab)."""
    pab = (run.benchmark_params or {}).get("phase_ab") or {}
    if not pab:
        return None
    phase = Table(title="PHASE A/B TRANSPARENCY")
    phase.add_column("Item", style="cyan")
    phase.add_column("Value", style="yellow")
    phase.add_row(
        "Contexts (frozen)",
        f"speed={pab.get('speed_context', '?')} "
        f"quality={pab.get('quality_context', '?')} "
        f"budget={pab.get('budget_class', '?')}",
    )
    raw_id = str(pab.get("raw_fastest_id") or "")
    raw = next((c for c in run.configurations if str(c.id) == raw_id), None)
    if raw is not None:
        q = f"{raw.quality_score.overall:.3f}" if raw.quality_score else "N/A"
        phase.add_row(
            "Raw fastest",
            f"ctx {raw.context_length} "
            f"{raw.get_avg_generation_tok_s():.1f} tok/s quality {q} "
            f"status={raw.status.value}",
        )
    phase.add_row(
        "Quality-safe",
        f"ctx {best.context_length} "
        f"{best.get_avg_generation_tok_s():.1f} tok/s "
        f"(highest acceptable speed above threshold)",
    )
    for entry in pab.get("recovery_log") or []:
        if isinstance(entry, dict) and entry.get("suspect"):
            phase.add_row(
                "Recovery",
                f"{entry.get('suspect')} "
                f"{entry.get('old_value')}->{entry.get('new_value')} "
                f"passed={entry.get('recheck_passed')}",
            )
    for note in pab.get("excluded_notes") or []:
        phase.add_row("Not auto-tuned", str(note)[:100])
    if pab.get("phase_b_enabled"):
        pb = pab.get("phase_b_result") or {}
        phase.add_row(
            "Phase B",
            f"GPU-only max={pb.get('gpu_only_max', '?')} "
            f"system max={pb.get('system_max', '?')} "
            f"final={pb.get('final_capacity', '?')}",
        )
    return phase


def _print_phase_dry_run(
    client,
    model_info,
    max_context: int,
    gpu_via_cli: bool,
    optimize_context: bool,
    quality_context: int | None = None,
) -> None:
    """Phase A/B plan preview (no Cartesian estimate, no server writes)."""
    from lm_optimizer.domain.models import LoadConfiguration
    from lm_optimizer.services.optimizer import resolve_phase_contexts
    from lm_optimizer.services.speed_search import SpeedCaps, build_speed_plan

    speed_ctx, quality_ctx = resolve_phase_contexts(
        {"max_context": max_context, "quality_context": quality_context},
        model_info.context_limit,
    )
    caps = client.capabilities
    rest = {"context_length"}
    for attr, key in (
        ("supports_flash_attention", "flash_attention"),
        ("supports_kv_cache_placement", "offload_kv_cache_to_gpu"),
        ("supports_eval_batch_size", "eval_batch_size"),
        ("supports_physical_batch_size", "physical_batch_size"),
        ("supports_parallel", "parallel"),
        ("supports_context_checkpoints", "context_checkpoints"),
        ("supports_num_experts", "num_experts"),
    ):
        if getattr(caps, attr, False):
            rest.add(key)
    anchor = LoadConfiguration(
        context_length=speed_ctx,
        flash_attention=True,
        offload_kv_cache_to_gpu=True,
        eval_batch_size=2048,
        physical_batch_size=512,
        parallel=4,
        context_checkpoints=32,
    )
    plan, notes = build_speed_plan(
        anchor,
        speed_ctx,
        SpeedCaps(
            rest_keys=frozenset(rest),
            gpu_via_cli=gpu_via_cli,
            is_moe=bool(model_info.is_moe),
        ),
    )
    stages: dict[str, int] = {}
    for item in plan:
        stages[item.stage] = stages.get(item.stage, 0) + 1
    console.print("[yellow]Dry run - Phase A/B plan[/yellow]")
    console.print(f"  Speed context (frozen): {speed_ctx}")
    console.print(f"  Quality context (frozen): {quality_ctx}")
    console.print(
        "  Speed probes: "
        + ", ".join(f"{s}x{n}" for s, n in sorted(stages.items()))
        + " (+S5 interactions)"
    )
    console.print("  Quality finalists: up to 5 (5% band + contrarian)")
    console.print(f"  Phase B max-context: {'ON' if optimize_context else 'OFF'}")
    for note in notes:
        console.print(f"  Not auto-tuned: {note}")


def _save_preset_and_report(
    model,
    profile,
    run,
    hardware,
    output="results",
    extra=None,
    generation=None,
    model_info=None,
    note=None,
    sort="run",
) -> bool:
    """Save preset + per-model .md report. Returns True if best existed."""
    best_config = run.get_best_config()
    if best_config is None:
        console.print("[yellow]No successful configuration to save[/yellow]")
        try:
            failed_path = save_failed_report(run, out_dir=output)
            if failed_path:
                console.print(f"[yellow]Failed-run debug report: {failed_path}[/yellow]")
        except Exception as e:
            logger.debug("Failed report save failed", error=str(e))
        return False
    if model_info is not None:
        moe = expert_advice(model_info)
        if moe:
            extra = dict(extra or {})
            extra.setdefault("MoE experts (num_experts)", "\n".join(moe))
    from lm_optimizer.services.reporting import is_exploratory_run

    if is_exploratory_run(run):
        # Exploratory (quality-skipped) runs never overwrite validated presets.
        console.print("[yellow]Exploratory run: preset not saved (no validated winner)[/yellow]")
    else:
        preset_data = {
            "model_id": model,
            "profile": profile,
            "config": best_config.config.to_dict(),
            "context_length": best_config.context_length,
            "best_config_id": str(best_config.id),
            "metrics": {
                "generation_tok_s": best_config.get_avg_generation_tok_s(),
                "prompt_tok_s": best_config.get_avg_prompt_tok_s(),
                "ttft_ms": best_config.get_avg_estimated_ttft_ms(),
                "quality_score": best_config.quality_score.overall if best_config.quality_score else 0,
            },
            "quality": {
                "overall": best_config.quality_score.overall if best_config.quality_score else 0,
            },
            "run_id": str(run.id),
            "optimizer_version": "1.0.0-beta",
        }

        preset_repo.save(preset_data)
        console.print("\n[green]Preset saved[/green]")

    try:
        ver = _version_info()
        snap_id = capability_repo.save(str(run.id), ver["lms"], ver["backend"])
        console.print(f"[dim]Capability snapshot #{snap_id} recorded[/dim]")
    except Exception as e:
        logger.debug("Capability snapshot failed", error=str(e))

    report_path = save_best_report(
        run,
        out_dir=output,
        host=_host_info_dict(hardware, note=note),
        extra=extra,
        generation=generation,
        sort=sort,
    )
    if report_path:
        console.print(f"[green]Report saved: {report_path}[/green]")
    return True


@app.callback()
def callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    config_file: Path | None = typer.Option(None, "--config", "-c", help="Config file path"),
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="Inference backend: lm-studio (default), ollama, llama-cpp (phase 2)",
    ),
):
    """LM Studio Auto Optimizer."""
    if verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)
    if backend is not None:
        config.backend = _require_backend(backend)


@app.command()
def status(
    url: str | None = typer.Option(
        None, "--url", help="LM Studio API URL (overrides .env, not persisted)"
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Connection check only: no capability probing, no model loads",
    ),
):
    """Show LM Studio connection status and hardware info."""
    setup_logging()

    # Resolve URL: CLI override > backend default (.env) > hardcoded default.
    # Ollama must never inherit the LM Studio URL (different port/server).
    if (config.backend or "lm-studio") == "ollama":
        effective_url = url or config.ollama_base_url
    else:
        effective_url = url or config.lm_studio.base_url

    async def _status():
        client = get_client(base_url=effective_url)
        try:
            await client.connect(echo_probe=not quick)
            if quick:
                console.print("\n[bold]LM Studio (quick check: nothing was loaded)[/bold]")
            hardware = hardware_detector.detect()
            hardware = hardware_detector.detect()

            console.print("\n[bold]Hardware Information[/bold]")
            console.print(f"  OS: {hardware.os}")
            console.print(
                f"  CPU: {hardware.cpu_name} ({hardware.cpu_cores_physical}P/{hardware.cpu_cores_logical}L)"
            )
            console.print(f"  RAM: {hardware.total_ram_gb:.1f} GB")

            if hardware.gpus:
                for gpu in hardware.gpus:
                    console.print(
                        f"  GPU {gpu.index}: {gpu.name} ({gpu.vram_gb:.1f} GB VRAM, {gpu.vendor})"
                    )
            else:
                console.print("  GPU: Not detected")

            console.print(f"\n[bold]{'Ollama' if (config.backend or 'lm-studio') == 'ollama' else 'LM Studio'}[/bold]")
            console.print(f"  URL: {client.base_url}")
            console.print("  Status: [green]Connected[/green]")
            console.print(f"  API Version: {client.capabilities.version}")
            supported = getattr(client.capabilities, "get_supported_load_params", None)
            if callable(supported):
                console.print(f"  Supported Parameters: {', '.join(supported())}")
            if url:
                console.print("  [dim]Using CLI override (not persisted to .env)[/dim]")

        except Exception as e:
            _handle_connection_error(effective_url, e)
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_status())


@app.command()
def models(
    full_ids: bool = typer.Option(
        False,
        "--full-ids",
        help="Print numbered full model IDs (one per line, no truncation)",
    ),
):
    """List all available models (load-free: never loads anything)."""
    setup_logging()

    async def _models():
        client = get_client()
        try:
            await client.connect(echo_probe=False)
            models = await client.list_models()
            if full_ids:
                for i, m in enumerate(models, 1):
                    console.print(f"{i}|{m.id}", highlight=False)
                return

            table = Table(title="Available Models")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Architecture", style="yellow")
            table.add_column("Parameters", justify="right")
            table.add_column("Quantization", style="magenta")
            table.add_column("Context", justify="right")
            table.add_column("MoE", justify="center")
            table.add_column("Size (GB)", justify="right")
            table.add_column("Loaded", justify="center")

            for m in models:
                size_gb = f"{m.size_bytes / (1024**3):.2f}" if m.size_bytes else "N/A"
                table.add_row(
                    m.id,
                    m.name,
                    m.architecture or "N/A",
                    f"{m.parameter_count:,}" if m.parameter_count else "N/A",
                    m.quantization or "N/A",
                    str(m.context_limit or "N/A"),
                    "yes" if m.is_moe else "",
                    size_gb,
                    "yes" if client.get_loaded_model(m.id) else "",
                )

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_models())


@app.command()
def inspect(model: str = typer.Argument(..., help="Model ID to inspect")):
    """Inspect a model's capabilities in detail."""
    setup_logging()

    async def _inspect():
        client = get_client()
        try:
            await client.connect()

            with console.status(f"Inspecting {model}..."):
                model_info = await client.get_model(model)
                if not model_info:
                    console.print(f"[red]Model not found: {model}[/red]")
                    sys.exit(1)

            console.print(Panel.fit(f"[bold]Model Capabilities: {model}[/bold]"))

            info_table = Table()
            info_table.add_column("Property", style="cyan")
            info_table.add_column("Value", style="green")

            info_table.add_row("ID", model_info.id)
            info_table.add_row("Name", model_info.name)
            info_table.add_row("Architecture", model_info.architecture or "Unknown")
            info_table.add_row(
                "Parameters",
                f"{model_info.parameter_count:,}" if model_info.parameter_count else "Unknown",
            )
            info_table.add_row("Quantization", model_info.quantization or "Unknown")
            info_table.add_row("Context Limit", str(model_info.context_limit or "Unknown"))
            info_table.add_row("MoE Model", "Yes" if model_info.is_moe else "No")
            if model_info.is_moe:
                info_table.add_row("Experts", str(model_info.num_experts or "Unknown"))
            info_table.add_row(
                "Size",
                f"{model_info.size_bytes / (1024**3):.2f} GB"
                if model_info.size_bytes
                else "Unknown",
            )
            info_table.add_row(
                "Supported Params", ", ".join(client.capabilities.get_supported_load_params())
            )

            console.print(info_table)

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_inspect())


@app.command()
def benchmark(
    model: str = typer.Argument(..., help="Model ID to benchmark"),
    context: int = typer.Option(4096, "--context", "-c", help="Context length"),
    gpu_ratio: float = typer.Option(1.0, "--gpu-ratio", "-g", help="GPU offload ratio (0-1)"),
    flash: bool = typer.Option(True, "--flash/--no-flash", help="Enable Flash Attention"),
    kv_gpu: bool = typer.Option(True, "--kv-gpu/--kv-cpu", help="KV cache on GPU"),
    batch: int = typer.Option(256, "--batch", "-b", help="Eval batch size"),
    physical_batch: int | None = typer.Option(
        None, "--physical-batch", help="Physical batch size (server default 512)"
    ),
    parallel: int | None = typer.Option(
        None, "--parallel", help="Max concurrency (server default 4)"
    ),
    checkpoints: int | None = typer.Option(
        None, "--checkpoints", help="Context checkpoints (server default 32)"
    ),
    style: str = typer.Option(
        "balanced", "--style", help="Benchmark style (precise/balanced/creative)"
    ),
    repetitions: int = typer.Option(
        3, "--repetitions", "-r", help="Number of benchmark repetitions"
    ),
    max_tokens_scale: float = typer.Option(
        1.0,
        "--max-tokens-scale",
        help="Scale case max-tokens 0.25-1.0 for slow models (may fail quality gates)",
    ),
):
    """Run benchmark with specific configuration."""
    setup_logging()
    style = _require_style(style)

    async def _benchmark():
        client = get_client()
        try:
            snap = await _connect_guarded_or_abort(client, "benchmark")
            for line in format_snapshot(snap):
                console.print(f"  {line}")
            if not snap.get("verified_empty") and snap.get("leftovers"):
                console.print("[red]Stale models loaded, aborting. Unload them first.[/red]")
                sys.exit(1)

            load_config = LoadConfiguration(
                context_length=context,
                gpu_ratio=gpu_ratio,
                flash_attention=flash,
                offload_kv_cache_to_gpu=kv_gpu,
                eval_batch_size=batch,
                physical_batch_size=physical_batch,
                parallel=parallel,
                context_checkpoints=checkpoints,
            )

            benchmark_service = BenchmarkService(
                client,
                benchmark_config=_benchmark_config_obj(repetitions, max_tokens_scale),
                style=style,
            )

            with console.status("Running benchmark..."):
                result = await benchmark_service.run_benchmark(
                    model, load_config, context, style=style
                )

            _display_benchmark_result(result, model)

        except Exception as e:
            logger.exception("Benchmark failed")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_benchmark())


@app.command(name="manual-memory-duel")
def manual_memory_duel(
    model: str = typer.Option(
        ..., "--model", help="Model ID (must match results/<model>-best.md)"
    ),
    stage: str = typer.Option(..., "--stage", help="Manual toggle to duel: mmap|keep"),
    repetitions: int = typer.Option(
        3, "--repetitions", "-r", help="Benchmark repetitions per re-measure"
    ),
    style: str = typer.Option(
        "balanced", "--style", help="Benchmark style (precise/balanced/creative)"
    ),
    output: Path = typer.Option(
        Path("results"), "--output", help="Report output directory"
    ),
):
    """Duel a MANUAL GUI toggle (mmap OFF / Keep OFF) against the auto best.

    Prints the auto best, pauses for the GUI toggle + Enter, re-measures the
    SAME LoadConfiguration (this tool never toggles mmap/keep itself), then
    prints a keep/revert verdict. Fail-closed unload before and after.
    """
    from lm_optimizer.services import manual_memory_duel as _duel

    setup_logging()
    style = _require_style(style)
    if stage not in ("mmap", "keep"):
        console.print("[red]Invalid --stage (mmap|keep)[/red]")
        sys.exit(2)
    try:
        interactive = sys.stdin.isatty()
    except Exception:
        interactive = False
    if not interactive:
        console.print(
            "[red]manual-memory-duel needs an interactive terminal: the GUI "
            "toggle + Enter confirmation cannot be verified otherwise. "
            "Re-measuring without it could compare the wrong setting.[/red]"
        )
        sys.exit(2)

    async def _duel_run():
        client = get_client()
        try:
            snap = await _connect_guarded_or_abort(client, "manual-memory-duel")
            for line in format_snapshot(snap):
                console.print(f"  {line}")
            if not snap.get("verified_empty") and snap.get("leftovers"):
                console.print("[red]Stale models loaded, aborting. Unload them first.[/red]")
                sys.exit(1)
            try:
                await assert_unloaded(client, purpose=f"manual-memory-duel:{model}:pre")
            except UnloadNotClean as e:
                console.print(f"[red]Host not clean, aborting:[/red] {e}")
                sys.exit(1)

            try:
                auto = _duel.load_auto_best(output, model, results_dir=str(output))
            except ValueError as e:
                console.print(f"[red]{e}[/red]")
                sys.exit(1)
            if auto.get("load_config") is None or not auto.get("context_length"):
                console.print(
                    "[red]No re-measurable auto best: the run DB has no load "
                    "configuration for this model (run auto/optimize first). "
                    "Refusing to guess a config.[/red]"
                )
                sys.exit(1)

            console.print(
                f"[bold]Auto best for {model}[/bold] (source: {auto['source']})"
            )
            console.print(
                f"  gen {auto['gen_tok_s']:.1f} tok/s | "
                f"est. TTFT {auto['ttft_ms']:.0f} ms | ctx {auto['context_length']}"
            )

            prompt = _duel.MMAP_PROMPT if stage == "mmap" else _duel.KEEP_PROMPT
            try:
                typer.prompt(prompt, default="")
            except Exception:
                console.print(
                    "[red]Confirmation aborted; nothing was re-measured "
                    "(this tool toggles nothing itself).[/red]"
                )
                sys.exit(1)

            benchmark_service = BenchmarkService(
                client,
                benchmark_config=_benchmark_config_obj(repetitions),
                style=style,
            )
            with console.status(f"Re-measuring with manual {stage} OFF..."):
                result = await benchmark_service.run_benchmark(
                    model, auto["load_config"], auto["context_length"], style=style
                )
            manual_gen = result.get_avg_generation_tok_s()
            manual_ttft = result.get_avg_estimated_ttft_ms()
            console.print(
                f"  manual ({stage} OFF): {manual_gen:.1f} tok/s | "
                f"est. TTFT {manual_ttft:.0f} ms"
            )
            v = _duel.verdict(auto["gen_tok_s"], manual_gen, auto["ttft_ms"], manual_ttft)
            if v["decision"] == "keep":
                console.print(f"[green]{_duel.GUIDANCE_KEEP}[/green] {v['reason']}")
            else:
                console.print(f"[yellow]{_duel.GUIDANCE_REVERT}[/yellow] {v['reason']}")
            other = "keep" if stage == "mmap" else "mmap"
            console.print(f"  Repeat for the other lever with: --stage {other}")

            try:
                await client.ensure_unloaded(model)
            except Exception:
                pass
            try:
                await assert_unloaded(
                    client, purpose=f"manual-memory-duel:{model}:{stage}:post"
                )
            except UnloadNotClean as e:
                console.print(f"[red]Done, but host not clean:[/red] {e}")
                sys.exit(1)
        except SystemExit:
            raise
        except Exception as e:
            logger.exception("Manual memory duel failed")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_duel_run())


def _preflight_optimize(
    model: str,
    profile: str,
    style: str,
    quality_threshold: float,
    repetitions: int,
    min_context: int,
    max_context: int,
    warmup_runs: int,
    report_sort: str,
    skip_phases: str | None,
    physical_batches: str | None,
    enable_rope: bool | None,
    enable_cpu_moe: bool | None,
    enable_speculative: bool | None,
    yes: bool,
    non_interactive: bool,
    dry_run: bool,
    # Granular test selection
    test: list[str] | None = None,
    test_only: bool = False,
    speed_only: bool = False,
    quality_only: bool = False,
    context: int | None = None,
    mtp_test: bool = False,
    reasoning_sweep: bool = False,
    quality_lab: bool = False,
    # Speed floor (spec §3)
    min_generation_tok_s: float = 2.0,
    enforce_min_generation_speed: bool = True,
    # Preset (spec §2)
    run_preset: str = "standard",
    # Stage selection (spec §7)
    selected_stages: list[str] | None = None,
    # Budget controls (spec §4)
    max_speed_probes: int | None = None,
    max_finalists: int | None = None,
    benchmark_repetitions: int | None = None,
    validation_repetitions_min: int | None = None,
    validation_repetitions_max: int | None = None,
    recovery_budget: int | None = None,
    quality_context: int = 8192,
    # Scheduled artifact test (fourth option)
    artifact_test_id: str | None = None,
    # Context strategies (spec §5)
    context_strategy_modes: list[str] | None = None,
    # Final sandbox (spec §6)
    final_sandbox_mode: str = "off",
    final_sandbox_preset_ids: list[str] | None = None,
    final_sandbox_custom_prompts: list[str] | None = None,
    sandbox_compare_baseline: bool = False,
) -> dict:
    """Pure local validation + resolution for optimize. No server calls."""
    if not 0 <= warmup_runs <= 5:
        console.print("[red]Invalid --warmup-runs (0-5)[/red]")
        sys.exit(2)
    if report_sort not in ("run", "speed", "quality"):
        console.print("[red]Invalid --report-sort (run|speed|quality)[/red]")
        sys.exit(2)
    if min_generation_tok_s < 0:
        console.print("[red]Invalid --min-generation-tok-s (must be >= 0)[/red]")
        sys.exit(2)
    if quality_context < 256:
        console.print("[red]Invalid --quality-context (must be >= 256)[/red]")
        sys.exit(2)
    from lm_optimizer.services.optimizer import validate_artifact_test_id

    try:
        artifact_test_id = validate_artifact_test_id(artifact_test_id)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(2)
    if run_preset not in ("quick", "standard", "thorough", "custom"):
        console.print("[red]Invalid --preset (quick|standard|thorough|custom)[/red]")
        sys.exit(2)
    if final_sandbox_mode not in ("off", "one", "all", "custom"):
        console.print("[red]Invalid --final-sandbox (off|one|all|custom)[/red]")
        sys.exit(2)

    valid_stages = {"preflight", "speed", "quality", "recovery", "validation", "context", "sandbox", "experimental"}
    if selected_stages:
        invalid = [s for s in selected_stages if s not in valid_stages]
        if invalid:
            console.print(f"[red]Invalid --stage: {', '.join(invalid)}. Valid: {', '.join(sorted(valid_stages))}[/red]")
            sys.exit(2)

    valid_strategies = {"winner", "gpu_resident", "model_first_kv_cpu", "system_spillover"}
    if context_strategy_modes:
        invalid = [s for s in context_strategy_modes if s not in valid_strategies]
        if invalid:
            console.print(f"[red]Invalid --context-strategy: {', '.join(invalid)}. Valid: {', '.join(sorted(valid_strategies))}[/red]")
            sys.exit(2)

    skips: list[str] = []
    if skip_phases:
        from lm_optimizer.services.optimizer import parse_phase_skips

        try:
            skips = sorted(parse_phase_skips(skip_phases))
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(2)
    batches: list[int] | None = None
    if physical_batches:
        try:
            batches = sorted(
                {int(x.strip()) for x in physical_batches.split(",") if x.strip()}
            )
        except ValueError:
            console.print("[red]Invalid --physical-batches (comma-separated integers)[/red]")
            sys.exit(2)
        if not batches or any(b <= 0 for b in batches):
            console.print("[red]Invalid --physical-batches (positive integers required)[/red]")
            sys.exit(2)

    # Parse test selection options
    include_categories = None
    if test:
        # Map test names to categories
        test_to_category = {
            "short_instruction": "instruction",
            "medium_reasoning": "reasoning",
            "long_context": "context",
            "coding_task": "coding",
            "structured_output": "format",
        }
        cats = set()
        for t in test:
            if t in test_to_category:
                cats.add(test_to_category[t])
            else:
                console.print(f"[red]Unknown test: {t}. Valid: {', '.join(test_to_category.keys())}[/red]")
                sys.exit(2)
        include_categories = cats

    # Validate test selection options
    if speed_only and quality_only:
        console.print("[red]Cannot use --speed-only and --quality-only together[/red]")
        sys.exit(2)
    if speed_only and test:
        console.print("[red]Cannot use --speed-only with --test[/red]")
        sys.exit(2)
    if speed_only and context is None:
        console.print("[red]--speed-only requires --context[/red]")
        sys.exit(2)
    if mtp_test and speed_only:
        console.print("[red]Cannot use --mtp-only with --speed-only[/red]")
        sys.exit(2)
    if mtp_test and quality_only:
        console.print("[red]Cannot use --mtp-only with --quality-only[/red]")
        sys.exit(2)
    if mtp_test and context is None:
        console.print("[red]--mtp-only requires --context[/red]")
        sys.exit(2)
    if reasoning_sweep and speed_only:
        console.print("[red]Cannot use --reasoning-only with --speed-only[/red]")
        sys.exit(2)
    if reasoning_sweep and quality_only:
        console.print("[red]Cannot use --reasoning-only with --quality-only[/red]")
        sys.exit(2)
    if reasoning_sweep and mtp_test:
        console.print("[red]Cannot use --reasoning-only with --mtp-only[/red]")
        sys.exit(2)
    if context is not None and not (speed_only or mtp_test):
        console.print("[red]--context only valid with --speed-only or --mtp-only[/red]")
        sys.exit(2)

    # Granular test modes compose into the phase-skip machinery so the
    # confirmation summary, the runner and the resume path share one plan.
    granular_skips: set[str] = set()
    if speed_only:
        granular_skips |= {"quality", "recovery", "validation", "context"}
    if quality_only:
        granular_skips |= {"speed", "recovery", "validation", "context"}
    if reasoning_sweep:
        granular_skips |= {"speed", "recovery", "validation", "context"}
    if mtp_test:
        granular_skips |= {"quality", "recovery", "validation", "context"}
    if test_only:
        # --test-only runs ONLY the selected tests: skip speed discovery and
        # every downstream phase (same contract as quality-only).
        granular_skips |= {"speed", "recovery", "validation", "context"}
    if granular_skips:
        skips = sorted(set(skips) | granular_skips)
    # Reasoning-only mode: the battery IS the 5-level reasoning sweep.
    if reasoning_sweep:
        include_categories = {"reasoning"}

    experimental_flags = _resolve_experimental_flags(
        enable_rope,
        enable_cpu_moe,
        enable_speculative,
        non_interactive=(non_interactive or yes),
    )
    return {
        "model": model,
        "profile": profile,
        "style": style,
        "quality_threshold": quality_threshold,
        "repetitions": repetitions,
        "min_context": min_context,
        "max_context": max_context,
        "warmup_runs": warmup_runs,
        "report_sort": report_sort,
        "skip_phases": skips,
        "physical_batches": batches,
        "experimental": experimental_flags,
        "dry_run": dry_run,
        # Test selection options
        "include_categories": include_categories,
        "test_only": test_only,
        "speed_only": speed_only,
        "quality_only": quality_only,
        "speed_context": context,
        "mtp_test": mtp_test,
        "reasoning_sweep": reasoning_sweep,
        "quality_lab": quality_lab,
        "test_names": list(test or []),
        # Speed floor
        "min_generation_tok_s": min_generation_tok_s,
        "enforce_min_generation_speed": enforce_min_generation_speed,
        # Preset
        "run_preset": run_preset,
        # Stage selection
        "selected_stages": selected_stages or [],
        # Budget controls
        "max_speed_probes": max_speed_probes,
        "max_finalists": max_finalists,
        "benchmark_repetitions": benchmark_repetitions,
        "validation_repetitions_min": validation_repetitions_min,
        "validation_repetitions_max": validation_repetitions_max,
        "recovery_budget": recovery_budget,
        # Quality context (explicit default 8192, independent of --max-context)
        "quality_context": quality_context,
        # Scheduled artifact test
        "artifact_test_id": artifact_test_id,
        # Context strategies
        "context_strategy_modes": context_strategy_modes or [],
        # Sandbox
        "final_sandbox_mode": final_sandbox_mode,
        "final_sandbox_preset_ids": final_sandbox_preset_ids or [],
        "final_sandbox_custom_prompts": final_sandbox_custom_prompts or [],
        "sandbox_compare_baseline": sandbox_compare_baseline,
    }


@app.command()
def optimize(
    model: str = typer.Argument(..., help="Model ID to optimize"),
    profile: str = typer.Option(
        "balanced", "--profile", "-p", help="Optimization profile (speed/balanced/context/quality)"
    ),
    quality_threshold: float = typer.Option(
        0.97, "--quality", "-q", help="Minimum quality threshold (0.9-1.0)"
    ),
    repetitions: int = typer.Option(
        3, "--repetitions", "-r", help="Benchmark repetitions per config"
    ),
    validation_repetitions: int = typer.Option(
        5, "--validation", "-v", help="Validation repetitions for best config"
    ),
    min_context: int = typer.Option(2048, "--min-context", help="Minimum context length"),
    max_context: int = typer.Option(32768, "--max-context", help="Maximum context length"),
    min_gpu: float = typer.Option(0.0, "--min-gpu", help="Minimum GPU ratio"),
    max_gpu: float = typer.Option(1.0, "--max-gpu", help="Maximum GPU ratio"),
    style: str = typer.Option(
        "balanced", "--style", help="Benchmark style (precise/balanced/creative)"
    ),
    min_tps: float = typer.Option(
        1.0,
        "--min-tps",
        help="Interactive speed floor (spec 8.2 default user limit); hysteresis band scales with it",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show search space without running"),
    resume_id: str | None = typer.Option(
        None, "--resume", help="Resume run ID from checkpoint (no re-runs)"
    ),
    revalidate: bool = typer.Option(
        False, "--revalidate", help="With --resume: re-run completed candidates"
    ),
    workload: str = typer.Option(
        "interactive", "--workload", help="Workload: interactive (TTFT-first) or throughput"
    ),
    gpu_via_cli: bool = typer.Option(
        True,
        "--gpu-via-cli/--no-gpu-via-cli",
        help="Load GPU offload ratios via lms CLI (REST cannot set them)",
    ),
    max_tokens_scale: float = typer.Option(
        1.0,
        "--max-tokens-scale",
        help="Scale case max-tokens 0.25-1.0 for slow models (may fail quality gates)",
    ),
    min_speed: float = typer.Option(
        0.0,
        "--min-speed",
        help="Minimum generation tok/s to accept a config; slower configs are skipped and marked as too slow",
    ),
    # Speed floor (spec §3) - default 2.0 tok/s
    min_generation_tok_s: float = typer.Option(
        2.0,
        "--min-generation-tok-s",
        help="Minimum median generation speed for a winner (default 2.0 tok/s)",
    ),
    enforce_min_generation_speed: bool = typer.Option(
        True,
        "--enforce-min-generation-speed/--no-enforce-min-generation-speed",
        help="Enforce the speed floor as a hard eligibility rule (default ON)",
    ),
    # Preset (spec §2)
    run_preset: str = typer.Option(
        "standard",
        "--preset",
        help="Optimization preset: quick, standard, thorough, custom",
    ),
    optimize_context: bool = typer.Option(
        False,
        "--optimize-context",
        help="Phase B opt-in: max-context sweep on the frozen runtime winner",
    ),
    # Stage selection (spec §7) - custom mode only
    selected_stages: list[str] = typer.Option(
        None,
        "--stage",
        help="Stage to run (repeatable): preflight,speed,quality,recovery,validation,context,sandbox,experimental",
    ),
    # Budget controls (spec §4)
    max_speed_probes: int | None = typer.Option(
        None,
        "--max-speed-probes",
        help="Maximum speed probe count (preset default if unset)",
    ),
    max_finalists: int | None = typer.Option(
        None,
        "--max-finalists",
        help="Maximum finalists for quality suite (preset default if unset)",
    ),
    benchmark_repetitions: int | None = typer.Option(
        None,
        "--benchmark-repetitions",
        help="Benchmark repetitions per config (preset default if unset)",
    ),
    validation_repetitions_min: int | None = typer.Option(
        None,
        "--validation-repetitions-min",
        help="Minimum validation repetitions (preset default if unset)",
    ),
    validation_repetitions_max: int | None = typer.Option(
        None,
        "--validation-repetitions-max",
        help="Maximum validation repetitions (preset default if unset)",
    ),
    recovery_budget: int | None = typer.Option(
        None,
        "--recovery-budget",
        help="Recovery probe budget (preset default if unset)",
    ),
    quality_context: int = typer.Option(
        8192,
        "--quality-context",
        help="Context length for the quality suite (default 8192; independent of --max-context)",
    ),
    artifact_test: str | None = typer.Option(
        None,
        "--artifact-test",
        help="Scheduled artifact test run last on the winner: web_dashboard, "
        "blender_skyblock, research_synthesis, or custom:<path.json>",
    ),
    # Context strategies (spec §5)
    context_strategy_modes: list[str] = typer.Option(
        None,
        "--context-strategy",
        help="Context strategy to test (repeatable): winner,gpu_resident,model_first_kv_cpu,system_spillover",
    ),
    # Final sandbox (spec §6)
    final_sandbox_mode: str = typer.Option(
        "off",
        "--final-sandbox",
        help="Final sandbox mode: off, one, all, custom",
    ),
    final_sandbox_preset_ids: list[str] = typer.Option(
        None,
        "--final-sandbox-preset",
        help="Sandbox preset ID to run (repeatable, for --final-sandbox one/all)",
    ),
    final_sandbox_custom_prompts: list[str] = typer.Option(
        None,
        "--final-sandbox-prompt",
        help="Custom sandbox prompt as JSON (repeatable, for --final-sandbox custom)",
    ),
    sandbox_compare_baseline: bool = typer.Option(
        False,
        "--sandbox-compare-baseline",
        help="Also run sandbox tasks against baseline/default config for comparison",
    ),
    warmup_runs: int = typer.Option(
        1, "--warmup-runs", help="Warmup suite passes per config 0-5 (0 = fastest, noisier)"
    ),
    report_sort: str = typer.Option(
        "run", "--report-sort", help="Tested-configs table order: run, speed, quality"
    ),
    skip_phases: str | None = typer.Option(
        None,
        "--skip-phases",
        help="Comma-separated phases to skip: speed,quality,recovery,validation,context",
    ),
    physical_batches: str | None = typer.Option(
        None,
        "--physical-batches",
        help="Comma-separated physical batch sizes to test (e.g. 256,512,1024)",
    ),
    enable_rope: bool | None = typer.Option(
        None,
        "--enable-rope/--no-enable-rope",
        help="Experimental tail opt-in: RoPE probes after the FINAL winner (default OFF)",
    ),
    enable_cpu_moe: bool | None = typer.Option(
        None,
        "--enable-cpu-moe/--no-enable-cpu-moe",
        help="Experimental tail opt-in: CPU-MoE probes after the FINAL winner (default OFF)",
    ),
    enable_speculative: bool | None = typer.Option(
        None,
        "--enable-speculative/--no-enable-speculative",
        help="Experimental tail opt-in: speculative probes after the FINAL winner (default OFF)",
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Accept the pre-flight summary without prompting (never enables experiments)"
    ),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", help="Never prompt; unspecified experiments stay OFF"
    ),
    wait_interval: float = typer.Option(
        60.0,
        "--wait-interval",
        help="Seconds between foreign-model eviction re-checks before start",
    ),
    max_waits: int = typer.Option(
        30,
        "--max-waits",
        help="Max re-checks for a foreign model before pausing with a resume hint",
    ),
    # Granular test selection
    test: list[str] = typer.Option(
        None,
        "--test",
        help="Comma-separated test names to run (short_instruction,medium_reasoning,long_context,coding_task,structured_output)",
    ),
    test_only: bool = typer.Option(
        False,
        "--test-only",
        help="Run only specified tests, skip speed discovery phase",
    ),
    speed_only: bool = typer.Option(
        False,
        "--speed-only",
        help="Run only speed test (short_instruction) at specific context",
    ),
    quality_only: bool = typer.Option(
        False,
        "--quality-only",
        help="Skip speed discovery, run quality tests only",
    ),
    context: int = typer.Option(
        None,
        "--context",
        help="Context size for speed-only test",
    ),
    mtp_test: bool = typer.Option(
        False,
        "--mtp-only",
        "--mtp-test",
        help="Run only the speculative decoding (MTP) test "
        "(--mtp-test kept as an alias)",
    ),
    reasoning_sweep: bool = typer.Option(
        False,
        "--reasoning-only",
        "--reasoning-sweep",
        help="Run only the reasoning sweep (off,low,medium,high,xhigh); "
        "--reasoning-sweep kept as an alias",
    ),
    quality_lab: bool = typer.Option(
        False,
        "--quality-lab",
        help="Run the deterministic Quality Lab tasks on every quality finalist "
        "(hard gates can veto the winner)",
    ),
    flash: bool = typer.Option(
        True,
        "--flash/--no-flash",
        help="Enable Flash Attention (required for V cache quantization)",
    ),
    kv_gpu: bool = typer.Option(
        True,
        "--kv-gpu/--kv-cpu",
        help="Offload KV cache to GPU (requires Flash Attention when V cache quant enabled)",
    ),
):
    """Optimize a model configuration."""
    setup_logging()
    style = _require_style(style)
    from lm_optimizer.services.workload import normalize_workload

    try:
        workload = normalize_workload(workload)
    except Exception:
        console.print("[red]Invalid --workload (interactive/throughput)[/red]")
        sys.exit(2)
    if not resume_id:
        # Auto-recovery: if a previous process for this model died (crash,
        # kill, reboot) and left a stale "running" run with a usable
        # checkpoint, resume it automatically instead of starting fresh.
        # This is what lets an optimization survive a BSOD/reboot: after the
        # machine comes back up, re-running `optimize <model>` continues
        # where it left off.
        try:
            from lm_optimizer.services.joblock import _holder_alive, _read_holder
            from lm_optimizer.services.run_reconcile import reconcile_stale_runs
            from lm_optimizer.storage.run_checkpoint import load_checkpoint

            holder_dead = True
            try:
                from lm_optimizer.services.joblock import _lock_path

                holder = _read_holder(_lock_path())
                if holder and _holder_alive(holder):
                    holder_dead = False
            except Exception:
                pass

            if holder_dead:
                recon = reconcile_stale_runs(
                    dry_run=True, model_id=model, is_fresh=lambda rid, now: False
                )
                if recon.get("changed", 0) > 0:
                    for ch in recon.get("changes", []):
                        try:
                            ckpt = load_checkpoint(ch["id"])
                        except Exception:
                            ckpt = None
                        if ckpt:
                            console.print(
                                f"[yellow]Stale run {ch['id'][:8]} found for "
                                f"{model} with a valid checkpoint. Auto-resuming.[/yellow]"
                            )
                            resume_id = ch["id"]
                            break
        except Exception as e:
            logger.debug("Auto-resume detection failed", error=str(e))

    if resume_id:
        # Delegate to resume flow (model arg is ignored except for display).
        resume(run_id=resume_id, revalidate=revalidate, style=style, skip_phases=None)
        return

    _reject_bool_like_model_args([model])

    effective = _preflight_optimize(
        model, profile, style, quality_threshold, repetitions,
        min_context, max_context, warmup_runs, report_sort,
        skip_phases, physical_batches,
        enable_rope, enable_cpu_moe, enable_speculative,
        yes, non_interactive, dry_run,
        speed_only=speed_only,
        quality_only=quality_only,
        context=context,
        mtp_test=mtp_test,
        reasoning_sweep=reasoning_sweep,
        quality_lab=quality_lab,
        min_generation_tok_s=min_generation_tok_s,
        enforce_min_generation_speed=enforce_min_generation_speed,
        run_preset=run_preset,
        selected_stages=selected_stages,
        max_speed_probes=max_speed_probes,
        max_finalists=max_finalists,
        benchmark_repetitions=benchmark_repetitions,
        validation_repetitions_min=validation_repetitions_min,
        validation_repetitions_max=validation_repetitions_max,
        recovery_budget=recovery_budget,
        quality_context=quality_context,
        artifact_test_id=artifact_test,
        context_strategy_modes=context_strategy_modes,
        final_sandbox_mode=final_sandbox_mode,
        final_sandbox_preset_ids=final_sandbox_preset_ids,
        final_sandbox_custom_prompts=final_sandbox_custom_prompts,
        sandbox_compare_baseline=sandbox_compare_baseline,
    )

    if not _preflight_confirm(effective, dry_run, yes, non_interactive):
        return

    async def _optimize():
        client = get_client()
        try:
            try:
                snap = await _connect_guarded(
                    client,
                    "optimize",
                    wait_interval_s=wait_interval,
                    max_waits=max_waits,
                )
            except HostBusyTimeout as e:
                _print_host_busy_paused(model, e)
                sys.exit(1)
            except HostUnreachableError as e:
                _handle_connection_error(_client_base_url(client), e.cause or e)
                sys.exit(1)
            # Auto-reconcile stale runs for this model: a previous process that
            # died (crash, kill, reboot) leaves a "running" row behind. Mark
            # such rows interrupted so they stop accumulating and the user can
            # resume explicitly with --resume <id> if they want to continue.
            try:
                from lm_optimizer.services.run_reconcile import reconcile_stale_runs

                recon = reconcile_stale_runs(dry_run=False, model_id=model)
                if recon.get("changed"):
                    console.print(
                        f"[yellow]Reconciled {recon['changed']} stale run(s) for "
                        f"{model} (marked interrupted).[/yellow]"
                    )
                    for ch in recon.get("changes", []):
                        console.print(
                            f"  [dim]{ch['id'][:8]}  {ch['from']} -> {ch['to']} "
                            f"(resume with --resume {ch['id']})[/dim]"
                        )
            except Exception as e:
                logger.debug("Auto-reconcile failed", error=str(e))
            for line in format_snapshot(snap):
                console.print(f"  {line}")
            if not snap.get("verified_empty") and snap.get("leftovers"):
                console.print("[red]Stale models loaded, aborting. Unload them first.[/red]")
                sys.exit(1)

            # Get model info
            model_info = await client.get_model(model)
            if not model_info:
                console.print(f"[red]Model not found: {model}[/red]")
                sys.exit(1)

            hardware = hardware_detector.detect()

            # Advanced settings
            advanced = _optimize_advanced(
                min_context,
                max_context,
                min_gpu,
                max_gpu,
                workload,
                gpu_via_cli,
                optimize_context,
                min_generation_tok_s,
                enforce_min_generation_speed,
                run_preset,
                selected_stages,
                max_speed_probes,
                max_finalists,
                benchmark_repetitions,
                validation_repetitions_min,
                validation_repetitions_max,
                recovery_budget,
                quality_context,
                context_strategy_modes,
                final_sandbox_mode,
                final_sandbox_preset_ids,
                final_sandbox_custom_prompts,
                sandbox_compare_baseline,
                artifact_test,
            )

            if dry_run:
                _print_phase_dry_run(
                    client, model_info, max_context, gpu_via_cli, optimize_context,
                    quality_context,
                )
                return

            advanced.update(effective["experimental"])
            advanced["min_speed"] = min_speed
            advanced["min_tps"] = min_tps
            advanced["warmup_runs"] = effective["warmup_runs"]
            advanced["report_sort"] = effective["report_sort"]
            if effective["skip_phases"]:
                advanced["skip_phases"] = effective["skip_phases"]
            if effective["physical_batches"]:
                advanced["physical_batches"] = effective["physical_batches"]

            # Granular test selection must reach the service AND the persisted
            # run: resume() rebuilds BenchmarkService from run.advanced_settings,
            # so dropping these keys would silently turn a resumed granular run
            # back into a full campaign.
            advanced["speed_only"] = bool(speed_only)
            advanced["quality_only"] = bool(quality_only)
            advanced["mtp_test"] = bool(mtp_test)
            advanced["reasoning_sweep"] = bool(reasoning_sweep)
            advanced["quality_lab"] = bool(quality_lab)
            if test:
                advanced["test_names"] = list(test)
            if context is not None:
                advanced["speed_context"] = int(context)
            advanced["flash"] = flash
            advanced["kv_gpu"] = kv_gpu
            # Explicit CLI repetition overrides beat the preset; untouched
            # defaults (3 / 5) leave the preset budget in charge.
            if repetitions != 3:
                advanced["benchmark_repetitions"] = int(repetitions)
            if validation_repetitions != 5:
                advanced["validation_repetitions_max"] = max(2, int(validation_repetitions))

            console.print(f"[bold]Starting optimization for {model}[/bold]")
            console.print(f"  Profile: {profile}")
            console.print(f"  Quality threshold: {quality_threshold}")
            console.print(f"  Repetitions: {repetitions}")
            console.print(f"  Validation repetitions: {validation_repetitions}")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Optimizing {model}...", total=None)

                def _bar(stage: str, tested: int, total: int) -> None:
                    progress.update(
                        task,
                        description=f"Optimizing {model}... [{stage} {tested}/{total} configs]",
                    )

                advanced["progress_cb"] = _bar
                result = await _run_optimization(
                    client,
                    model_info,
                    hardware,
                    profile,
                    quality_threshold,
                    repetitions,
                    style,
                    advanced,
                    max_tokens_scale=max_tokens_scale,
                )

            _save_preset_and_report(
                model,
                profile,
                result,
                hardware,
                generation=_generation_for(model, model_info.architecture),
                model_info=model_info,
                sort=report_sort,
            )
            run_status = str(getattr(result, "status", "") or "")
            if run_status == "failed":
                # Hard run-level failure (no config produced usable output).
                # Nonzero exit so unattended runners (granular suite) can
                # distinguish it from a soft no-winner outcome (rc=0).
                console.print("[red]Run failed; exiting with code 3.[/red]")
                sys.exit(3)

            # Fail-closed post-hygiene: the host must be clean for whatever runs next.
            try:
                await assert_unloaded(client, purpose=f"optimize:{model}")
            except UnloadNotClean as e:
                logger.error("Host not clean after optimize", model=model)
                console.print(f"[red]Done, but host not clean:[/red] {e}")
                sys.exit(1)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted — checkpoint saved, resume with:[/yellow]")
            console.print("  lm-optimizer checkpoints")
            sys.exit(130)
        except Exception as e:
            logger.exception("Optimization failed")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            await client.close()

    _acquire_run_lock(f"optimize:{model}")
    try:
        asyncio.run(_optimize())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — checkpoint saved.[/yellow]")
        sys.exit(130)
    finally:
        release_lock()


@app.command()
def apply(
    model: str = typer.Argument(..., help="Model ID to apply preset to"),
    profile: str = typer.Option("balanced", "--profile", "-p", help="Profile to apply"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be applied"),
):
    """Apply a saved preset to LM Studio."""
    setup_logging()

    async def _apply():
        client = get_client()
        preset = preset_repo.get_by_model_and_profile(model, profile)
        if not preset:
            console.print(f"[red]No preset found for {model} with profile {profile}[/red]")
            console.print("What you can do:")
            console.print("  - Run 'lm-optimizer optimize' first to create a preset.")
            console.print(f"  - Check results/<model>-best.md for the measured winner and apply it manually.")
            console.print(f"  - Run 'lm-optimizer judge <run-id>' to pick a config from an existing run.")
            console.print(f"  - List presets with 'lm-optimizer presets'.")
            sys.exit(1)

        console.print(f"Preset for {model} ({profile}):")
        console.print(JSON.from_data(preset["config"]))

        if dry_run:
            console.print("[yellow]Dry run: would load model with this configuration[/yellow]")
            return

        try:
            await client.connect()
            load_config = LoadConfiguration(**preset["config"])

            with console.status("Loading model with optimized config..."):
                result = await client.load_model(model, load_config)

            if result.success:
                console.print("[green]Model loaded successfully![/green]")
                console.print(f"Identifier: {result.identifier}")
            else:
                console.print(f"[red]Failed to load: {result.error}[/red]")
                console.print("What you can do:")
                console.print("  - The host may be busy: unload other models in LM Studio first.")
                console.print("  - The config may exceed VRAM: try a preset with smaller context.")
                console.print(f"  - Check run state with 'lm-optimizer run-status' and host with 'lm-optimizer status'.")
                sys.exit(1)

        except Exception as e:
            logger.exception("Apply failed")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_apply())


@dataclasses.dataclass
class ApplyEntry:
    """One applicable configuration in the apply-menu picker."""

    backend: str  # "lmstudio" | "ollama"
    label: str  # human-readable model/tag
    detail: str  # one-line config summary
    model: str = ""  # LM Studio model id (lmstudio backend)
    profile: str = "balanced"  # preset profile (lmstudio backend)
    path: str = ""  # modelfile path (ollama backend)


def _collect_apply_entries(presets: list[dict], modelfiles: list[dict]) -> list[ApplyEntry]:
    """Build picker entries from DB presets + local ollama modelfiles.

    Pure function (no servers, no DB): callers pass
    ``preset_repo.list_all()`` and parsed ``ollama-configs/*.modelfile.md``.
    """
    entries: list[ApplyEntry] = []
    for p in presets or []:
        model_id = str(p.get("model_id", "") or "")
        profile = str(p.get("profile", "") or "balanced")
        if not model_id:
            continue
        cfg = p.get("config") or {}
        ctx = cfg.get("context_length", "?")
        entries.append(
            ApplyEntry(
                backend="lmstudio",
                label=f"{model_id} [{profile}]",
                detail=f"ctx {ctx}",
                model=model_id,
                profile=profile,
            )
        )
    for m in modelfiles or []:
        path = str(m.get("path", "") or "")
        base = str(m.get("from", "") or "?")
        if not path:
            continue
        n_params = max(0, len(m) - 2)  # minus path/from keys
        entries.append(
            ApplyEntry(
                backend="ollama",
                label=f"{path} (from {base})",
                detail=f"{n_params} option(s)",
                path=path,
            )
        )
    return entries


def _resolve_apply_choice(entries: list[ApplyEntry], choice: int) -> ApplyEntry | None:
    """1-based picker resolution (0/Enter/out-of-range → None = cancel)."""
    if not entries or choice <= 0 or choice > len(entries):
        return None
    return entries[choice - 1]


def _scan_ollama_config_dir() -> list[dict]:
    """Parse local ollama-configs/*.modelfile.md (no server calls)."""
    from lm_optimizer.backends.ollama.modelfile import OLLAMA_CONFIG_DIR, parse_modelfile

    found: list[dict] = []
    try:
        files = sorted(OLLAMA_CONFIG_DIR.glob("*.modelfile.md"))
    except Exception:
        return found
    for f in files:
        try:
            parsed = parse_modelfile(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        parsed["path"] = str(f)
        found.append(parsed)
    return found


@app.command(name="apply-menu")
def apply_menu(
    backend: str = typer.Option(
        "lmstudio", "--backend", "-b", help="Backend to list: lmstudio or ollama"
    ),
):
    """Interactive picker for applying optimized configs (no benchmarks).

    lmstudio: applies a saved DB preset (loads the model once).
    ollama: applies a local ollama-configs/*.modelfile.md as a NEW tag.
    """
    setup_logging()
    normalized = (backend or "").strip().lower()
    if normalized not in ("lmstudio", "ollama"):
        console.print(f"[red]Invalid --backend: {backend} (expected lmstudio|ollama)[/red]")
        sys.exit(2)

    if normalized == "lmstudio":
        entries = _collect_apply_entries(preset_repo.list_all(), [])
    else:
        entries = _collect_apply_entries([], _scan_ollama_config_dir())
    if not entries:
        if normalized == "lmstudio":
            console.print("[yellow]No saved presets. Run 'lm-optimizer optimize' first.[/yellow]")
        else:
            console.print(
                "[yellow]No local modelfiles. Create one with 'lm-optimizer ollama-export -o name.modelfile.md'.[/yellow]"
            )
        return

    table = Table(title=f"Apply configs ({normalized}, select by number)")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Config", style="green")
    table.add_column("Detail", style="dim")
    for idx, e in enumerate(entries, 1):
        table.add_row(str(idx), e.label, e.detail)
    console.print(table)
    console.print()

    try:
        choice = typer.prompt(
            "Select config number to apply (or Enter to cancel)",
            type=int,
            default=0,
            show_default=False,
        )
    except (typer.Abort, ValueError):
        console.print("[yellow]Cancelled[/yellow]")
        return
    selected = _resolve_apply_choice(entries, choice)
    if selected is None:
        console.print("[red]Invalid selection[/red]")
        return

    console.print(f"\nSelected: [bold]{selected.label}[/bold] ({selected.detail})")
    if not typer.confirm("Apply this config? (loads the model)", default=True):
        console.print("[yellow]Cancelled[/yellow]")
        return
    if selected.backend == "lmstudio":
        apply(selected.model, selected.profile)
    else:
        from pathlib import Path as _Path

        ollama_apply(file=_Path(selected.path), as_tag=None, yes=True, base_url=None)


@app.command()
def runs(
    model: str | None = typer.Argument(None, help="Filter by model ID"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of runs to show"),
):
    """List optimization runs."""
    setup_logging()

    if model:
        runs_list = run_repo.get_by_model(model, limit)
    else:
        runs_list = run_repo.list_all(limit)

    if not runs_list:
        console.print("[yellow]No runs found[/yellow]")
        return

    table = Table(title="Optimization Runs")
    table.add_column("Date", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Profile", style="yellow")
    table.add_column("Status", justify="center")
    table.add_column("Best Score", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Configs", justify="right")

    for r in runs_list:
        status_style = {
            "completed": "green",
            "success": "green",
            "partial_success": "yellow",
            "running": "yellow",
            "resumed": "yellow",
            "failed": "red",
            "cancelled": "red",
            "interrupted": "red",
        }.get(r.status.value, "white")

        # Lean list rows carry no configurations; resolve the full run for best.
        full = run_repo.get(str(r.id)) or r
        best = full.get_best_config()
        n_configs = len(full.configurations) if full.configurations else len(r.configurations)
        model_name = full.model.name or full.model.id
        table.add_row(
            r.created_at.strftime("%Y-%m-%d %H:%M"),
            model_name,
            r.profile.value,
            f"[{status_style}]{r.status.value}[/{status_style}]",
            f"{best.score:.3f}" if best and best.score is not None else "-",
            f"{r.duration_seconds:.1f}s" if r.duration_seconds else "-",
            str(n_configs),
        )

    console.print(table)


TEST_MODEL_IDS = {"m", "resume-model-x", "isolation-probe", "iso-ckpt"}


@app.command()
def cleanup_test_runs(
    confirm: bool = typer.Option(
        False, "--confirm", help="Actually delete (default lists candidates only)"
    ),
):
    """Remove test-fixture runs (e.g. resume-model-x, m) from the real database.

    Default is a dry run listing candidates. Genuine optimization results are
    never touched; stale non-test rows are listed, not deleted.
    """
    setup_logging()
    from lm_optimizer.storage import run_checkpoint as _ckpt

    candidates = []
    for model_id in sorted(TEST_MODEL_IDS):
        candidates.extend(run_repo.get_by_model(model_id, 10000))
    stale = [
        r
        for r in run_repo.list_all(10000)
        if r.status.value in ("running", "pending") and r.model.id not in TEST_MODEL_IDS
    ]

    if not candidates:
        console.print("[green]No test-fixture runs found.[/green]")
    else:
        table = Table(title="Test-fixture runs (safe to delete)")
        table.add_column("Run", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Status")
        table.add_column("Created", style="dim")
        for r in candidates:
            table.add_row(
                str(r.id)[:8], r.model.id, r.status.value, r.created_at.strftime("%Y-%m-%d %H:%M")
            )
        console.print(table)

    if stale:
        console.print("[yellow]Stale non-test running/pending rows (NOT deleted):[/yellow]")
        for r in stale[:20]:
            console.print(
                f"  {str(r.id)[:8]} {r.model.id} {r.status.value} "
                f"{r.created_at.strftime('%Y-%m-%d %H:%M')}"
            )

    if not candidates:
        return
    if not confirm:
        console.print(
            "\n[yellow]Dry run: pass --confirm to delete the rows above "
            "(runs + configurations + checkpoints).[/yellow]"
        )
        return

    removed_runs = removed_ckpts = 0
    for r in candidates:
        if run_repo.delete(str(r.id)):
            removed_runs += 1
        if _ckpt.remove_checkpoint(r.id):
            removed_ckpts += 1
    for model_id in sorted(TEST_MODEL_IDS):
        run_repo.delete_orphan_model(model_id)
    console.print(f"[green]Removed {removed_runs} runs and {removed_ckpts} checkpoints.[/green]")


@app.command()
def presets(
    model: str | None = typer.Argument(None, help="Filter by model ID"),
):
    """List saved presets."""
    setup_logging()

    if model:
        presets_list = preset_repo.list_by_model(model)
    else:
        presets_list = preset_repo.list_all()

    if not presets_list:
        console.print("[yellow]No presets found[/yellow]")
        return

    table = Table(title="Saved Presets")
    table.add_column("Model", style="cyan")
    table.add_column("Profile", style="green")
    table.add_column("Created", style="yellow")
    table.add_column("Gen tok/s", justify="right")
    table.add_column("Quality", justify="right")

    for p in presets_list:
        metrics = p.get("metrics", {})
        table.add_row(
            p["model_id"],
            p["profile"],
            p["created_at"][:19] if p["created_at"] else "-",
            f"{metrics.get('generation_tok_s', 0):.1f}",
            f"{metrics.get('quality_score', 0):.3f}",
        )

    console.print(table)


@app.command(name="preset-export")
def preset_export(
    model: str = typer.Argument(..., help="Model ID with a saved winning preset"),
    profile: str = typer.Option("balanced", "--profile", "-p", help="Preset profile to export"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output file (default: lmstudio-presets/<model>_<profile>.preset.json)"
    ),
):
    """Export the winning preset as an importable LM Studio JSON file.

    LM Studio has no direct preset API; this writes a validated JSON preset
    (applied_params + recorded-only params + source run/candidate) that you can
    apply with `lm-optimizer apply` or copy into LM Studio's GUI preset editor.
    """
    setup_logging()
    from lm_optimizer.services.preset_export import (
        LMSTUDIO_PRESET_DIR,
        build_lmstudio_preset,
        default_preset_filename,
        write_lmstudio_preset,
    )

    preset = preset_repo.get_by_model_and_profile(model, profile)
    if not preset:
        console.print(f"[red]No saved preset for {model} (profile {profile})[/red]")
        console.print("Run 'lm-optimizer optimize' first, or list with 'lm-optimizer presets'.")
        sys.exit(1)
    cfg = preset.get("config") or {}
    ctx = preset.get("context_length")
    if ctx is None:
        ctx = cfg.get("context_length")
    try:
        export = build_lmstudio_preset(
            model_id=preset.get("model_id", model),
            profile=preset.get("profile", profile),
            config=cfg,
            context_length=ctx,
            run_id=preset.get("run_id"),
            best_config_id=preset.get("best_config_id"),
            metrics=preset.get("metrics"),
            quality=preset.get("quality"),
            optimizer_version=preset.get("optimizer_version"),
        )
        if output is not None:
            out_dir = output.parent if str(output.parent) not in ("", ".") else LMSTUDIO_PRESET_DIR
            path = write_lmstudio_preset(export, out_dir=out_dir, filename=output.name)
        else:
            path = write_lmstudio_preset(export)
    except (ValueError, OSError) as e:
        console.print(f"[red]Preset export failed: {e}[/red]")
        sys.exit(1)
    console.print(f"[green]Exported LM Studio preset:[/green] {path}")
    console.print(f"  Context: {export['context_length']}")
    if export["unsupported_params"]:
        console.print(
            f"  [yellow]Recorded-only (REST cannot set): "
            f"{', '.join(export['unsupported_params'])}[/yellow]"
        )
    console.print("  Apply with 'lm-optimizer apply' or copy applied_params into the GUI.")


@app.command()
def restore(model: str = typer.Argument(..., help="Model ID to restore")):
    """Restore previous model configuration."""
    setup_logging()

    async def _restore():
        client = get_client()
        try:
            await client.connect()
            await client.unload_model(model_id=model)
            console.print(f"[green]Model {model} unloaded[/green]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_restore())


@app.command()
def recommend(
    vram: float | None = typer.Option(
        None, "--vram", help="VRAM budget in GB (default: detected GPU, else 6)"
    ),
    style: str = typer.Option(
        "balanced", "--style", help="Benchmark style (precise/balanced/creative)"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Max models to show"),
    estimate: bool = typer.Option(
        False, "--estimate", help="Add lms memory estimates (slow, no loading)"
    ),
):
    """Recommend models for a VRAM budget + manual GUI/host checklist."""
    setup_logging()
    style = _require_style(style)

    async def _recommend():
        client = get_client()
        try:
            await client.connect()
            models = await client.list_models()

            hardware = hardware_detector.detect()
            budget = vram
            if budget is None:
                budget = hardware.gpus[0].vram_gb if hardware.gpus else 6.0
            console.print(f"\n[bold]VRAM budget: {budget:.1f} GB[/bold]")
            for line in threads_advice(hardware.cpu_cores_physical, hardware.cpu_cores_logical):
                console.print(f"  {line}")

            recs = recommend_for_vram(models, budget)[:limit]
            estimates: dict[str, dict] = {}
            if estimate:
                from lm_optimizer.services import lms_cli

                console.print("[dim]Requesting lms memory estimates (no loading)...[/dim]")
                for r in recs:
                    estimates[r.model_id] = lms_cli.estimate(r.model_id)
            table = Table(title="Model fit")
            table.add_column("Model", style="cyan")
            table.add_column("Size (GB)", justify="right")
            table.add_column("Quant", style="magenta")
            table.add_column("Context", justify="right")
            table.add_column("Fit", justify="center")
            if estimate:
                table.add_column("lms est. GPU", justify="right")
            table.add_column("Note", style="dim")
            for r in recs:
                fit = "yes" if r.fits else ("marginal" if r.marginal else "no")
                row = [
                    r.model_id,
                    f"{r.size_gb:.2f}" if r.size_gb else "N/A",
                    r.quantization or "N/A",
                    str(r.context_limit or "N/A"),
                    fit,
                ]
                if estimate:
                    est = estimates.get(r.model_id, {})
                    gpu = est.get("gpu_gib")
                    row.append(f"{gpu:.2f}GB ({est.get('confidence', '?')})" if gpu else "N/A")
                row.append(r.reason)
                table.add_row(*row)
            console.print(table)

            console.print(f"\n[bold]Suggested style for benchmarks: {style}[/bold]")
            console.print("  precise: code/math  |  balanced: general  |  creative: summarization")

            console.print("\n[bold]Manual checklist (GUI/host only, REST cannot set these):[/bold]")
            for i, item in enumerate(manual_checklist(budget), 1):
                console.print(f"  {i}. {item}")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_recommend())


def _parse_skip_models(value: str | None) -> list[str]:
    """Comma-separated skip substrings, lowercased; empty means skip nothing."""
    if not value:
        return []
    return [p.strip().lower() for p in value.split(",") if p.strip()]


def _apply_skip_models(targets: list, skip_terms: list[str]) -> tuple[list, list]:
    """Split targets into (kept, skipped) by case-insensitive substring match."""
    kept, skipped = [], []
    for t in targets:
        mid = (getattr(t, "id", "") or "").lower()
        (skipped if any(s in mid for s in skip_terms) else kept).append(t)
    return kept, skipped


def _preflight_auto(
    models: list[str] | None,
    profile: str,
    style: str,
    quality_threshold: float,
    repetitions: int,
    skip_matrix: bool,
    skip_optimize: bool,
    skip: str | None,
    max_size_gb: float | None,
    speculative_draft: str | None,
    kv: str,
    precision: bool,
    output: Path,
    dry_run: bool,
    workload: str,
    gpu_via_cli: bool,
    max_tokens_scale: float,
    min_speed: float,
    warmup_runs: int,
    report_sort: str,
    skip_phases: str | None,
    physical_batches: str | None,
    enable_rope: bool | None,
    enable_cpu_moe: bool | None,
    enable_speculative: bool | None,
    yes: bool,
    non_interactive: bool,
    wait_interval: float,
    max_waits: int,
) -> tuple[dict, list | None]:
    """Validate all local options, resolve experimental flags, determine targets if explicit models given.

    Returns (effective_settings, targets_if_known).
    If no explicit models given, targets is None (need to connect to list).
    """
    _reject_bool_like_model_args(models)
    # Local validation
    if kv not in ("gpu", "cpu", "both"):
        console.print("[red]Invalid --kv (gpu/cpu/both)[/red]")
        sys.exit(2)

    if not 0 <= warmup_runs <= 5:
        console.print("[red]Invalid --warmup-runs (0-5)[/red]")
        sys.exit(2)
    if report_sort not in ("run", "speed", "quality"):
        console.print("[red]Invalid --report-sort (run|speed|quality)[/red]")
        sys.exit(2)

    # Parse skip-phases
    auto_skips: list[str] = []
    if skip_phases:
        from lm_optimizer.services.optimizer import parse_phase_skips
        try:
            auto_skips = sorted(parse_phase_skips(skip_phases))
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(2)

    # Parse physical-batches
    auto_physical_batches: list[int] | None = None
    if physical_batches:
        try:
            auto_physical_batches = sorted(
                {int(x.strip()) for x in physical_batches.split(",") if x.strip()}
            )
        except ValueError:
            console.print("[red]Invalid --physical-batches (comma-separated integers)[/red]")
            sys.exit(2)
        if not auto_physical_batches or any(b <= 0 for b in auto_physical_batches):
            console.print("[red]Invalid --physical-batches (positive integers required)[/red]")
            sys.exit(2)

    # Resolve experimental flags
    experimental_flags = _resolve_experimental_flags(
        enable_rope,
        enable_cpu_moe,
        enable_speculative,
        non_interactive=(non_interactive or yes),
    )

    # Determine explicit targets
    explicit_targets: list | None = None
    if models:
        explicit_targets = models  # model IDs, will be resolved after connect

    # Build effective settings for summary
    effective = {
        "models": explicit_targets or "all (small first)",
        "profile": profile,
        "style": style,
        "quality_threshold": quality_threshold,
        "repetitions": repetitions,
        "skip_matrix": skip_matrix,
        "skip_optimize": skip_optimize,
        "skip": skip,
        "max_size_gb": max_size_gb,
        "speculative_draft": speculative_draft,
        "kv": kv,
        "precision": precision,
        "output": str(output),
        "dry_run": dry_run,
        "workload": workload,
        "gpu_via_cli": gpu_via_cli,
        "max_tokens_scale": max_tokens_scale,
        "min_speed": min_speed,
        "warmup_runs": warmup_runs,
        "report_sort": report_sort,
        "skip_phases": auto_skips,
        "physical_batches": auto_physical_batches,
        "experimental": experimental_flags,
        "wait_interval": wait_interval,
        "max_waits": max_waits,
    }

    return effective, explicit_targets


def _show_preflight_summary(effective: dict) -> None:
    """Display consolidated pre-flight summary."""
    from rich.table import Table
    table = Table(title="Pre-flight Summary", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="yellow")
    table.add_column("Value", style="white")

    for key, value in effective.items():
        if key == "models" and isinstance(value, list):
            table.add_row(key, ", ".join(value) if value else "(none)")
        elif key == "experimental":
            exp_active = [k for k, v in value.items() if v]
            table.add_row(key, ", ".join(exp_active) if exp_active else "all OFF")
        elif isinstance(value, (list, dict)):
            table.add_row(key, str(value))
        else:
            table.add_row(key, str(value))

    console.print(table)


def _preflight_confirm(effective: dict, dry_run: bool, yes: bool, non_interactive: bool) -> bool:
    """Show summary and get confirmation. Returns True to proceed, False to abort."""
    _show_preflight_summary(effective)

    if dry_run:
        console.print("[yellow]Dry run — showing plan only, not executing.[/yellow]")
        return False

    if non_interactive or yes:
        console.print("[green]Proceeding (non-interactive / --yes).[/green]")
        return True

    try:
        return typer.confirm("Start optimization?", default=True)
    except Exception:
        console.print("[red]Confirmation aborted; nothing was started.[/red]")
        return False


@app.command()
def auto(
    models: list[str] | None = typer.Argument(None, help="Model keys (default: all, small first)"),
    profile: str = typer.Option(
        "balanced", "--profile", "-p", help="Optimization profile (speed/balanced/context/quality)"
    ),
    quality_threshold: float = typer.Option(
        0.97, "--quality", "-q", help="Minimum quality threshold (0.9-1.0)"
    ),
    style: str = typer.Option(
        "balanced", "--style", help="Benchmark style (precise/balanced/creative)"
    ),
    max_context: int = typer.Option(8192, "--max-context", help="Maximum context length"),
    repetitions: int = typer.Option(
        2, "--repetitions", "-r", help="Benchmark repetitions per config"
    ),
    skip_matrix: bool = typer.Option(False, "--skip-matrix", help="Skip VRAM matrix"),
    skip_optimize: bool = typer.Option(False, "--skip-optimize", help="Skip optimization"),
    skip: str | None = typer.Option(
        None,
        "--skip",
        help="Comma-separated model id substrings to skip (e.g. --skip 27b,gpt-oss)",
    ),
    max_size_gb: float | None = typer.Option(
        None, "--max-size-gb", help="Auto-pick only models up to this file size (default: all)"
    ),
    speculative_draft: str | None = typer.Option(
        None, "--speculative-draft", help="Draft model key for opt-in speculative A/B stage"
    ),
    kv: str = typer.Option(
        "both",
        "--kv",
        help="KV placement to test (gpu/cpu/both); gpu-only keeps small models off CPU",
    ),
    precision: bool = typer.Option(
        True, "--precision/--no-precision", help="Test temp 0.2 for max precision"
    ),
    output: Path = typer.Option("results", "--output", help="Report output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without running"),
    workload: str = typer.Option(
        "interactive", "--workload", help="Workload: interactive or throughput"
    ),
    gpu_via_cli: bool = typer.Option(
        True,
        "--gpu-via-cli/--no-gpu-via-cli",
        help="Load GPU offload ratios via lms CLI (REST cannot set them)",
    ),
    max_tokens_scale: float = typer.Option(
        1.0,
        "--max-tokens-scale",
        help="Scale case max-tokens 0.25-1.0 for slow models (may fail quality gates)",
    ),
    min_speed: float = typer.Option(
        0.0,
        "--min-speed",
        help="Minimum generation tok/s to accept a config; slower configs are skipped and marked as too slow",
    ),
    warmup_runs: int = typer.Option(
        1, "--warmup-runs", help="Warmup suite passes per config 0-5 (0 = fastest, noisier)"
    ),
    report_sort: str = typer.Option(
        "run", "--report-sort", help="Tested-configs table order: run, speed, quality"
    ),
    skip_phases: str | None = typer.Option(
        None,
        "--skip-phases",
        help="Comma-separated phases to skip: speed,quality,recovery,validation,context",
    ),
    physical_batches: str | None = typer.Option(
        None,
        "--physical-batches",
        help="Comma-separated physical batch sizes to test (e.g. 256,512,1024)",
    ),
    enable_rope: bool | None = typer.Option(
        None,
        "--enable-rope/--no-enable-rope",
        help="Experimental tail opt-in: RoPE probes after the FINAL winner (default OFF)",
    ),
    enable_cpu_moe: bool | None = typer.Option(
        None,
        "--enable-cpu-moe/--no-enable-cpu-moe",
        help="Experimental tail opt-in: CPU-MoE probes after the FINAL winner (default OFF)",
    ),
    enable_speculative: bool | None = typer.Option(
        None,
        "--enable-speculative/--no-enable-speculative",
        help="Experimental tail opt-in: speculative probes after the FINAL winner (default OFF)",
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Accept the pre-flight summary without prompting (never enables experiments)"
    ),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", help="Never prompt; unspecified experiments stay OFF"
    ),
    wait_interval: float = typer.Option(
        60.0,
        "--wait-interval",
        help="Seconds between foreign-model eviction re-checks before start",
    ),
    max_waits: int = typer.Option(
        30,
        "--max-waits",
        help="Max re-checks for a foreign model before aborting with a message",
    ),
):
    """Unattended pipeline: smoke -> precision -> ladder(ctx max) -> matrix -> optimize."""
    setup_logging()
    style = _require_style(style)

    _reject_bool_like_model_args(models)

    # ========== PRE-FLIGHT: all local validation, resolution, summary, confirmation ========
    effective, explicit_targets = _preflight_auto(
        models,
        profile,
        style,
        quality_threshold,
        repetitions,
        skip_matrix,
        skip_optimize,
        skip,
        max_size_gb,
        speculative_draft,
        kv,
        precision,
        output,
        dry_run,
        workload,
        gpu_via_cli,
        max_tokens_scale,
        min_speed,
        warmup_runs,
        report_sort,
        skip_phases,
        physical_batches,
        enable_rope,
        enable_cpu_moe,
        enable_speculative,
        yes,
        non_interactive,
        wait_interval,
        max_waits,
    )

    if not _preflight_confirm(effective, dry_run, yes, non_interactive):
        return

    # Resolve experimental flags now (already computed in preflight, but need them for execution)
    experimental_flags = effective["experimental"]
    auto_skips = effective["skip_phases"]
    auto_physical_batches = effective["physical_batches"]
    test_kv_gpu = kv in ("gpu", "both")
    test_kv_cpu = kv in ("cpu", "both")

    # ========== EXECUTION: connect and run ========
    async def _auto():
        client = get_client()
        summary = []
        pipeline_t0 = time.perf_counter()
        try:
            snap = await _connect_guarded_or_abort(
                client,
                "auto",
                wait_interval_s=wait_interval,
                max_waits=max_waits,
            )
            remote_note = _warn_if_remote(client.base_url)
            for line in format_snapshot(snap):
                console.print(f"  {line}")
            hardware = hardware_detector.detect()
            has_gpu = bool(hardware.gpus)
            all_models = await client.list_models()

            # Resolve targets: explicit IDs or all (small first)
            if explicit_targets:
                targets = []
                for mid in explicit_targets:
                    info = await client.get_model(mid)
                    if not info:
                        console.print(f"[yellow]Model not found, skipping: {mid}[/yellow]")
                        summary.append({"model": mid, "stage": "skipped", "detail": "not found"})
                        continue
                    targets.append(info)
            else:
                targets = sorted(
                    (m for m in all_models if "embed" not in m.id.lower()),
                    key=lambda m: m.size_bytes or float("inf"),
                )

            targets, skipped_models = _apply_skip_models(targets, _parse_skip_models(skip))
            for m in skipped_models:
                console.print(f"[yellow]Skipping (user --skip list): {m.id}[/yellow]")
                summary.append({"model": m.id, "stage": "skipped", "detail": "user --skip list"})

            if dry_run:
                console.print("[yellow]Dry run plan (small first):[/yellow]")
                for m in targets:
                    size = f"{m.size_bytes / 1024**3:.2f}GB" if m.size_bytes else "N/A"
                    console.print(
                        f"  {m.id} ({size}): smoke ctx 2048, precision 0.2, "
                        f"ladder ctx <= {max_context}, matrix, optimize "
                        f"profile={profile} style={style}"
                    )
                return

            smoke_service = BenchmarkService(client, style=style)
            quality_evaluator = QualityEvaluator(QualityConfig(minimum_score=quality_threshold))
            host_dirty_stop = False
            for m in targets:
                row = {
                    "model": m.id,
                    "stage": "done",
                    "detail": "",
                    "full": "",
                    "t0": time.perf_counter(),
                }
                try:
                    console.print(f"\n[bold]=== {m.id} ===[/bold]")
                    smoke_ctx = min(2048, m.context_limit or 2048)
                    smoke_kv = True if kv == "gpu" else (False if kv == "cpu" else has_gpu)
                    smoke_cfg = LoadConfiguration(
                        context_length=smoke_ctx,
                        flash_attention=True,
                        offload_kv_cache_to_gpu=smoke_kv,
                    )
                    ok, tok, err = await smoke_service.smoke_test(m.id, smoke_cfg)
                    if not ok:
                        diag = diagnose_load_error(err)
                        console.print(f"[red]Smoke failed: {err}[/red]")
                        for g in diag["guidance"]:
                            console.print(f"  - {g}")
                        row.update(stage="smoke-failed", detail=f"{diag['category']}: {err}"[:200])
                        row["full"] = err + "\n" + "\n".join(f"- {g}" for g in diag["guidance"])
                        row["elapsed_s"] = round(time.perf_counter() - row["t0"], 1)
                        summary.append(row)
                        continue
                    console.print(f"[green]Smoke OK: {tok:.1f} tok/s[/green]")
                    row["detail"] = f"smoke {tok:.1f} tok/s"
                    precision_lines: list[str] = []

                    if precision:
                        probe = await smoke_service.precision_probe(m.id, smoke_cfg)
                        if probe.get("error"):
                            precision_lines.append(f"probe error: {probe['error']}")
                        else:
                            q_def, q_low, n = 0.0, 0.0, 0
                            for tname, entry in probe["tests"].items():
                                s_def = quality_evaluator.evaluate(
                                    m.id, tname, entry["default_text"]
                                )
                                s_low = quality_evaluator.evaluate(m.id, tname, entry["low_text"])
                                q_def += s_def.overall
                                q_low += s_low.overall
                                n += 1
                                precision_lines.append(
                                    f"{tname}: default {entry['default_temp']} "
                                    f"-> {s_def.overall:.3f} vs 0.2 -> {s_low.overall:.3f}"
                                )
                            mean_low = q_low / max(n, 1)
                            mean_def = q_def / max(n, 1)
                            if mean_low >= mean_def:
                                verdict = "0.2 better-or-equal"
                                tail = "; use 0.2 for max precision"
                            else:
                                verdict = "suite default better"
                                tail = ""
                            precision_lines.append(
                                f"verdict: {verdict} ({mean_low:.3f} vs {mean_def:.3f}){tail}"
                            )
                            console.print("[bold]Precision probe (0.2 vs suite default):[/bold]")
                            for line in precision_lines:
                                console.print(f"  {line}")
                            row["detail"] += f"; precision: {verdict}"

                    # 1. Context ladder first: find max loadable ctx, then tune below it.
                    kv_ladder = [True, False] if kv == "both" else [kv == "gpu"]
                    ladder_contexts = [
                        c
                        for c in (2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
                        if c <= max_context and c <= (m.context_limit or 10**9)
                    ] or [smoke_ctx]
                    ladder_out = await ctx_ladder(smoke_service, m.id, ladder_contexts, kv_ladder)
                    relevant = [ladder_out["ceilings"].get(k) for k in kv_ladder]
                    ceil = max([v for v in relevant if v] or [0])
                    if ceil == 0:
                        first_err = ""
                        for k in kv_ladder:
                            for c in sorted(ladder_out["paths"].get(k, {})):
                                e = ladder_out["paths"][k][c].get("error", "")
                                if e:
                                    first_err = e
                                    break
                            if first_err:
                                break
                        diag = diagnose_load_error(first_err)
                        console.print(f"[red]No loadable context: {first_err}[/red]")
                        for g in diag["guidance"]:
                            console.print(f"  - {g}")
                        row.update(
                            stage="ladder-failed", detail=f"no loadable ctx: {first_err}"[:200]
                        )
                        row["full"] = (
                            first_err + "\n" + "\n".join(f"- {g}" for g in diag["guidance"])
                        )
                        row["elapsed_s"] = round(time.perf_counter() - row["t0"], 1)
                        summary.append(row)
                        continue
                    row["detail"] += f"; ladder ceiling {ceil}"
                    console.print(f"[green]Ladder ceiling: {ceil}[/green]")

                    if not skip_matrix:
                        contexts = [
                            c
                            for c in (2048, 4096, 8192)
                            if c <= min(max_context, ceil) and c <= (m.context_limit or 10**9)
                        ] or [smoke_ctx]
                        results = [
                            await run_one(client, m.id, c)
                            for c in build_matrix(
                                contexts,
                                [True, False],
                                [True] if kv == "gpu" else ([False] if kv == "cpu" else None),
                            )
                        ]
                        n_ok = sum(1 for r in results if r["ok"])
                        console.print(recommendation_line(results))
                        row["detail"] += f"; matrix {n_ok}/{len(results)} OK"
                        if n_ok == 0:
                            row.update(
                                stage="matrix-failed",
                                detail=(row["detail"] + "; matrix: no passing config")[:300],
                            )
                            row["full"] = "matrix configs failed; see matrix output above"
                            row["elapsed_s"] = round(time.perf_counter() - row["t0"], 1)
                            summary.append(row)
                            continue
                    if not skip_optimize:
                        from lm_optimizer.services.workload import normalize_workload

                        advanced = {
                            "min_context": 2048,
                            "max_context": min(max_context, ceil, m.context_limit or max_context),
                            "test_flash_on": True,
                            "test_flash_off": True,
                            "test_kv_gpu": test_kv_gpu,
                            "test_kv_cpu": test_kv_cpu,
                            "auto_batch": True,
                            "workload_type": normalize_workload(workload),
                            "selection_threshold": 0.05,
                            "gpu_via_cli": gpu_via_cli,
                            "min_speed": min_speed,
                            "skip_phases": auto_skips,
                            **experimental_flags,
                        }
                        if auto_physical_batches:
                            advanced["physical_batches"] = auto_physical_batches
                        advanced["warmup_runs"] = warmup_runs
                        run = await _run_optimization(
                            client,
                            m,
                            hardware,
                            profile,
                            quality_threshold,
                            repetitions,
                            style,
                            advanced,
                            max_tokens_scale=max_tokens_scale,
                        )
                        best = run.get_best_config()
                        if best is None:
                            row.update(
                                stage="optimize-failed",
                                detail=(row["detail"] + "; optimize: no passing config")[:300],
                            )
                        else:
                            extra = {}
                            if precision_lines:
                                extra["Precision probe (0.2 vs suite default)"] = "\n".join(
                                    precision_lines
                                )
                            if speculative_draft:
                                from lm_optimizer.services.speculative import (
                                    ab_compare,
                                    discover_drafts,
                                )

                                drafts = discover_drafts(all_models)
                                keys = {getattr(d, "id", "") for d in drafts}
                                if speculative_draft not in keys:
                                    console.print(
                                        f"[yellow]Draft model not found locally: "
                                        f"{speculative_draft} (discovered: "
                                        f"{sorted(keys) or 'none'}). Skipping stage.[/yellow]"
                                    )
                                else:
                                    console.print(
                                        "[bold]Speculative A/B stage "
                                        f"(draft {speculative_draft}):[/bold]"
                                    )
                                    ab = await ab_compare(
                                        client, m.id, best.config, speculative_draft
                                    )
                                    if ab.get("ran"):
                                        line = (
                                            f"baseline {ab['baseline_tok_s']} tok/s vs "
                                            f"spec {ab['spec_tok_s']} tok/s "
                                            f"(x{ab.get('speedup', '?')})"
                                        )
                                    else:
                                        line = f"skipped: {ab.get('error', 'unknown')}"
                                    console.print(f"  {line}")
                                    row["detail"] += f"; speculative: {line}"
                                    extra["Speculative decoding A/B"] = (
                                        line
                                        + f"\nacceptance: {ab.get('acceptance', 'n/a')}"
                                        + (
                                            f"\nVRAM delta: {ab['vram_delta_mb']} MB"
                                            if "vram_delta_mb" in ab
                                            else ""
                                        )
                                    )
                            _save_preset_and_report(
                                m.id,
                                profile,
                                run,
                                hardware,
                                output=str(output),
                                extra=extra or None,
                                generation=_generation_for(m.id, m.architecture),
                                model_info=m,
                                note=remote_note,
                                sort=report_sort,
                            )
                            row["detail"] += (
                                f"; best ctx {best.context_length} "
                                f"{best.get_avg_generation_tok_s():.1f} tok/s"
                            )
                    row["elapsed_s"] = round(time.perf_counter() - row["t0"], 1)
                    summary.append(row)
                except Exception as e:
                    logger.exception("Auto model failed", model=m.id)
                    row.update(stage="error", detail=f"{type(e).__name__}: {e}"[:200])
                    row["full"] = f"{type(e).__name__}: {e}"
                    row["elapsed_s"] = round(time.perf_counter() - row["t0"], 1)
                    summary.append(row)
                finally:
                    try:
                        await client.ensure_unloaded(m.id)
                    except Exception:
                        pass
                # Fail-closed boundary: the next model must never inherit a
                # dirty host (two models sharing the GPU invalidates speeds).
                try:
                    await assert_unloaded(client, purpose=f"auto:{m.id}")
                except UnloadNotClean as e:
                    logger.error("Host not clean, stopping pipeline", model=m.id)
                    console.print(f"[red]Stopping pipeline, host not clean:[/red] {e}")
                    summary.append({
                        "model": "pipeline",
                        "stage": "host-dirty",
                        "detail": str(e)[:200],
                        "full": str(e),
                        "t0": time.perf_counter(),
                        "elapsed_s": 0.0,
                    })
                    host_dirty_stop = True
                    break

            table = Table(title="Auto pipeline summary")
            table.add_column("Model", style="cyan")
            table.add_column("Stage", style="yellow")
            table.add_column("Detail", style="dim")
            table.add_column("Elapsed", justify="right")
            for row in summary:
                elapsed = f"{row.get('elapsed_s', 0):.0f}s"
                table.add_row(row["model"], row["stage"], row["detail"], elapsed)
            console.print(table)

            out_dir = Path(output)
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            total_elapsed = round(time.perf_counter() - pipeline_t0, 1)
            summary_path = out_dir / f"auto-summary-{stamp}.md"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(f"# Auto pipeline summary ({stamp})\n\n")
                f.write(f"- Profile: {profile}, style: {style}, max_context: {max_context}\n")
                f.write(f"- Total elapsed: {total_elapsed:.0f}s\n\n")
                f.write("| Model | Stage | Elapsed | Detail |\n| --- | --- | --- | --- |\n")
                for row in summary:
                    f.write(
                        f"| {row['model']} | {row['stage']} | "
                        f"{row.get('elapsed_s', 0):.0f}s | {row['detail']} |\n"
                    )
                failures = [r for r in summary if r.get("full")]
                if failures:
                    f.write("\n## Failures in detail (full errors)\n\n")
                    for r in failures:
                        f.write(f"### {r['model']} ({r['stage']})\n\n")
                        f.write(f"{r['full']}\n\n")
            console.print(f"[green]Summary saved: {summary_path}[/green]")
            if host_dirty_stop:
                console.print(
                    "[red]Pipeline stopped: host could not be cleaned. "
                    "Unload models in LM Studio (disable Keep Model in Memory) "
                    "and resume.[/red]"
                )
                sys.exit(1)

        except Exception as e:
            logger.exception("Auto pipeline failed")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            try:
                await client.unload_all()
            except Exception:
                pass
            await client.close()

    _acquire_run_lock("auto")
    try:
        asyncio.run(_auto())
    finally:
        release_lock()


@app.command()
def fit(
    models: list[str] | None = typer.Argument(
        None, help="Model keys (default: all 5GB+, big first)"
    ),
    contexts: list[int] = typer.Option(
        [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144],
        "--contexts",
        help="Context ladder (ascending, capped by model limit)",
    ),
    kv: str = typer.Option("both", "--kv", help="KV paths to probe (gpu/cpu/both)"),
    output: Path = typer.Option("results", "--output", help="Report output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without running"),
    wait_interval: float = typer.Option(
        60.0,
        "--wait-interval",
        help="Seconds between foreign-model eviction re-checks before start",
    ),
    max_waits: int = typer.Option(
        30,
        "--max-waits",
        help="Max re-checks for a foreign model before aborting with a message",
    ),
):
    """Max-context ladder: escalate ctx until load refusal, per KV path (fit, no tuning)."""
    setup_logging()
    if kv not in ("gpu", "cpu", "both"):
        console.print("[red]Invalid --kv (gpu/cpu/both)[/red]")
        sys.exit(2)
    kv_modes = [True, False] if kv == "both" else [kv == "gpu"]

    async def _fit():
        client = get_client()
        try:
            snap = await _connect_guarded_or_abort(
                client, "fit", wait_interval_s=wait_interval, max_waits=max_waits
            )
            for line in format_snapshot(snap):
                console.print(f"  {line}")
            hardware = hardware_detector.detect()
            gpu_vram = hardware.gpus[0].vram_gb if hardware.gpus else 0.0
            all_models = await client.list_models()

            if models:
                targets = []
                for mid in models:
                    info = await client.get_model(mid)
                    if not info:
                        console.print(f"[yellow]Model not found, skipping: {mid}[/yellow]")
                        continue
                    targets.append(info)
            else:
                targets = sorted(
                    (
                        m
                        for m in all_models
                        if "embed" not in m.id.lower() and (m.size_bytes or 0) >= 5_000_000_000
                    ),
                    key=lambda m: -(m.size_bytes or 0),
                )

            oversized = [m for m in targets if (m.size_bytes or 0) / 1024**3 > gpu_vram > 0]
            if oversized:
                console.print(
                    "[bold yellow]Reminder for models bigger than VRAM "
                    f"({', '.join(m.id for m in oversized)}):[/bold yellow] set GPU "
                    "offload to maximum in LM Studio GUI (or run "
                    "`lms load <model> --gpu max`) so the most layers land in VRAM. "
                    "KV quant and threads stay GUI-only."
                )

            if dry_run:
                console.print("[yellow]Dry run plan (big first):[/yellow]")
                for m in targets:
                    size = f"{m.size_bytes / 1024**3:.2f}GB" if m.size_bytes else "N/A"
                    console.print(f"  {m.id} ({size}): ladder {sorted(contexts)}")
                return

            smoke_service = BenchmarkService(client)
            host_info = _host_info_dict(hardware)
            for m in targets:
                try:
                    console.print(f"\n[bold]=== {m.id} ===[/bold]")
                    model_contexts = sorted(
                        c for c in contexts if c <= (m.context_limit or 10**9)
                    ) or [2048]
                    result = await ctx_ladder(smoke_service, m.id, model_contexts, kv_modes)
                    save_fit_report(result, out_dir=str(output), host=host_info)
                    for path_kv, label in ((True, "KV-GPU"), (False, "KV-CPU")):
                        if path_kv not in kv_modes:
                            continue
                        console.print(f"  {label} ceiling: {result['ceilings'][path_kv]}")
                except Exception as e:
                    logger.exception("Fit model failed", model=m.id)
                    console.print(f"[red]{m.id}: {type(e).__name__}: {e}[/red]")
                finally:
                    try:
                        await client.ensure_unloaded(m.id)
                    except Exception:
                        pass
                # Fail-closed boundary (see auto loop): never ladder the next
                # model on a dirty host.
                try:
                    await assert_unloaded(client, purpose=f"fit:{m.id}")
                except UnloadNotClean as e:
                    logger.error("Host not clean, stopping fit", model=m.id)
                    console.print(f"[red]Stopping fit, host not clean:[/red] {e}")
                    sys.exit(1)
        except Exception as e:
            logger.exception("Fit failed")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            try:
                await client.unload_all()
            except Exception:
                pass
            await client.close()

    _acquire_run_lock("fit")
    try:
        asyncio.run(_fit())
    finally:
        release_lock()


async def _ctx_quality_suites(
    client,
    model: str,
    report: dict,
    top_n: int,
    threshold: float,
    repetitions: int,
    gpu_ratio: float,
    flash: bool,
    kv_gpu: bool,
    batch: int,
    style: str = "balanced",
    warmup_runs: int = 1,
) -> list[dict]:
    """Full quality suites on the top-N stable sweep rungs (ctx --quality).

    Same runtime as the capacity probes, descending ctx. Returns per-rung
    dicts (ctx, tok/s, quality, checks, VRAM, TTFT, pass). Never raises
    for a single failed rung (records it, continues).
    """
    from lm_optimizer.services.benchmark import BenchmarkService
    from lm_optimizer.services.context_sweep import pick_quality_rungs
    from lm_optimizer.services.quality import QualityConfig, QualityEvaluator

    rungs = pick_quality_rungs(report, top_n)
    if not rungs:
        console.print("[yellow]No stable rungs for quality suites[/yellow]")
        return []
    svc = BenchmarkService(
        client, benchmark_config=_benchmark_config_obj(max(1, repetitions), 1.0, warmup_runs)
    )
    evaluator = QualityEvaluator(QualityConfig(minimum_score=threshold))
    rows: list[dict] = []
    table = Table(title=f"Quality suites (top-{len(rungs)} stable contexts): {model}")
    table.add_column("Context", justify="right")
    table.add_column("Gen tok/s", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Checks", justify="right")
    table.add_column("VRAM GB", justify="right")
    table.add_column("TTFT ms", justify="right")
    table.add_column("Verdict")
    for ctx_len in rungs:
        cfg = LoadConfiguration(
            context_length=ctx_len,
            gpu_ratio=gpu_ratio,
            flash_attention=flash,
            offload_kv_cache_to_gpu=kv_gpu,
            eval_batch_size=batch,
        )
        try:
            with console.status(f"Quality suite ctx={ctx_len}..."):
                res = await svc.run_benchmark(model, cfg, ctx_len, style=style)
            scores = evaluator.evaluate_all(model, res.metrics)
            agg = evaluator.aggregate_quality(scores)
            ok = agg.overall >= threshold
            row = {
                "ctx": ctx_len,
                "tok_s": round(res.get_avg_generation_tok_s(), 1),
                "quality": round(agg.overall, 3),
                "checks": f"{agg.checks_passed}/{agg.checks_total}",
                "vram_gb": round(res.peak_vram_gb, 1) if res.peak_vram_gb else None,
                "ttft_ms": round(res.get_avg_estimated_ttft_ms(), 0),
                "pass": bool(ok),
            }
        except Exception as e:
            logger.exception("Quality rung failed", ctx=ctx_len)
            row = {"ctx": ctx_len, "tok_s": 0.0, "quality": 0.0, "checks": "0/0",
                   "vram_gb": None, "ttft_ms": 0, "pass": False,
                   "error": f"{type(e).__name__}: {e}"[:160]}
        rows.append(row)
        table.add_row(
            str(row["ctx"]),
            f"{row['tok_s']:.1f}",
            f"{row['quality']:.3f}",
            str(row["checks"]),
            f"{row['vram_gb']:.1f}" if row["vram_gb"] is not None else "—",
            f"{row['ttft_ms']:.0f}",
            "[green]PASS[/green]" if row["pass"] else "[red]FAIL[/red]",
        )
    console.print(table)
    return rows


@app.command()
def ctx(
    model: str = typer.Argument(..., help="Model ID to probe"),
    min_context: int = typer.Option(2048, "--min-context", help="Minimum context"),
    max_context: int = typer.Option(65536, "--max-context", help="Maximum context"),
    step: int = typer.Option(1000, "--step", help="Refinement granularity (500/1000)"),
    factor: float = typer.Option(
        2.0, "--factor", help="Geometric growth factor (1.5 for spec-compliant bracketing)"
    ),
    round_to: int = typer.Option(512, "--round-to", help="Rung rounding granularity"),
    profile: str = typer.Option("balanced", "--profile", "-p", help="Profile for recommendation"),
    gpu_ratio: float = typer.Option(1.0, "--gpu-ratio", help="GPU offload ratio (0-1)"),
    kv_gpu: bool = typer.Option(True, "--kv-gpu/--kv-cpu", help="KV cache on GPU"),
    flash: bool = typer.Option(True, "--flash/--no-flash", help="Flash attention"),
    batch: int = typer.Option(256, "--batch", "-b", help="Eval batch size"),
    repetitions: int = typer.Option(1, "--repetitions", "-r", help="Smoke repetitions per point"),
    warmup_runs: int = typer.Option(
        1, "--warmup-runs", help="Warmup suite passes per quality rung 0-5 (0 = fastest)"
    ),
    quality_rungs: int = typer.Option(
        0, "--quality-rungs", help="Full quality suites on top-N stable rungs (0 = off)"
    ),
    quality_threshold: float = typer.Option(
        0.97, "--quality-threshold", help="Pass threshold for quality-rung verdicts"
    ),
    min_tps: float = typer.Option(
        1.0,
        "--min-tps",
        help="Interactive speed floor for usable-context classification (tok/s)",
    ),
    retrieval: bool = typer.Option(
        True,
        "--retrieval/--no-retrieval",
        help="Validate needle retrieval on the largest stable rung (spec 8.6)",
    ),
    detailed_fill: bool = typer.Option(
        False,
        "--detailed-fill",
        help="Full fill-ratio matrix (0/25/50/75/90/98%) instead of fast (0/50/90%)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without running"),
    wait_interval: float = typer.Option(
        60.0,
        "--wait-interval",
        help="Seconds between foreign-model eviction re-checks before start",
    ),
    max_waits: int = typer.Option(
        30,
        "--max-waits",
        help="Max re-checks for a foreign model before aborting with a message",
    ),
):
    """Test context capacity for ONE model with a fixed runtime configuration."""
    setup_logging()
    if step not in (500, 1000):
        console.print("[red]Invalid --step (use 500 or 1000)[/red]")
        sys.exit(2)
    ladder = _ctx_ladder(min_context, max_context, factor=factor, round_to=round_to)
    if dry_run:
        console.print(
            f"[yellow]Dry run: would probe {ladder} then bisect with step {step}[/yellow]"
        )
        console.print(
            f"Runtime: gpu={gpu_ratio} kv={'GPU' if kv_gpu else 'CPU'} "
            f"flash={flash} batch={batch} profile={profile}"
        )
        return

    async def _ctx():
        client = get_client()
        try:
            snap = await _connect_guarded_or_abort(
                client, "ctx", wait_interval_s=wait_interval, max_waits=max_waits
            )
            for line in format_snapshot(snap):
                console.print(f"  {line}")
            model_info = await client.get_model(model)
            if not model_info:
                console.print(f"[red]Model not found: {model}[/red]")
                sys.exit(1)
            # Clamp to model limit.
            hi = min(max_context, model_info.context_limit or max_context)
            svc = BenchmarkService(client)
            # Estimate-before-load (§12): lms estimate prunes escalation;
            # boundary candidates are still empirically loaded via precheck.
            from lm_optimizer.services import lms_cli as _lms

            _use_estimate = _lms.lms_available() and not dry_run
            _est_cache: dict[int, dict] = {}

            def _precheck(ctx_len: int) -> str | None:
                if not _use_estimate:
                    return None
                est = _lms.estimate_for(
                    model,
                    context=ctx_len,
                    gpu=gpu_ratio if kv_gpu else "off",
                )
                _est_cache[ctx_len] = est
                if est.get("returncode", 0) != 0:
                    return f"estimate refused (rc={est.get('returncode')})"
                return None

            async def _probe(ctx_len: int) -> dict:
                cfg = LoadConfiguration(
                    context_length=ctx_len,
                    gpu_ratio=gpu_ratio,
                    flash_attention=flash,
                    offload_kv_cache_to_gpu=kv_gpu,
                    eval_batch_size=batch,
                )
                toks: list[float] = []
                last_err = ""
                ok_count = 0
                for _ in range(max(1, repetitions)):
                    ok, tok, err = await svc.smoke_test(model, cfg)
                    last_err = err
                    if ok:
                        ok_count += 1
                        toks.append(tok)
                if ok_count == 0:
                    return {"ok": False, "tok_s": 0.0, "error": (last_err or "FAIL")[:160]}
                toks_sorted = sorted(toks)
                med = toks_sorted[len(toks_sorted) // 2]
                return {"ok": True, "tok_s": round(med, 1), "error": ""}

            report = await _ctx_sweep(_probe, min_context, hi, step=step, precheck=_precheck, min_tps=min_tps)
            if _est_cache:
                console.print("[dim]lms estimates (not proof):[/dim]")
                for ctx_len in sorted(_est_cache):
                    est = _est_cache[ctx_len]
                    console.print(
                        f"  ctx {ctx_len}: gpu={est.get('gpu_gib')}GiB "
                        f"total={est.get('total_gib')}GiB "
                        f"confidence={est.get('confidence')}"
                    )
            console.print(_ctx_format(report, model))
            console.print("")
            console.print(f"Maximum stable context:\n{report['maximum_stable_context']}")
            perf = report["performance_optimal"]
            bal = report["balanced_recommended"]
            if perf:
                console.print(
                    f"\nPerformance-optimal context:\n{perf['ctx']} "
                    f"({perf.get('tok_s', 0):.1f} tok/s)"
                )
            if bal:
                console.print(f"\nBalanced recommended:\n{bal['ctx']}")

            # --- 8.6 Retrieval validation on the largest stable rung ---
            if retrieval and report.get("maximum_stable_context"):
                from lm_optimizer.services.context_explorer import (
                    build_filler,
                    build_needles,
                    fill_ratios,
                    score_retrieval,
                )

                target_ctx = int(report["maximum_stable_context"])
                ratios = fill_ratios(detailed=detailed_fill)
                # Use the deepest fill ratio that still fits comfortably.
                fill_ratio = max(r for r in ratios if r <= 0.90)
                filler_tokens = max(256, int(target_ctx * fill_ratio))
                filler = build_filler(filler_tokens, kind="mixed", seed=42)
                needles = build_needles(filler)
                if needles:
                    needle = needles[-1]
                    cfg = LoadConfiguration(
                        context_length=target_ctx,
                        gpu_ratio=gpu_ratio,
                        flash_attention=flash,
                        offload_kv_cache_to_gpu=kv_gpu,
                        eval_batch_size=batch,
                    )
                    console.print(
                        f"\n[dim]Retrieval validation @ ctx {target_ctx} "
                        f"(fill {fill_ratio:.0%}, needle at 95%)[/dim]"
                    )
                    ok_load, _i, _l, _ch, _ap, err = await svc._channel_load(model, cfg)
                    if not ok_load:
                        console.print(
                            f"[yellow]Retrieval skipped: load failed ({err[:120]})[/yellow]"
                        )
                    else:
                        prompt = needle.prompt + "\n\n" + needle.question
                        try:
                            response = await client.chat_completion(
                                model=model,
                                input_text=prompt,
                                temperature=0.0,
                                max_output_tokens=64,
                            )
                            text = ""
                            choices = response.get("choices") or []
                            if choices:
                                text = choices[0].get("message", {}).get("content", "")
                            passed = score_retrieval(text, needle.key)
                            console.print(
                                f"Retrieval: "
                                f"{'[green]PASS[/green]' if passed else '[red]FAIL[/red]'} "
                                f"(key {needle.key})"
                            )
                        except Exception as e:  # noqa: BLE001
                            console.print(f"[yellow]Retrieval failed: {e}[/yellow]")
                        finally:
                            try:
                                await svc.ensure_unloaded(model)
                            except Exception:
                                pass

            if quality_rungs > 0:
                if not 0 <= warmup_runs <= 5:
                    console.print("[red]Invalid --warmup-runs (0-5)[/red]")
                    sys.exit(2)
                await _ctx_quality_suites(
                    client, model, report, quality_rungs, quality_threshold,
                    repetitions, gpu_ratio, flash, kv_gpu, batch, style="balanced",
                    warmup_runs=warmup_runs,
                )
        except KeyboardInterrupt:
            console.print(
                "\n[yellow]Interrupted — partial sweep results above are preserved.[/yellow]"
            )
            sys.exit(130)
        except Exception as e:
            logger.exception("ctx failed")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            try:
                await assert_unloaded(client, purpose=f"ctx:{model}")
            except UnloadNotClean as e:
                logger.error("Host not clean after ctx sweep", model=model)
                console.print(f"[red]Done, but host not clean:[/red] {e}")
                try:
                    await client.close()
                except Exception:
                    pass
                sys.exit(1)
            await client.close()

    _acquire_run_lock(f"ctx:{model}")
    try:
        asyncio.run(_ctx())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    finally:
        release_lock()


@app.command(name="quality-ladder")
def quality_ladder(
    run: str = typer.Option(..., "--run", help="Source run ID (prefix accepted) with no quality winner"),
    contexts: str | None = typer.Option(
        None, "--contexts", help="Comma-separated context rungs (default: fine 2048-steps below max)"
    ),
    min_ctx: int = typer.Option(8192, "--min-ctx", help="Lowest context rung"),
    step: int = typer.Option(2048, "--step", help="Rung granularity (fine steps, not power-of-2 only)"),
    threshold: float = typer.Option(0.97, "--threshold", help="Standard quality threshold"),
    fallback_threshold: float = typer.Option(
        0.90, "--fallback-threshold", help="Single lowered-threshold attempt on max-ctx result"
    ),
    repetitions: int = typer.Option(3, "--repetitions", "-r", help="Benchmark repetitions per rung"),
    profile: str = typer.Option("balanced", "--profile", "-p", help="Optimization profile"),
    style: str = typer.Option("balanced", "--style", help="Benchmark style"),
):
    """Context fallback for a quality-failed run: lower ctx until quality passes.

    Re-tests the source run's fastest runtime at progressively lower contexts
    (fine steps, not power-of-2 only). Stops at the first rung passing the
    standard threshold. Then makes exactly ONE lowered-threshold attempt on
    the max-context result: pass or not, no further lowering afterwards.
    Prints both options (max ctx + lower quality vs lower ctx + quality OK)
    so the USER picks. Saves a JSON report next to the other results.
    """
    from lm_optimizer.services.quality_ladder import (
        RungOutcome,
        build_quality_ladder,
        pick_options,
    )

    setup_logging()

    resolved = _resolve_run_id(run)
    if resolved is None:
        # Fall back to DB prefix match.
        candidates = [r for r in run_repo.list_all(200) if str(r.id).startswith(run)]
        if len(candidates) != 1:
            console.print(f"[red]No unique run for prefix {run!r}[/red]")
            sys.exit(1)
        resolved = str(candidates[0].id)
    source = run_repo.get(resolved)
    if not source:
        console.print(f"[red]Run not found: {run}[/red]")
        sys.exit(1)

    passed_cfgs = [c for c in (source.configurations or []) if c.status == ConfigurationStatus.PASSED]
    if not passed_cfgs:
        console.print("[red]Source run has no passed configs to derive a runtime from[/red]")
        sys.exit(1)
    base = max(passed_cfgs, key=lambda c: c.get_avg_generation_tok_s())
    max_ctx = max((c.context_length for c in source.configurations or []), default=0)
    if not max_ctx:
        console.print("[red]Source run has no measured contexts[/red]")
        sys.exit(1)

    explicit = None
    if contexts:
        try:
            explicit = [int(x.strip()) for x in contexts.split(",") if x.strip()]
        except ValueError:
            console.print("[red]Invalid --contexts (comma-separated integers)[/red]")
            sys.exit(2)
    model_limit = None
    try:
        _mi = source.model
        model_limit = getattr(_mi, "context_limit", None)
    except Exception:
        model_limit = None
    ladder = build_quality_ladder(
        max_ctx, min_ctx=min_ctx, step=step, model_limit=model_limit, explicit=explicit
    )
    if not ladder:
        console.print("[red]Empty ladder: check --contexts/--min-ctx/--step[/red]")
        sys.exit(2)

    # Max-context source evidence (already measured, below standard threshold).
    src_q = [c for c in (source.configurations or [])
             if c.context_length == max_ctx and c.quality_score is not None]
    src_best = max(src_q, key=lambda c: c.quality_score.overall) if src_q else None
    source_outcome = RungOutcome(
        context=max_ctx,
        status="quality_failed",
        quality_overall=src_best.quality_score.overall if src_best else None,
        checks=(f"{src_best.quality_score.checks_passed}/{src_best.quality_score.checks_total}"
                if src_best else ""),
        gen_tok_s=src_best.get_avg_generation_tok_s() if src_best else None,
        config_id=str(src_best.id) if src_best else "",
    ) if src_best else None

    console.print(f"[bold]Quality ladder for {source.model.id}[/bold]")
    console.print(f"  Base runtime: fastest passed config {str(base.id)[:8]} "
                  f"({base.get_avg_generation_tok_s():.1f} tok/s @ ctx {base.context_length})")
    console.print(f"  Max ctx (known failed): {max_ctx}")
    console.print(f"  Ladder: {ladder}")
    console.print(f"  Standard threshold: {threshold} | single fallback attempt: {fallback_threshold}")

    async def _ladder():
        client = get_client()
        try:
            snap = await _connect_guarded_or_abort(client, "quality-ladder")
            for line in format_snapshot(snap):
                console.print(f"  {line}")
            hardware = hardware_detector.detect()
            benchmark_service = BenchmarkService(
                client,
                benchmark_config=_benchmark_config_obj(repetitions),
                style=style,
            )
            quality_evaluator = QualityEvaluator(QualityConfig(minimum_score=threshold))
            search_generator = SearchSpaceGenerator(client)
            optimizer = AdaptiveOptimizer(
                client, benchmark_service, quality_evaluator, search_generator
            )
            new_run, _, advanced = await optimizer.prepare_run(
                source.model, hardware, OptimizationProfile(profile),
                quality_threshold=threshold, advanced_settings={}, style=style,
            )
            new_run.stage = OptimizationStage.QUALITY_CHECK
            run_repo.save(new_run)
            console.print(f"  Ladder run: {new_run.id}")

            outcomes: list[RungOutcome] = []
            for ctx_len in ladder:
                data = {k: v for k, v in base.config.to_dict().items()}
                data["context_length"] = ctx_len
                qcfg = LoadConfiguration(
                    **{k: v for k, v in data.items() if hasattr(LoadConfiguration, k)}
                )
                qcfg.context_length = ctx_len
                console.print(f"\n[bold]Rung ctx={ctx_len}[/bold]")
                try:
                    result = await optimizer._test_config(qcfg, ctx_len, score_context=True)
                except KeyboardInterrupt:
                    optimizer.checkpoint_now(reason="interrupted-ladder")
                    console.print("\n[yellow]Interrupted — checkpoint saved.[/yellow]")
                    sys.exit(130)
                if result is None:
                    outcomes.append(RungOutcome(context=ctx_len, status="failed"))
                    continue
                qs = result.quality_score
                outcomes.append(RungOutcome(
                    context=ctx_len,
                    status=result.status.value if hasattr(result.status, "value") else str(result.status),
                    quality_overall=qs.overall if qs else None,
                    checks=(f"{qs.checks_passed}/{qs.checks_total}" if qs else ""),
                    gen_tok_s=result.get_avg_generation_tok_s(),
                    config_id=str(result.id),
                ))
                if result.status == ConfigurationStatus.PASSED:
                    console.print(f"[green]PASS @ ctx={ctx_len} — stopping ladder[/green]")
                    break
            return new_run, outcomes
        finally:
            await client.close()

    _acquire_run_lock(f"quality-ladder:{resolved[:8]}")
    try:
        try:
            new_run, outcomes = asyncio.run(_ladder())
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            sys.exit(130)
    finally:
        release_lock()

    verdict = pick_options(outcomes, source_max_ctx=source_outcome,
                           fallback_threshold=fallback_threshold)

    table = Table(title=f"Quality ladder: {source.model.id}")
    table.add_column("Option", style="cyan")
    table.add_column("Context", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Checks", justify="right")
    table.add_column("Gen tok/s", justify="right")
    if verdict.max_ctx_option:
        m = verdict.max_ctx_option
        table.add_row("A: max context", str(m.context),
                      f"{m.quality_overall:.3f}" if m.quality_overall else "—",
                      m.checks or "—",
                      f"{m.gen_tok_s:.1f}" if m.gen_tok_s else "—")
    if verdict.quality_ok_option:
        q = verdict.quality_ok_option
        table.add_row("B: quality OK", str(q.context),
                      f"{q.quality_overall:.3f}" if q.quality_overall else "—",
                      q.checks or "—",
                      f"{q.gen_tok_s:.1f}" if q.gen_tok_s else "—")
    else:
        table.add_row("B: quality OK", "—", "no rung passed", "—", "—")
    console.print(table)
    console.print(f"\n[bold]Single fallback attempt (max ctx vs {fallback_threshold}):[/bold] "
                  f"{verdict.fallback_message}")

    from datetime import datetime as _dt

    from lm_optimizer.domain.models import RunStatus as _RunStatus

    new_run.status = (
        _RunStatus.PARTIAL_SUCCESS if verdict.quality_ok_option else _RunStatus.FAILED
    )
    new_run.completed_at = _dt.now()
    new_run.error = (
        f"ladder done: {len(verdict.rungs)} rungs, "
        + ("quality OK @ " + str(verdict.quality_ok_option.context)
           if verdict.quality_ok_option else "no rung passed standard threshold")
    )
    run_repo.save(new_run)

    report = {
        "source_run_id": resolved,
        "ladder_run_id": str(new_run.id),
        "model": source.model.id,
        "standard_threshold": threshold,
        "fallback_threshold": fallback_threshold,
        "rungs": [vars(o) for o in verdict.rungs],
        "option_max_ctx": vars(verdict.max_ctx_option) if verdict.max_ctx_option else None,
        "option_quality_ok": vars(verdict.quality_ok_option) if verdict.quality_ok_option else None,
        "fallback_passed": verdict.fallback_passed,
        "fallback_message": verdict.fallback_message,
    }
    out_path = Path("results") / f"{source.model.id.replace('/', '-')}-ladder.json"
    try:
        import orjson as _orjson

        out_path.write_bytes(_orjson.dumps(report, option=_orjson.OPT_INDENT_2))
        console.print(f"\nReport saved: {out_path}")
    except OSError as e:
        console.print(f"[red]Cannot write report: {e}[/red]")
    console.print("\n[dim]YOU pick: A (max context, lower quality) or B (lower context, quality OK).[/dim]")


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run ID to resume from checkpoint"),
    revalidate: bool = typer.Option(False, "--revalidate", help="Re-run completed candidates"),
    style: str = typer.Option("balanced", "--style", help="Benchmark style"),
    skip_phases: str | None = typer.Option(
        None,
        "--skip-phases",
        help="Comma-separated phases to skip for the remainder: speed,quality,recovery,validation,context",
    ),
):
    """Resume an interrupted run from its checkpoint without re-running completed work."""
    setup_logging()
    style = _require_style(style)

    async def _resume():
        client = get_client()
        try:
            await client.connect()
            from lm_optimizer.storage.run_checkpoint import load_checkpoint

            try:
                ckpt = load_checkpoint(run_id)
            except ValueError as e:
                console.print(f"[red]Corrupted checkpoint: {e}[/red]")
                sys.exit(1)
            if not ckpt:
                console.print(f"[red]No checkpoint for run {run_id}[/red]")
                sys.exit(1)
            run = run_repo.get(run_id)
            if not run:
                console.print(f"[red]Run not found: {run_id}[/red]")
                sys.exit(1)
            _apply_resume_skips(run, skip_phases)
            completed = ckpt.get("completed_candidate_ids", [])
            console.print("[bold]Resuming optimization[/bold]")
            console.print(f"Completed:\n{len(completed)} configurations")
            hardware = hardware_detector.detect()
            # Granular runs must resume as granular runs: the flags were
            # persisted into run.advanced_settings at start.
            adv = run.advanced_settings or {}
            benchmark_service = BenchmarkService(
                client,
                style=style,
                include_categories=adv.get("include_categories"),
                speed_only=bool(adv.get("speed_only")),
                quality_only=bool(adv.get("quality_only")),
                speed_context=adv.get("speed_context"),
                mtp_test=bool(adv.get("mtp_test")),
                reasoning_sweep=bool(adv.get("reasoning_sweep")),
                quality_lab=bool(adv.get("quality_lab")),
                test_names=adv.get("test_names") or None,
            )
            quality_evaluator = QualityEvaluator(QualityConfig(minimum_score=run.quality_threshold))
            search_generator = SearchSpaceGenerator(client)
            optimizer = AdaptiveOptimizer(
                client, benchmark_service, quality_evaluator, search_generator
            )
            try:
                resumed = await optimizer.resume_from_checkpoint(
                    run_id, revalidate=revalidate, style=style
                )
            except KeyboardInterrupt:
                optimizer.checkpoint_now(reason="interrupted-resume")
                console.print("\n[yellow]Interrupted — checkpoint saved.[/yellow]")
                sys.exit(130)
            remaining = [
                c for c in resumed.configurations if c.status != ConfigurationStatus.PASSED
            ]
            console.print(
                f"Remaining:\n{len(remaining)} non-passing / "
                f"{len(resumed.configurations)} total tested"
            )
            _display_optimization_result(resumed)
        except SystemExit:
            raise
        except Exception as e:
            logger.exception("Resume failed")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            await client.close()

    _acquire_run_lock(f"resume:{run_id}")
    try:
        asyncio.run(_resume())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    finally:
        release_lock()


@app.command()
def pause(
    run_id: str = typer.Argument(..., help="Run ID to pause (prefix accepted)"),
):
    """Request graceful pause of a running optimization (same machine).

    Writes a pause-flag file the running process consumes at the next safe
    boundary (after the current benchmark/suite/recovery step finishes).
    The run keeps run ID, checkpoint and DB state; resume with `resume`.
    """
    setup_logging()
    from lm_optimizer.storage.run_checkpoint import checkpoint_dir

    resolved = _resolve_run_id(run_id)
    if resolved is None:
        console.print(f"[red]No checkpoint for run prefix {run_id!r}[/red]")
        sys.exit(1)
    from datetime import datetime as _dt
    from pathlib import Path as _Path

    import orjson

    flag = _Path(str(checkpoint_dir())) / f"pause_{resolved}.flag"
    try:
        payload = {
            "requested_at": _dt.now().isoformat(timespec="seconds"),
            "reason": "user_requested",
        }
        flag.write_bytes(orjson.dumps(payload))
    except OSError as e:
        console.print(f"[red]Cannot write pause flag: {e}[/red]")
        sys.exit(1)
    console.print(f"[yellow]Pause requested for run {resolved}[/yellow]")
    console.print("The running process pauses after its current operation finishes.")
    console.print(f"Check status with: run-status {resolved}")


def _apply_resume_skips(run, skip_phases: str | None) -> None:
    """Merge --skip-phases into a stored run before resume (CLI wins).

    Exits 2 on unknown phase names. No-op when skip_phases is falsy.
    """
    # Guard against typer OptionInfo leaking in when resume() is called
    # directly (e.g. from optimize --resume) instead of via the CLI.
    if isinstance(skip_phases, typer.models.OptionInfo):
        skip_phases = getattr(skip_phases, "default", None)
    if not skip_phases or not isinstance(skip_phases, str):
        return
    from lm_optimizer.services.optimizer import parse_phase_skips

    try:
        skips = sorted(parse_phase_skips(skip_phases))
        merged = dict(run.advanced_settings or {})
        merged["skip_phases"] = skips
        run.advanced_settings = merged
        params = dict(run.benchmark_params or {})
        params["skip_phases"] = skips
        run.benchmark_params = params
        run_repo.save(run)
        console.print(
            f"[yellow]Skipping phases for the remainder: "
            f"{', '.join(skips)}[/yellow]"
        )
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(2)


def _resolve_run_id(prefix: str) -> str | None:
    """Full run ID from exact ID or unique checkpoint-file prefix."""
    from pathlib import Path as _Path

    from lm_optimizer.storage.run_checkpoint import checkpoint_dir, load_checkpoint

    try:
        if load_checkpoint(prefix) is not None:
            return prefix
    except Exception:
        pass
    try:
        matches = sorted(
            _Path(str(checkpoint_dir())).glob(f"run_{prefix}*.json"), key=lambda p: p.name
        )
        if len(matches) == 1:
            stem = matches[0].stem
            return stem[4:] if stem.startswith("run_") else stem
    except Exception:
        pass
    return None


@app.command(name="run-status")
def run_status(
    run_id: str = typer.Argument(..., help="Run ID to inspect (prefix accepted)"),
):
    """Show run state: PAUSED/phase/stage/progress and whether resume is possible."""
    setup_logging()
    from lm_optimizer.storage.run_checkpoint import load_checkpoint

    resolved = _resolve_run_id(run_id)
    if resolved is None:
        console.print(f"[red]No checkpoint for run prefix {run_id!r}[/red]")
        sys.exit(1)
    run = run_repo.get(resolved)
    if run is None:
        console.print(f"[red]Run not found: {run_id}[/red]")
        sys.exit(1)
    ckpt = load_checkpoint(resolved) or {}
    pab = ((run.benchmark_params or {}).get("phase_ab") or {})
    pause_info = pab.get("pause") or {}
    tested = list(run.configurations or [])
    table = Table(title=f"Run {resolved}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Status", str(run.status.value if hasattr(run.status, "value") else run.status))
    table.add_row("Phase", str(pab.get("phase", run.phase if hasattr(run, "phase") else "?")))
    table.add_row("Stage", str(run.stage.value if hasattr(run.stage, "value") else run.stage))
    table.add_row("Tested", str(len(tested)))
    if pause_info:
        table.add_row("Paused at", str(pause_info.get("paused_at", "?")))
        table.add_row("Pause reason", str(pause_info.get("reason", "?")))
        table.add_row("Resume possible", "YES (same run ID)")
    else:
        table.add_row("Resume possible", "YES (same run ID)" if ckpt else "unknown")
    console.print(table)


@app.command()
def checkpoints():
    """List interrupt checkpoints available for resume."""
    setup_logging()
    from lm_optimizer.storage.run_checkpoint import list_checkpoints as _list

    rows = _list()
    if not rows:
        console.print("[yellow]No checkpoints found[/yellow]")
        return
    table = Table(title="Checkpoints")
    table.add_column("Run ID", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Stage", style="yellow")
    table.add_column("Completed", justify="right")
    table.add_column("Saved", style="dim")
    for r in rows:
        table.add_row(
            str(r.get("run_id", "-")),
            str(r.get("model", "-")),
            str(r.get("stage", "-")),
            str(r.get("completed", "?")),
            str(r.get("saved_at", "-")),
        )
    console.print(table)


@app.command(name="reconcile-runs")
def reconcile_runs(
    apply: bool = typer.Option(
        False, "--apply", help="Actually mark stale runs interrupted (default: preview only)"
    ),
):
    """Preview/apply reconciliation of stale `running` runs (keeps history)."""
    setup_logging()
    from lm_optimizer.services.run_reconcile import reconcile_stale_runs

    result = reconcile_stale_runs(dry_run=not apply)
    changes = result["changes"]
    if not changes:
        console.print("[green]No stale running runs found.[/green]")
        return
    table = Table(title=("Stale runs to interrupt" if not apply else "Stale runs interrupted"))
    table.add_column("Run ID", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("From", style="yellow")
    table.add_column("To", style="magenta")
    for c in changes:
        table.add_row(str(c["id"]), str(c.get("model_id") or "-"), c["from"], c["to"])
    console.print(table)
    if not apply:
        console.print(
            f"[yellow]Dry run: {len(changes)} stale run(s). "
            "Re-run with --apply to mark them interrupted (history preserved).[/yellow]"
        )
    else:
        console.print(f"[green]Reconciled {result['changed']} stale run(s).[/green]")


@app.command(name="resume-menu")
def resume_menu(
    status: list[str] | None = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status (running, paused, resumed, partial_success). Can repeat.",
    ),
    limit: int = typer.Option(20, "--limit", "-l", help="Max runs to show"),
    all: bool = typer.Option(
        False, "--all", help="Resume every listed run sequentially (one confirmation)"
    ),
):
    """Interactive menu to resume paused/incomplete runs.

    Shows runs with checkpoints that can be resumed (not terminal states).
    Select a run by number to resume from its checkpoint, or pass --all
    to finish every listed run one after another.
    """
    setup_logging()

    # Default to resumable states
    target_statuses = set(status) if status else {"running", "paused", "resumed", "partial_success"}

    runs = run_repo.list_all(limit=limit)
    if not runs:
        console.print("[yellow]No optimization runs found[/yellow]")
        return

    # Filter to resumable states
    resumable_runs = [r for r in runs if r.status.value in target_statuses]
    if not resumable_runs:
        console.print(f"[yellow]No runs with status in {target_statuses}[/yellow]")
        return

    if all:
        resume_all_runs(resumable_runs)
        return

    # Build display table
    table = Table(title="Resumable Runs (select by number)")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Run ID", style="dim")
    table.add_column("Model", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Profile", style="blue")
    table.add_column("Stage", style="magenta")
    table.add_column("Configs", justify="right")
    table.add_column("Completed", justify="right")
    table.add_column("Checkpoint", style="dim")
    table.add_column("Phase/Cursor", style="cyan")
    table.add_column("Last progress", style="white")
    table.add_column("Resume hint", style="green")
    table.add_column("Created", style="dim")

    for idx, run in enumerate(resumable_runs, 1):
        # Get checkpoint info
        from lm_optimizer.storage.run_checkpoint import load_checkpoint
        try:
            ckpt = load_checkpoint(str(run.id))
        except ValueError:
            ckpt = {"corrupted": True}
        completed = len(ckpt.get("completed_candidate_ids", [])) if ckpt else 0
        total = run.config_count or len(run.configurations) if run.configurations else 0

        pab = ((ckpt or {}).get("phase_ab") or {})
        _ph = getattr(run, "phase", "?")
        _ph = _ph.value if hasattr(_ph, "value") else _ph
        phase = pab.get("phase") or _ph or "?"
        step = (pab.get("recovery_cursor") or {}).get("step", 0) or 0
        events = (ckpt or {}).get("event_log") or []
        last_progress = ""
        if events:
            last = events[-1] or {}
            last_progress = f"{last.get('message', '')} {last.get('detail', '')}".strip()[:60]
        if not ckpt:
            ckpt_cell = "no"
            hint = "no checkpoint"
        elif ckpt.get("corrupted"):
            ckpt_cell = "corrupt"
            hint = "checkpoint unreadable"
        else:
            ckpt_cell = f"yes {str(ckpt.get('saved_at', ''))[:16]}"
            hint = "safe to resume"

        status_badge = _status_badge(run.status.value)
        table.add_row(
            str(idx),
            str(run.id)[:8] + "...",
            run.model.name or run.model.id,
            status_badge,
            run.profile.value,
            run.stage.value if run.stage else "?",
            str(total),
            str(completed),
            ckpt_cell,
            f"{phase}@{step}",
            last_progress or "-",
            hint,
            run.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)
    console.print()

    # Prompt for selection
    try:
        choice = typer.prompt(
            "Select run number to resume (or Enter to cancel)",
            type=int,
            default=0,
            show_default=False,
        )
    except (typer.Abort, ValueError):
        console.print("[yellow]Cancelled[/yellow]")
        return

    if choice <= 0 or choice > len(resumable_runs):
        console.print("[red]Invalid selection[/red]")
        return

    selected = resumable_runs[choice - 1]

    # Confirm
    console.print(f"\nSelected: [bold]{selected.model.name}[/bold] (run {selected.id})")
    console.print(f"Status: {_status_badge(selected.status.value)} | Stage: {selected.stage}")
    if not typer.confirm("Resume this run?", default=True):
        console.print("[yellow]Cancelled[/yellow]")
        return

    # Run resume (reuse existing logic)
    _run_resume(str(selected.id), revalidate=False, style="balanced")


def resume_all_runs(runs: list) -> dict:
    """Resume every given run sequentially (bulk finish, --all path).

    One confirmation up front; per-run failures are collected, never abort
    the batch. Returns {"resumed": [...ids], "failed": {id: error}}.
    """
    result: dict = {"resumed": [], "failed": {}}
    if not runs:
        console.print("[yellow]Nothing to resume[/yellow]")
        return result
    console.print(f"[bold]About to resume {len(runs)} run(s), one after another.[/bold]")
    try:
        go = typer.confirm("Resume all?", default=True)
    except (typer.Abort, ValueError):
        console.print("[yellow]Cancelled[/yellow]")
        return result
    if not go:
        console.print("[yellow]Cancelled[/yellow]")
        return result
    for run in runs:
        rid = str(run.id)
        try:
            console.print(f"\n[bold]=== {run.model.name or run.model.id} ({rid[:8]}...) ===[/bold]")
            _run_resume(rid, revalidate=False, style="balanced")
            result["resumed"].append(rid)
        except SystemExit as e:
            # _run_resume exits nonzero on failure; keep going with the rest.
            if getattr(e, "code", 1) in (0, None):
                result["resumed"].append(rid)
            else:
                result["failed"][rid] = f"exit {e.code}"
        except Exception as e:
            result["failed"][rid] = f"{type(e).__name__}: {e}"[:200]
    console.print(
        f"\n[bold]Resume-all done: {len(result['resumed'])} resumed, "
        f"{len(result['failed'])} failed[/bold]"
    )
    for rid, err in result["failed"].items():
        console.print(f"  [red]{rid[:8]}...: {err}[/red]")
    return result


def _run_resume(run_id: str, revalidate: bool, style: str, skip_phases: str | None = None):
    """Internal: run the resume logic."""
    style = _require_style(style)

    async def _resume():
        client = get_client()
        try:
            await client.connect()
            from lm_optimizer.storage.run_checkpoint import load_checkpoint

            try:
                ckpt = load_checkpoint(run_id)
            except ValueError as e:
                console.print(f"[red]Corrupted checkpoint: {e}[/red]")
                sys.exit(1)
            if not ckpt:
                console.print(f"[red]No checkpoint for run {run_id}[/red]")
                sys.exit(1)
            run = run_repo.get(run_id)
            if not run:
                console.print(f"[red]Run not found: {run_id}[/red]")
                sys.exit(1)
            _apply_resume_skips(run, skip_phases)
            completed = ckpt.get("completed_candidate_ids", [])
            console.print("[bold]Resuming optimization[/bold]")
            console.print(f"Completed:\n{len(completed)} configurations")
            hardware = hardware_detector.detect()
            # Granular runs must resume as granular runs: the flags were
            # persisted into run.advanced_settings at start.
            adv = run.advanced_settings or {}
            benchmark_service = BenchmarkService(
                client,
                style=style,
                include_categories=adv.get("include_categories"),
                speed_only=bool(adv.get("speed_only")),
                quality_only=bool(adv.get("quality_only")),
                speed_context=adv.get("speed_context"),
                mtp_test=bool(adv.get("mtp_test")),
                reasoning_sweep=bool(adv.get("reasoning_sweep")),
                quality_lab=bool(adv.get("quality_lab")),
                test_names=adv.get("test_names") or None,
            )
            quality_evaluator = QualityEvaluator(QualityConfig(minimum_score=run.quality_threshold))
            search_generator = SearchSpaceGenerator(client)
            optimizer = AdaptiveOptimizer(
                client, benchmark_service, quality_evaluator, search_generator
            )
            try:
                resumed = await optimizer.resume_from_checkpoint(
                    run_id, revalidate=revalidate, style=style
                )
            except KeyboardInterrupt:
                optimizer.checkpoint_now(reason="interrupted-resume")
                console.print("\n[yellow]Interrupted — checkpoint saved.[/yellow]")
                sys.exit(130)
            remaining = [
                c for c in resumed.configurations if c.status != ConfigurationStatus.PASSED
            ]
            console.print(
                f"Remaining:\n{len(remaining)} non-passing / "
                f"{len(resumed.configurations)} total tested"
            )
            _display_optimization_result(resumed)
        except SystemExit:
            raise
        except Exception as e:
            logger.exception("Resume failed")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            await client.close()

    _acquire_run_lock(f"resume:{run_id}")
    try:
        asyncio.run(_resume())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    finally:
        release_lock()


def _status_badge(status: str) -> str:
    """Return styled status badge."""
    badges = {
        "running": "[green]running[/green]",
        "paused": "[yellow]paused[/yellow]",
        "resumed": "[blue]resumed[/blue]",
        "partial_success": "[magenta]partial_success[/magenta]",
        "success": "[green]success[/green]",
        "failed": "[red]failed[/red]",
        "cancelled": "[dim]cancelled[/dim]",
        "interrupted": "[dim]interrupted[/dim]",
    }
    return badges.get(status, status)


@app.command(name="judge")
def judge_models(
    run_id: str = typer.Argument(..., help="Run ID to evaluate"),
    priority: str = typer.Option("balanced", "--priority", "-p", help="Priority: speed, quality, balanced, context"),
    use_case: str = typer.Option("general", "--use-case", "-u", help="Use case: coding, chat, reasoning, general"),
    vram_limit: float | None = typer.Option(None, "--vram-limit", help="VRAM limit in GB (optional)"),
    backend: str = typer.Option("heuristic", "--backend", "-b", help="Judge backend: heuristic, local, api"),
    model: str | None = typer.Option(None, "--model", "-m", help="Judge model (for local/api backend)"),
):
    """AI-assisted model selection - get a second opinion on which config to keep.

    The USER decides first. This provides analysis to help make informed decisions.
    """
    from lm_optimizer.services.model_judge import JudgeConfig, JudgeBackend, create_judge, ModelComparison

    setup_logging()

    run = run_repo.get(run_id)
    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]")
        sys.exit(1)

    configs = run.configurations or []
    passed = [c for c in configs if c.status == "passed"]
    if not passed:
        console.print("[red]No passing configurations to evaluate[/red]")
        sys.exit(1)

    # Hardware constraints
    hw = hardware_detector.detect()
    hardware_constraints = {
        "vram_gb": vram_limit or (hw.gpu.vram_gb if hw.gpu else None),
        "ram_gb": hw.total_ram_gb,
    }

    # Create judge
    judge_config = JudgeConfig(
        backend=JudgeBackend(backend),
        model=model,
    )
    judge = create_judge(judge_config)

    # Run evaluation
    async def _evaluate():
        comparison = ModelComparison(
            run=run,
            configs=configs,
            use_case=use_case,
            hardware_constraints=hardware_constraints,
            priority=priority,
        )
        return await judge.evaluate(comparison)

    try:
        verdict = asyncio.run(_evaluate())
    except Exception as e:
        console.print(f"[red]Judge evaluation failed: {e}[/red]")
        sys.exit(1)
    finally:
        asyncio.run(judge.close())

    # Display results
    console.print(Panel.fit(f"[bold]Model Judge Verdict[/bold] — Run: {run_id[:8]}..."))

    # Find recommended config
    rec_config = next((c for c in configs if c.id == verdict.recommended_config_id), None)
    if rec_config:
        avg_gen = rec_config.get_avg_generation_tok_s()
        avg_prompt = rec_config.get_avg_prompt_tok_s()
        table = Table(title="Recommended Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Config ID", str(rec_config.id)[:8] + "...")
        table.add_row("Context", str(rec_config.context_length))
        table.add_row("Speed", f"{avg_gen:.1f} tok/s" if avg_gen else "N/A")
        table.add_row("Prompt Speed", f"{avg_prompt:.0f} tok/s" if avg_prompt else "N/A")
        qual = rec_config.quality_score.overall if rec_config.quality_score else None
        table.add_row("Quality", f"{qual:.3f}" if qual else "N/A")
        table.add_row("VRAM", f"{rec_config.peak_vram_gb:.1f} GB" if rec_config.peak_vram_gb else "N/A")
        table.add_row("Score", f"{rec_config.score:.3f}" if rec_config.score else "N/A")
        console.print(table)

    console.print(f"\n[bold]Reasoning:[/bold] {verdict.reasoning}")
    console.print(f"[bold]Confidence:[/bold] {verdict.confidence:.0%}")

    if verdict.pros:
        console.print("\n[bold green]Pros:[/bold green]")
        for pro in verdict.pros:
            console.print(f"  ✓ {pro}")

    if verdict.cons:
        console.print("\n[bold red]Cons / Warnings:[/bold red]")
        for con in verdict.cons:
            console.print(f"  ✗ {con}")

    if verdict.alternative_config_ids:
        console.print("\n[bold]Alternatives:[/bold]")
        for alt_id in verdict.alternative_config_ids:
            alt = next((c for c in configs if c.id == alt_id), None)
            if alt:
                avg_gen = alt.get_avg_generation_tok_s()
                qual = alt.quality_score.overall if alt.quality_score else None
                aid = str(alt.id)[:8]
                console.print(f"  • {aid}... — {alt.context_length} ctx, {avg_gen:.1f} tok/s, Q:{qual:.3f}" if avg_gen and qual else f"  • {aid}...")

    console.print("\n[dim]Remember: YOU decide. This is just a second opinion.[/dim]")


@app.command(name="param-matrix")
def param_matrix(
    record: bool = typer.Option(False, "--record", help="Save capability snapshot to DB"),
) -> None:
    """Print the parameter registry matrix (control method + verification)."""
    setup_logging()
    from lm_optimizer.services.parameter_registry import matrix_rows

    table = Table(title="Parameter control matrix")
    table.add_column("Parameter", style="cyan")
    table.add_column("Backend", style="yellow")
    table.add_column("Phase", justify="center")
    table.add_column("Control", style="green")
    table.add_column("Verified", justify="center")
    table.add_column("Lifecycle", style="magenta")
    table.add_column("Optimizer", justify="center")
    for row in matrix_rows():
        table.add_row(
            row["name"],
            row["backend"],
            row["phase"],
            row["control"],
            row["verified"],
            row["lifecycle"],
            row["optimizer"],
        )
    console.print(table)
    console.print(
        "[dim]Control: REST = native API verified live; CLI = lms flags verified; "
        "SDK/schema = known but programmatically unverified; none = not exposed.[/dim]"
    )
    console.print(
        "[dim]What you get from this table: which knobs the optimizer can turn "
        "itself during a run (and how), vs what stays fixed or needs manual/CLI steps.[/dim]"
    )
    if record:
        try:
            ver = _version_info()
            snap_id = capability_repo.save(None, ver["lms"], ver["backend"])
            console.print(f"[green]Capability snapshot #{snap_id} recorded[/green]")
        except Exception as e:
            console.print(f"[red]Snapshot failed: {e}[/red]")
            sys.exit(1)


def _display_benchmark_result(result, model_id: str):
    """Display benchmark results (ASCII only for Windows console)."""
    console.print(Panel.fit(f"[bold]Benchmark Result: {model_id}[/bold]"))

    cfg = result.config
    table = Table(title="Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Context Length", str(result.context_length))
    table.add_row(
        "GPU Ratio",
        f"{cfg.gpu_ratio:.2f} (not sent via REST)" if cfg.gpu_ratio else "Auto",
    )
    table.add_row("Flash Attention", "Enabled" if cfg.flash_attention else "Disabled")
    table.add_row("KV Cache GPU", "Yes" if cfg.offload_kv_cache_to_gpu else "No")
    table.add_row("Eval Batch Size", str(cfg.eval_batch_size) if cfg.eval_batch_size else "Auto")
    table.add_row(
        "Physical Batch Size",
        str(cfg.physical_batch_size) if cfg.physical_batch_size else "Server default (512)",
    )
    table.add_row("Parallel", str(cfg.parallel) if cfg.parallel else "Server default (4)")
    table.add_row(
        "Checkpoints",
        str(cfg.context_checkpoints)
        if cfg.context_checkpoints is not None
        else "Server default (32)",
    )
    if result.generation:
        table.add_row("Style", str(result.generation.get("style", "balanced")))

    console.print(table)

    # Per-test metrics - TTFT is estimated when streaming not available
    metrics_table = Table(title="Per-Test Metrics (Est. TTFT)")
    metrics_table.add_column("Test", style="cyan")
    metrics_table.add_column("Category", style="yellow")
    metrics_table.add_column("Success", justify="center")
    metrics_table.add_column("Gen tok/s", justify="right")
    metrics_table.add_column("Prompt tok/s", justify="right")
    metrics_table.add_column("Est. TTFT (ms)", justify="right")
    metrics_table.add_column("Tokens", justify="right")

    for m in result.metrics:
        # Use estimated_ttft_ms, with fallback to ttft_ms for compat
        ettft = getattr(m, "estimated_ttft_ms", getattr(m, "ttft_ms", 0))
        metrics_table.add_row(
            m.test_name,
            m.category,
            "PASS" if m.success else "FAIL",
            f"{m.generation_tok_s:.1f}" if m.success else "N/A",
            f"{m.prompt_tok_s:.0f}" if m.success else "N/A",
            f"{ettft:.0f}" if m.success else "N/A",
            f"{m.total_tokens}" if m.success else "N/A",
        )

    console.print(metrics_table)

    # Summary - quality is heuristic checks
    summary_table = Table(title="Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Avg Generation", f"{result.get_avg_generation_tok_s():.1f} tok/s")
    summary_table.add_row("Avg Prompt", f"{result.get_avg_prompt_tok_s():.0f} tok/s")
    summary_table.add_row(
        "Avg Est. TTFT", f"{result.get_avg_estimated_ttft_ms():.0f} ms (estimated, no streaming)"
    )
    if result.quality_score:
        qs = result.quality_score
        checks = (
            f"{qs.checks_passed}/{qs.checks_total} checks passed"
            if hasattr(qs, "checks_passed")
            else ""
        )
        summary_table.add_row(
            "Correctness / Quality", f"{qs.overall:.3f} ({checks}) - heuristic checks"
        )
    else:
        summary_table.add_row("Correctness / Quality", "N/A")
    summary_table.add_row("Stability Score", f"{result.stability_score:.3f}")
    summary_table.add_row(
        "Status",
        "Passed" if result.status == ConfigurationStatus.PASSED else f"Failed ({result.error})",
    )

    console.print(summary_table)


def _display_optimization_result(run):
    """Display optimization results with baseline comparison and score breakdown.

    Zero-pass runs show an explicit failure state (never a fake N/A winner and
    never an Apply offer). Partial runs are labeled and only PASSED configs
    may win. Successful runs show exactly ONE primary + ALTERNATIVES crowns.
    """
    from lm_optimizer.services.run_summary import (
        alternatives,
        classify_run,
        failure_reason_text,
        why_text,
    )

    state, info = classify_run(list(run.configurations), run)
    if state == "FAILED":
        console.print(Panel.fit("[bold red]Optimization unsuccessful[/bold red]"))
        console.print("No valid configurations were successfully benchmarked.")
        console.print(f"\nReason:\n{failure_reason_text(run)}")
        fb = info.get("failure_classes", {})
        table = Table(title="Failures")
        table.add_column("Class", style="cyan")
        table.add_column("Count", justify="right")
        for k in ("load_failures", "oom", "timeouts", "quality_rejected", "unsupported_parameters"):
            table.add_row(k, str(fb.get(k, 0)))
        console.print(table)
        base = run.get_baseline_config()
        if base is not None:
            console.print(
                f"Last successful baseline: ctx {base.context_length} "
                f"{base.get_avg_generation_tok_s():.1f} tok/s"
            )
        else:
            console.print("[dim]No successful baseline available.[/dim]")
        return  # never offer Apply

    from lm_optimizer.services.run_summary import final_recommendation

    best = final_recommendation(list(run.configurations), run)
    if best is None:
        console.print("[yellow]No successful configuration found[/yellow]")
        return
    bc = best.config
    if state == "NO_WINNER":
        console.print(
            Panel.fit(
                "[bold yellow]Optimization completed — no quality-validated winner[/bold yellow]"
            )
        )
        console.print(
            "The model generated usable output, but no configuration met the "
            f"quality threshold ({run.quality_threshold}). Best diagnostic candidate below."
        )
    else:
        console.print(Panel.fit(f"[bold green]Optimization Complete: {run.model.id}[/bold green]"))
    console.print(f"[dim]Run state: {state} (status={run.status.value})[/dim]")

    # ---- FINAL RECOMMENDATION: exactly one primary ----
    try:
        gen = _generation_for(run.model.id, getattr(run.model, "architecture", None))
    except Exception:
        gen = {}
    rec = Table(title="FINAL RECOMMENDATION")
    rec.add_column("Field", style="cyan")
    rec.add_column("Value", style="green")
    rec.add_row("Model", str(run.model.id))
    rec.add_row("Profile", str(run.profile.value))
    rec.add_row("Context", str(best.context_length))
    rec.add_row("GPU Offload", f"{bc.gpu_ratio:.2f}" if bc.gpu_ratio is not None else "Auto")
    rec.add_row("KV Cache", "GPU" if bc.offload_kv_cache_to_gpu else "CPU")
    rec.add_row("Flash Attention", "Enabled" if bc.flash_attention else "Disabled")
    rec.add_row("Eval Batch", str(bc.eval_batch_size) if bc.eval_batch_size else "Auto")
    rec.add_row("CPU Threads", f"{run.hardware.cpu_cores_logical} recommended (REST cannot set)")
    rec.add_row("Generation", f"temperature={gen.get('temperature', 'suite default')}")
    rec.add_row("Prefill", f"{best.get_avg_prompt_tok_s():.0f} tok/s")
    rec.add_row("Estimated TTFT", f"{best.get_avg_estimated_ttft_ms():.0f} ms (estimated)")
    rec.add_row("VRAM", f"{best.peak_vram_gb:.1f} GB" if best.peak_vram_gb else "N/A")
    rec.add_row("RAM", f"{best.peak_ram_gb:.1f} GB" if best.peak_ram_gb else "N/A")
    if best.quality_score:
        rec.add_row(
            "Correctness",
            f"{best.quality_score.overall:.3f} "
            f"({best.quality_score.checks_passed}/{best.quality_score.checks_total} checks)",
        )
    else:
        rec.add_row("Correctness", "N/A")
    rec.add_row("Stability", f"{best.stability_score:.3f}")
    if run.baseline_metrics and run.baseline_metrics.get("generation_tok_s"):
        try:
            base = run.baseline_metrics["generation_tok_s"]
            cur = best.get_avg_generation_tok_s()
            pct = (cur - base) / base * 100 if base else 0
            rec.add_row("Baseline improvement", f"{pct:+.1f}% ({base:.1f} -> {cur:.1f} tok/s)")
        except Exception:
            rec.add_row("Baseline improvement", "N/A")
    else:
        rec.add_row("Baseline improvement", "N/A (no baseline)")
    rec.add_row("Why", why_text(best, run.baseline_metrics, run.profile.value))
    try:
        from lm_optimizer.services.reporting import _identical_configs, _median

        _same = _identical_configs(list(run.configurations), best)
        if len(_same) > 1:
            _med = _median([c.get_avg_generation_tok_s() for c in _same])
            rec.add_row(
                "Typical (identical runs)",
                f"{_med:.1f} tok/s median of {len(_same)} "
                f"(stored repeat {best.get_avg_generation_tok_s():.1f})",
            )
    except Exception:
        pass
    console.print(rec)

    # ---- PHASE A/B transparency: raw fastest (even if rejected) ----
    phase_table = _phase_transparency_table(run, best)
    if phase_table is not None:
        console.print(phase_table)

    # ---- ALTERNATIVES crowns ----
    crowns = alternatives(list(run.configurations), best)
    if crowns:
        alt = Table(title="ALTERNATIVES")
        alt.add_column("Crown", style="cyan")
        alt.add_column("Context", justify="right")
        alt.add_column("Gen tok/s", justify="right")
        alt.add_column("Quality", justify="right")
        alt.add_column("VRAM", justify="right")
        for label, key in (
            ("Best Speed", "best_speed"),
            ("Best Context", "best_context"),
            ("Best Quality", "best_quality"),
            ("Best Memory", "best_memory"),
        ):
            c = crowns.get(key)
            if c is None:
                continue
            q = f"{c.quality_score.overall:.3f}" if c.quality_score else "N/A"
            alt.add_row(
                label,
                str(c.context_length),
                f"{c.get_avg_generation_tok_s():.1f}",
                q,
                f"{c.peak_vram_gb:.1f}" if c.peak_vram_gb else "N/A",
            )
        console.print(alt)

    # Winner score line (FINAL RECOMMENDATION above is authoritative).
    console.print(
        f"[dim]Primary winner score={best.score:.3f}; "
        "failed configurations never enter the winner set.[/dim]"
    )
    try:
        from lm_optimizer.services.run_summary import speed_range as _speed_range

        _range = _speed_range(list(run.configurations))
        if _range is not None:
            _fastest_cfg, _slowest_cfg = _range
            console.print(
                "[dim]Run speed range (passed configs): "
                f"best {_fastest_cfg.get_avg_generation_tok_s():.1f} tok/s "
                f"(ctx {_fastest_cfg.context_length}) / worst "
                f"{_slowest_cfg.get_avg_generation_tok_s():.1f} tok/s "
                f"(ctx {_slowest_cfg.context_length}).[/dim]"
            )
    except Exception:
        pass

    # Metrics with baseline comparison (percentage changes from actual measurements)
    metrics_table = Table(title="Performance Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Optimized", style="green")
    if run.baseline_metrics:
        metrics_table.add_column("Baseline", style="yellow")
        metrics_table.add_column("Change", style="magenta")

    def fmt_change(new, old):
        if old and old != 0:
            pct = (new - old) / old * 100
            sign = "+" if pct >= 0 else ""
            return f"{sign}{pct:.1f}%"
        return "-"

    # Gather baseline if available
    baseline = run.baseline_metrics or {}
    if baseline:
        metrics_table.add_row(
            "Generation Speed",
            f"{best.get_avg_generation_tok_s():.1f} tok/s",
            f"{baseline.get('generation_tok_s', 0):.1f} tok/s",
            fmt_change(best.get_avg_generation_tok_s(), baseline.get("generation_tok_s")),
        )
        metrics_table.add_row(
            "Prompt Processing",
            f"{best.get_avg_prompt_tok_s():.0f} tok/s",
            f"{baseline.get('prompt_tok_s', 0):.0f} tok/s",
            fmt_change(best.get_avg_prompt_tok_s(), baseline.get("prompt_tok_s")),
        )
        metrics_table.add_row(
            "Est. TTFT",
            f"{best.get_avg_estimated_ttft_ms():.0f} ms",
            f"{baseline.get('estimated_ttft_ms', 0):.0f} ms",
            fmt_change(best.get_avg_estimated_ttft_ms(), baseline.get("estimated_ttft_ms")),
        )
        if best.quality_score:
            qs = best.quality_score
            checks = (
                f"{qs.checks_passed}/{qs.checks_total} checks"
                if hasattr(qs, "checks_passed")
                else ""
            )
            baseline_q = baseline.get("quality_overall", 0)
            metrics_table.add_row(
                "Correctness / Quality",
                f"{qs.overall:.3f} ({checks})",
                f"{baseline_q:.3f}" if baseline_q else "N/A",
                fmt_change(qs.overall, baseline_q),
            )
        else:
            metrics_table.add_row("Correctness / Quality", "N/A", "N/A", "-")
        metrics_table.add_row(
            "Stability Score",
            f"{best.stability_score:.3f}",
            f"{baseline.get('stability', 0):.3f}" if baseline.get("stability") else "N/A",
            "-",
        )
        metrics_table.add_row(
            "Context Length",
            str(best.context_length),
            str(baseline.get("context_length", "N/A")),
            fmt_change(best.context_length, baseline.get("context_length")),
        )
        metrics_table.add_row(
            "VRAM",
            f"{best.peak_vram_gb:.1f} GB" if best.peak_vram_gb else "not measured",
            f"{baseline.get('peak_vram_gb', 0):.1f} GB" if baseline.get("peak_vram_gb") else "-",
            fmt_change(best.peak_vram_gb or 0, baseline.get("peak_vram_gb") or 0),
        )
        metrics_table.add_row(
            "RAM",
            f"{best.peak_ram_gb:.1f} GB" if best.peak_ram_gb else "not measured",
            "-",
            "-",
        )
        metrics_table.add_row("Optimization Time", f"{run.duration_seconds:.1f} s", "-", "-")
    else:
        metrics_table.add_row("Generation Speed", f"{best.get_avg_generation_tok_s():.1f} tok/s")
        metrics_table.add_row("Prompt Processing", f"{best.get_avg_prompt_tok_s():.0f} tok/s")
        metrics_table.add_row("Est. TTFT", f"{best.get_avg_estimated_ttft_ms():.0f} ms (estimated)")
        if best.quality_score:
            qs = best.quality_score
            checks = (
                f"{qs.checks_passed}/{qs.checks_total} checks passed"
                if hasattr(qs, "checks_passed")
                else ""
            )
            metrics_table.add_row(
                "Correctness / Quality",
                f"{qs.overall:.3f} ({checks}, heuristic checks)",
            )
        metrics_table.add_row("Stability Score", f"{best.stability_score:.3f}")
        metrics_table.add_row("Context Length", str(best.context_length))
        metrics_table.add_row(
            "VRAM", f"{best.peak_vram_gb:.1f} GB" if best.peak_vram_gb else "not measured"
        )
        metrics_table.add_row(
            "RAM", f"{best.peak_ram_gb:.1f} GB" if best.peak_ram_gb else "not measured"
        )
        metrics_table.add_row("Optimization Time", f"{run.duration_seconds:.1f} s")

    console.print(metrics_table)

    # Score breakdown explainability
    if best.score_breakdown:
        console.print("\n[bold]Score Breakdown (why this config won):[/bold]")
        bd_table = Table(title=f"{run.profile.value.capitalize()} score: {best.score:.3f}")
        bd_table.add_column("Component", style="cyan")
        bd_table.add_column("Normalized", justify="right")
        bd_table.add_column("Weight", justify="right")
        bd_table.add_column("Contribution", justify="right")
        # Actual weights used for this config's phase (Phase A zeroes
        # context); profile defaults only as fallback for legacy data.
        weights = (best.generation or {}).get("score_weights") or (
            run.profile_weights if hasattr(run, "profile_weights") else {}
        )
        for comp, norm in best.score_breakdown.items():
            w = weights.get(comp, 0)
            contrib = norm * w
            bd_table.add_row(comp, f"{norm:.3f}", f"{w:.2f}", f"{contrib:.3f}")
        console.print(bd_table)
        console.print(
            "[dim]Normalization: speeds/TTFT run-relative, context model-relative, memory hardware-relative. See docs/OPTIMIZATION_METHOD.md[/dim]"
        )

    # Pareto frontier
    pareto = run.get_pareto_configs()
    if pareto:
        console.print("\n[bold]Pareto Frontier:[/bold]")
        pf_table = Table()
        pf_table.add_column("Profile", style="cyan")
        pf_table.add_column("Context", justify="right")
        pf_table.add_column("GPU", justify="right")
        pf_table.add_column("KV", style="yellow")
        pf_table.add_column("Batch", justify="right")
        pf_table.add_column("Gen tok/s", justify="right")
        pf_table.add_column("VRAM (GB)", justify="right")
        pf_table.add_column("Quality", justify="right")

        for r in pareto:
            rc = r.config
            pf_table.add_row(
                "Custom",
                str(r.context_length),
                f"{rc.gpu_ratio:.0%}" if rc.gpu_ratio else "N/A",
                "GPU" if rc.offload_kv_cache_to_gpu else "CPU",
                str(rc.eval_batch_size) if rc.eval_batch_size else "N/A",
                f"{r.get_avg_generation_tok_s():.1f}",
                f"{r.peak_vram_gb:.1f}" if r.peak_vram_gb else "N/A",
                f"{r.quality_score.overall:.3f}" if r.quality_score else "N/A",
            )

        console.print(pf_table)


def _display_compare_result(res) -> None:
    """Display A/B compare verdict (ASCII only for Windows console)."""
    from lm_optimizer.services.compare import CompareResult

    assert isinstance(res, CompareResult)
    console.print(Panel.fit(f"[bold]Compare Result: {res.model_id}[/bold]"))
    console.print(f"Tests: {', '.join(res.cases) or 'none'}")

    table = Table(title="A vs B")
    table.add_column("Metric", style="cyan")
    table.add_column("A", justify="right", style="green")
    table.add_column("B", justify="right", style="yellow")
    table.add_row(
        "Status",
        "OK" if res.side_a_ok else f"FAIL ({res.error_a or '?'})",
        "OK" if res.side_b_ok else f"FAIL ({res.error_b or '?'})",
    )
    table.add_row("Generation tok/s", f"{res.speed_a:.1f}", f"{res.speed_b:.1f}")
    table.add_row(
        "Correctness",
        f"{res.quality_a:.3f}" if res.quality_a is not None else "N/A",
        f"{res.quality_b:.3f}" if res.quality_b is not None else "N/A",
    )
    console.print(table)

    summary = Table(title="Verdict")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Speed winner", str(res.speed_winner or "N/A"))
    summary.add_row("Quality winner", str(res.quality_winner or "N/A"))
    summary.add_row("Delta A-B", f"{res.delta_tok_s:.1f} tok/s")
    if res.speedup:
        summary.add_row("Speedup A/B", f"{res.speedup:.2f}x")
    summary.add_row("Verdict", res.verdict)
    console.print(summary)


@app.command()
def compare(
    model: str = typer.Argument(..., help="Model ID to compare on"),
    config_a: Path = typer.Option(..., "--config-a", help="Side A config JSON file"),
    config_b: Path = typer.Option(..., "--config-b", help="Side B config JSON file"),
    tests: Path | None = typer.Option(
        None, "--tests", help="Custom tests JSON file (default: 5 campaign tests)"
    ),
    context: int = typer.Option(2048, "--context", "-c", help="Context length for both sides"),
    repetitions: int = typer.Option(2, "--repetitions", "-r", help="Repetitions per test"),
):
    """Compare two load configs on the SAME tests (A/B verdict)."""
    from lm_optimizer.services.compare import (
        default_compare_cases,
        load_compare_cases_file,
        load_config_file,
        run_ab_compare,
    )

    setup_logging()

    async def _compare():
        client = get_client()
        try:
            snap = await _connect_guarded_or_abort(client, "compare")
            for line in format_snapshot(snap):
                console.print(f"  {line}")
            if not snap.get("verified_empty") and snap.get("leftovers"):
                console.print("[red]Stale models loaded, aborting. Unload them first.[/red]")
                sys.exit(1)

            try:
                cfg_a = load_config_file(config_a)
            except ValueError as e:
                console.print(f"[red]Side A config error: {e}[/red]")
                sys.exit(1)
            try:
                cfg_b = load_config_file(config_b)
            except ValueError as e:
                console.print(f"[red]Side B config error: {e}[/red]")
                sys.exit(1)
            if tests is not None:
                try:
                    cases = load_compare_cases_file(tests)
                except ValueError as e:
                    console.print(f"[red]Tests file error: {e}[/red]")
                    sys.exit(1)
            else:
                cases = default_compare_cases()

            with console.status("Running A/B comparison..."):
                res = await run_ab_compare(
                    client, model, cfg_a, cfg_b, cases,
                    repetitions=repetitions, context_length=context,
                )
            _display_compare_result(res)
            if not res.side_a_ok and not res.side_b_ok:
                sys.exit(1)
        except SystemExit:
            raise
        except Exception as e:
            logger.exception("Compare failed")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_compare())


@app.command(name="sample-sweep")
def sample_sweep(
    model: str = typer.Argument(..., help="Model ID to sweep sampling on"),
    config: Path = typer.Option(..., "--config", help="Load config JSON file"),
    context: int = typer.Option(2048, "--context", "-c", help="Context length"),
    repetitions: int = typer.Option(1, "--repetitions", "-r", help="Repetitions per test"),
):
    """Sweep generation sampling (precision/chat/creative) with quality re-validation."""
    from lm_optimizer.services.compare import load_config_file
    from lm_optimizer.services.quality import QualityEvaluator
    from lm_optimizer.services.sampling_sweep import run_sampling_sweep

    setup_logging()

    async def _sweep():
        client = get_client()
        try:
            snap = await _connect_guarded_or_abort(client, "sample-sweep")
            for line in format_snapshot(snap):
                console.print(f"  {line}")
            if not snap.get("verified_empty") and snap.get("leftovers"):
                console.print("[red]Stale models loaded, aborting. Unload them first.[/red]")
                sys.exit(1)

            try:
                cfg = load_config_file(config)
            except ValueError as e:
                console.print(f"[red]Config error: {e}[/red]")
                sys.exit(1)
            try:
                info = await client.get_model(model)
            except Exception:
                info = None
            arch = getattr(info, "architecture", None)
            if info is None:
                console.print(f"[red]Model not found: {model}[/red]")
                sys.exit(1)

            from lm_optimizer.services.benchmark import BenchmarkService

            svc = BenchmarkService(client)
            with console.status("Running sampling sweep..."):
                res = await run_sampling_sweep(
                    svc, QualityEvaluator(), model, cfg, context,
                    architecture=arch, repetitions=repetitions,
                )
            table = Table(title=f"Sampling sweep: {model}")
            table.add_column("Profile")
            table.add_column("Temp", justify="right")
            table.add_column("top_p", justify="right")
            table.add_column("top_k", justify="right")
            table.add_column("tok/s", justify="right")
            table.add_column("Quality", justify="right")
            table.add_column("Status")
            for row in res["combos"]:
                table.add_row(
                    row["profile"],
                    str(row.get("temperature")),
                    str(row.get("top_p")),
                    str(row.get("top_k")),
                    f"{row.get('tok_s', 0):.1f}",
                    str(row.get("quality")),
                    "OK" if row.get("ok") else f"FAIL ({row.get('error', '?')})",
                )
            console.print(table)
            if res["best_profile"]:
                console.print(f"[green]Best: {res['best_profile']}[/green]")
            else:
                console.print("[red]No passing combo[/red]")
                sys.exit(1)
        except SystemExit:
            raise
        except Exception as e:
            logger.exception("Sampling sweep failed")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_sweep())


@app.command(name="ollama-export")
def ollama_export(
    model: str = typer.Option(..., "--model", help="Base Ollama model tag for the FROM line"),
    param: list[str] | None = typer.Option(
        None, "--param", help="Option override KEY=VALUE (repeatable, wins over --from-show)"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the Modelfile here (bare filename -> ollama-configs/)"
    ),
    from_show: bool = typer.Option(
        False, "--from-show", help="Prefill with live /api/show parameters (needs server)"
    ),
    probe_max_ctx: bool = typer.Option(
        False, "--probe-max-ctx", help="Measure largest working num_ctx (tiny probe per rung)"
    ),
    base_url: str | None = typer.Option(
        None, "--base-url", help="Ollama URL (default http://127.0.0.1:11434)"
    ),
):
    """Render an editable Ollama Modelfile: live defaults + optimal overrides.

    Without flags: base tag + options only (no server calls). With
    --from-show: prefill the server's effective parameters, then apply
    --param overrides on top. With --probe-max-ctx: measure the largest
    loadable num_ctx ladder rung (unless --param num_ctx is given).
    """
    from lm_optimizer.backends.ollama.modelfile import (
        MAX_CTX_PROBE_LADDER,
        parse_param_assignment,
        parse_show_parameters,
        render_modelfile,
    )

    setup_logging()
    params: dict = {}
    for item in param or []:
        try:
            key, value = parse_param_assignment(item)
        except ValueError as e:
            console.print(f"[red]Invalid --param: {e}[/red]")
            sys.exit(2)
        params[key] = value

    if from_show or probe_max_ctx:
        from lm_optimizer.backends.ollama.client import OLLAMA_DEFAULT_URL, OllamaClient

        async def _live():
            client = OllamaClient(base_url=base_url or OLLAMA_DEFAULT_URL)
            try:
                await client.connect(echo_probe=False)
                show = await client.show(model)
                probed = None
                if probe_max_ctx and "num_ctx" not in params:
                    probed = await _probe_max_ctx(client, model)
                return show, probed
            finally:
                await client.close()

        try:
            show, probed_ctx = asyncio.run(_live())
        except Exception as e:
            console.print(f"[red]Ollama query failed: {type(e).__name__} ({e})[/red]")
            sys.exit(1)
        if from_show:
            live = parse_show_parameters(show)
            params = {**live, **params}
            console.print(f"[dim]Prefilled {len(live)} live parameter(s) from /api/show[/dim]")
        if probed_ctx is not None:
            params["num_ctx"] = probed_ctx
            console.print(f"[green]Probed max working num_ctx: {probed_ctx}[/green]")
    try:
        text = render_modelfile(model, params)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(2)
    if output is not None:
        try:
            from lm_optimizer.backends.ollama.modelfile import OLLAMA_CONFIG_DIR

            if output.parent == Path("."):
                OLLAMA_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
                output = OLLAMA_CONFIG_DIR / output.name
            output.write_text(text, encoding="utf-8")
        except OSError as e:
            console.print(f"[red]Cannot write {output}: {e}[/red]")
            sys.exit(1)
        console.print(f"[green]Modelfile written: {output}[/green]")
    console.print(text, highlight=False)


async def _probe_max_ctx(client, model: str) -> int | None:
    """Largest working num_ctx rung (tiny non-streaming probe each).

    Rungs above the model's own context limit are skipped without probing.
    """
    from lm_optimizer.backends.ollama.modelfile import MAX_CTX_PROBE_LADDER

    try:
        info = await client.show(model)
        details = info.get("details") if isinstance(info, dict) else None
        model_max = None
        if isinstance(details, dict):
            raw_max = details.get("context_length")
            model_max = int(raw_max) if raw_max else None
    except Exception:
        model_max = None
    rungs = [c for c in MAX_CTX_PROBE_LADDER if model_max is None or c <= model_max]
    for ctx_len in rungs:
        try:
            m = await client.generate(
                model, "Reply with exactly: OK",
                options={"num_ctx": ctx_len, "num_predict": 8, "temperature": 0.0},
            )
            if (m.text or "").strip():
                return ctx_len
        except Exception as e:
            logger.debug("num_ctx probe failed", ctx=ctx_len, error=str(e)[:120])
            continue
    return None


@app.command(name="ollama-apply")
def ollama_apply(
    file: Path = typer.Option(..., "--file", help="Edited Modelfile (.md) to apply"),
    as_tag: str | None = typer.Option(
        None, "--as", help="New model tag (default: <base>-best, never overwrite)"
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Confirm without prompting (required when non-interactive)"
    ),
    base_url: str | None = typer.Option(
        None, "--base-url", help="Ollama URL (default http://127.0.0.1:11434)"
    ),
):
    """Validate a Modelfile and persist it as a NEW Ollama tag via POST /api/create.

    Refuses when the target tag already exists; creating a model is a
    server-side write, so interactive confirmation (or --yes) is required.
    """
    from lm_optimizer.backends.ollama.client import OLLAMA_DEFAULT_URL, OllamaClient
    from lm_optimizer.backends.ollama.modelfile import (
        apply_modelfile_text,
        default_tag_for,
        ensure_tag_free,
        parse_modelfile,
    )

    setup_logging()
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as e:
        console.print(f"[red]Cannot read {file}: {e}[/red]")
        sys.exit(1)
    try:
        parsed = parse_modelfile(text)
    except ValueError as e:
        console.print(f"[red]Invalid Modelfile: {e}[/red]")
        sys.exit(1)
    tag = (as_tag or "").strip() or default_tag_for(str(parsed["from"]))

    async def _apply():
        client = OllamaClient(base_url=base_url or OLLAMA_DEFAULT_URL)
        try:
            try:
                await ensure_tag_free(client, tag)
            except ValueError as e:
                console.print(f"[red]{e}[/red]")
                sys.exit(1)
            except Exception as e:
                console.print(
                    f"[red]Cannot reach Ollama at {client.base_url}: "
                    f"{type(e).__name__}[/red]"
                )
                sys.exit(1)
            if not yes:
                try:
                    interactive = sys.stdin.isatty()
                except Exception:
                    interactive = False
                if not interactive:
                    console.print(
                        "[red]Refusing without --yes on a non-interactive terminal: "
                        "creating a model is a server-side write. "
                        "Re-run with --yes to confirm.[/red]"
                    )
                    sys.exit(2)
                if not typer.confirm(
                    f"Create new Ollama model {tag!r} from {parsed['from']!r}?",
                    default=False,
                ):
                    console.print("[yellow]Aborted: nothing was created.[/yellow]")
                    sys.exit(1)
            try:
                result = await apply_modelfile_text(client, text, tag)
            except ValueError as e:
                console.print(f"[red]{e}[/red]")
                sys.exit(1)
            except Exception as e:
                console.print(f"[red]Create failed: {type(e).__name__} ({e})[/red]")
                sys.exit(1)
            console.print(f"[green]Created {tag}[/green]")
            status = result.get("status") if isinstance(result, dict) else None
            if status:
                console.print(f"  status: {status}")
        except SystemExit:
            raise
        except Exception as e:
            logger.exception("Ollama apply failed")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_apply())


@app.command(name="ollama-probe")
def ollama_probe(
    model: str = typer.Argument(..., help="Ollama model tag to exercise"),
    ctx: int = typer.Option(8192, "--ctx", help="Second num_ctx timing rung (first is 2048)"),
    keep_loaded: bool = typer.Option(
        False, "--keep-loaded", help="Skip the final unload verification step"
    ),
    base_url: str | None = typer.Option(
        None, "--base-url", help="Ollama URL (default http://127.0.0.1:11434)"
    ),
):
    """Endpoint control matrix for one Ollama model (no optimization).

    Connect, list/show details, tiny generate (one automatic think-on
    retry when the model spends tokens but answers empty), a num_ctx
    timing pair, and unload verification. Prints a step table.
    """
    from lm_optimizer.backends.ollama.client import OLLAMA_DEFAULT_URL, OllamaClient
    from lm_optimizer.backends.ollama.probe import probe_model

    setup_logging()

    async def _probe():
        client = OllamaClient(base_url=base_url or OLLAMA_DEFAULT_URL)
        try:
            report = await probe_model(client, model, ctx_pair=(2048, ctx))
            if keep_loaded:
                report["steps"]["unload_verify"] = {"ok": True, "skipped": True}
            return report
        finally:
            await client.close()

    try:
        report = asyncio.run(_probe())
    except Exception as e:
        logger.exception("Ollama probe failed")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    table = Table(title=f"Ollama endpoint probe: {model}")
    table.add_column("Step", style="cyan")
    table.add_column("Result", style="green")

    def _add_step(name: str, step) -> None:
        if not isinstance(step, dict):
            table.add_row(name, str(step))
            return
        if step.get("ok"):
            detail = step.get("text") or step.get("version") or ""
            if "gen_tok_s" in step:
                detail = f"{step.get('text', '')!r} {step.get('gen_tok_s')} tok/s"
            table.add_row(name, f"OK {detail}".strip())
        elif step.get("skipped"):
            table.add_row(name, "[dim]skipped[/dim]")
        else:
            table.add_row(name, f"[red]FAIL: {step.get('error', '?')}[/red]")

    for name, step in report.get("steps", {}).items():
        if name == "ctx_timings" and isinstance(step, dict):
            for ctx_len, timing in step.items():
                _add_step(f"ctx {ctx_len}", timing)
        else:
            _add_step(name, step)
    console.print(table)


@app.command(name="deepresearch-bench")
def deepresearch_bench(
    model: str = typer.Argument(..., help="Model name to evaluate (must be available in LM Studio)"),
    dr_bench_dir: Path = typer.Option(
        Path("deep_research_bench"), "--dr-bench-dir", help="Path to DeepResearch Bench repository"
    ),
    limit: int | None = typer.Option(None, "--limit", help="Limit number of queries to process (for testing)"),
    skip_cleaning: bool = typer.Option(False, "--skip-cleaning", help="Skip article cleaning step"),
    only_zh: bool = typer.Option(False, "--only-zh", help="Only process Chinese queries"),
    only_en: bool = typer.Option(False, "--only-en", help="Only process English queries"),
    force: bool = typer.Option(False, "--force", help="Force re-evaluation even if results exist"),
    max_workers: int = typer.Option(5, "--max-workers", help="Maximum number of worker threads"),
    openrouter_api_key: str | None = typer.Option(None, "--openrouter-api-key", help="OpenRouter API key"),
    openai_api_key: str | None = typer.Option(None, "--openai-api-key", help="OpenAI API key"),
    jina_api_key: str | None = typer.Option(None, "--jina-api-key", help="Jina API key for web scraping"),
    llm_backend: str = typer.Option("openrouter", "--llm-backend", help="LLM backend: openrouter or openai"),
    race_model: str = typer.Option("openai/gpt-5.5", "--race-model", help="Model for RACE evaluation"),
    clean_model: str = typer.Option("openai/gpt-5.6-luna", "--clean-model", help="Model for article cleaning"),
    fact_model: str = typer.Option("openai/gpt-5.4-mini", "--fact-model", help="Model for FACT evaluation"),
    base_url: str | None = typer.Option(None, "--base-url", help="LM Studio base URL (default from config)"),
) -> None:
    """Run DeepResearch Bench evaluation on a local LM Studio model.

    This command runs the complete DeepResearch Bench pipeline:
    1. Clones/updates the DeepResearch Bench repository
    2. Installs requirements
    3. Runs the local model on 100 PhD-level research queries
    4. Runs RACE evaluation (report quality)
    5. Runs FACT evaluation (citation accuracy)

    Results are saved to deep_research_bench/results/
    """
    from lm_optimizer.services.deepresearch import DeepResearchConfig, run_deepresearch_benchmark, print_results_summary
    from lm_optimizer.config import config as global_config

    setup_logging()

    # Use provided base_url or fall back to config
    lm_studio_url = base_url or global_config.lmstudio_base_url or "http://127.0.0.1:1234"

    console.print("[cyan]DeepResearch Bench Evaluation[/cyan]")
    console.print(f"  Model: {model}")
    console.print(f"  LM Studio: {lm_studio_url}")
    console.print(f"  DR Bench Dir: {dr_bench_dir}")

    # API keys: explicit CLI flags win, env vars are the fallback
    # (avoids leaking secrets via shell history when possible).
    # Build config
    dr_config = DeepResearchConfig(
        model_name=model,
        dr_bench_dir=dr_bench_dir,
        limit=limit,
        skip_cleaning=skip_cleaning,
        only_zh=only_zh,
        only_en=only_en,
        force=force,
        max_workers=max_workers,
        openrouter_api_key=openrouter_api_key or os.environ.get("OPENROUTER_API_KEY"),
        openai_api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
        jina_api_key=jina_api_key or os.environ.get("JINA_API_KEY"),
        llm_backend=llm_backend,
        race_model=race_model,
        clean_model=clean_model,
        fact_model=fact_model,
    )

    try:
        success = asyncio.run(run_deepresearch_benchmark(dr_config, lm_studio_url))
        if success:
            print_results_summary(dr_config)
            console.print("[green]DeepResearch Bench evaluation completed successfully![/green]")
        else:
            console.print("[red]DeepResearch Bench evaluation failed![/red]")
            sys.exit(1)
    except KeyboardInterrupt:
        console.print("[yellow]Evaluation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        logger.exception("DeepResearch Bench evaluation failed")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


from lm_optimizer.cli.auto_run import auto_run
from lm_optimizer.cli.host_watch import host_watch
from lm_optimizer.cli.sandbox_duel import sandbox_duel

app.command()(auto_run)
app.command(name="sandbox-duel")(sandbox_duel)
app.command(name="host-watch")(host_watch)

if __name__ == "__main__":
    app()
