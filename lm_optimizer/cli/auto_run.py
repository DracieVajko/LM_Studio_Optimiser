"""Auto-run CLI command: complete all incomplete model optimizations."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from lm_optimizer.config import config
from lm_optimizer.database.repositories import run_repo
from lm_optimizer.domain.models import (
    ConfigurationStatus,
    HardwareInfo,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
    OptimizationStage,
    RunStatus,
)
from lm_optimizer.logging_config import get_logger, setup_logging
from lm_optimizer.services.benchmark import BenchmarkService
from lm_optimizer.services.fingerprint import (
    compute_fingerprint,
    fingerprint_diff,
    hardware_fingerprint,
    should_revalidate_config,
)
from lm_optimizer.services.joblock import JobBusyError, acquire_lock, release_lock
from lm_optimizer.services.lm_studio import LMStudioClient
from lm_optimizer.services.model_eligibility import partition_models
from lm_optimizer.services.optimizer import AdaptiveOptimizer
from lm_optimizer.services.quality import QualityConfig, QualityEvaluator
from lm_optimizer.services.search_space import SearchSpaceGenerator
from lm_optimizer.services.version_identity import VersionIdentity, lms_identity
from lm_optimizer.services import lms_cli as _lms_cli
from lm_optimizer.benchmark.suite_version import BENCHMARK_SUITE_VERSION

logger = get_logger(__name__)
console = Console()

app = typer.Typer(name="auto-run", help="Complete all incomplete model optimizations")

Action = str  # FULL, SKIP, RESUME, RESUME_REVALIDATE, RE_RUN, FAILED


def _require_style(style: str) -> str:
    """Validate benchmark style."""
    valid = ["precise", "balanced", "creative"]
    if style not in valid:
        console.print(f"[red]Invalid style: {style}[/red]")
        console.print(f"Expected one of: {', '.join(valid)}")
        raise typer.Exit(2)
    return style


def _normalize_workload(workload: str) -> str:
    """Normalize workload string."""
    w = workload.strip().lower()
    if w not in ("interactive", "throughput"):
        console.print(f"[red]Invalid workload: {workload}[/red]")
        console.print("Expected: interactive or throughput")
        raise typer.Exit(2)
    return w


def _get_lms_version() -> str:
    """Raw version/build string for display only (never a fingerprint input)."""
    try:
        return _lms_cli.lms_version()
    except Exception:
        return "unknown"


def _version_identity(cli_performs_loads: bool) -> VersionIdentity:
    """Normalized lms identity used for the fingerprint's compatibility key."""
    return lms_identity(cli_performs_loads=cli_performs_loads)


def _compute_current_fingerprint(
    model: ModelIdentity, hardware: HardwareInfo, *, cli_performs_loads: bool = False
) -> tuple[str, dict]:
    """Compute current fingerprint for a model.

    ``lm_studio_version`` carries the normalized *compatibility key*, not the
    raw build string: an lms CLI rebuild that cannot affect a measurement must
    not invalidate results. The raw string and the parsed identity travel
    alongside for diagnostics only.
    """
    version = _version_identity(cli_performs_loads=cli_performs_loads)
    hw_fp = hardware_fingerprint(hardware)

    components = {
        "model_id": model.id,
        "quantization": model.quantization,
        "architecture": model.architecture,
        "lm_studio_version": version.compatibility_key,
        "hardware_fingerprint": hw_fp,
        "benchmark_suite_version": BENCHMARK_SUITE_VERSION,
    }

    fp = compute_fingerprint(
        components["model_id"],
        components["quantization"],
        components["architecture"],
        components["lm_studio_version"],
        components["hardware_fingerprint"],
        components["benchmark_suite_version"],
    )
    components["_version_identity"] = version.to_dict()
    return fp, components


# Identity fields describe WHICH model a run belongs to. A run was looked up
# by model id, so these can never legitimately differ for the same model; a
# mismatch means the stored metadata is absent/stale, not that measurements
# are invalid. They are excluded from revalidation scope.
IDENTITY_FIELDS = ("model_id", "quantization", "architecture")


def _is_normalized_version(value: object) -> bool:
    """True when a stored version value is a current-shape compatibility key.

    Current shapes carry enough identity to detect a real change:
    ``semver:<x.y.z>``, ``build:<hex commit>``, or the literal ``unknown``.
    Anything else (a raw build string from the pre-identity era, or the
    ``build:non-loading`` placeholder that discarded the commit hash) cannot
    distinguish one environment from another.
    """
    if not isinstance(value, str):
        return False
    v = value.strip()
    if v == "unknown":
        return True
    if v.startswith("semver:"):
        return len(v) > len("semver:")
    if v.startswith("build:"):
        suffix = v[len("build:"):]
        return bool(suffix) and suffix != "non-loading"
    return False


def _decide_action(
    run: OptimizationRun | None,
    current_fp: str,
    current_components: dict,
) -> tuple[Action, list[str] | None]:
    """Decide what action to take for a model based on run state and fingerprint.

    A run with NO stored fingerprint is UNKNOWN, not mismatched: we have no
    evidence conditions changed, so already-tested work is preserved and the
    fingerprint is backfilled on the next resume. Treating absent as mismatch
    would re-run every completed model on the first auto-run.
    """
    if run is None:
        return "FULL", None

    # Failed runs (not resumable) -> FULL re-run
    if run.status in (RunStatus.FAILED, RunStatus.CANCELLED):
        return "FULL", None

    is_complete = run.status in (
        RunStatus.SUCCESS,
        RunStatus.COMPLETED,
        RunStatus.COMPLETED_NO_WINNER,
    )

    old_fp = run.benchmark_params.get("fingerprint") if run.benchmark_params else None
    old_components = (
        run.benchmark_params.get("fingerprint_components", {}) if run.benchmark_params else {}
    )

    # Unknown fingerprint: preserve completed work, resume incomplete work.
    if not old_fp:
        return ("SKIP", None) if is_complete else ("RESUME", None)

    fingerprint_match = old_fp == current_fp
    if fingerprint_match:
        return ("SKIP", None) if is_complete else ("RESUME", None)

    diff = fingerprint_diff(old_components, current_components)
    scope = [f for f in diff if f not in IDENTITY_FIELDS]

    # A version stored in a legacy shape carries no usable identity: it was
    # either a raw build string or the `build:non-loading` placeholder that
    # deliberately discarded the commit hash. Neither can distinguish one
    # environment from another, so a difference against a current key is a
    # migration artifact, not evidence of change. Treat it as unknown (upgrade
    # the stored value) instead of re-benchmarking valid measurements.
    stored_version = old_components.get("lm_studio_version")
    if "lm_studio_version" in scope and not _is_normalized_version(stored_version):
        scope = [f for f in scope if f != "lm_studio_version"]

    # Nothing left in scope means nothing actually changed (the only
    # difference was a migration artifact). Do not turn that into work.
    if not scope:
        return ("SKIP", None) if is_complete else ("RESUME", None)

    if is_complete:
        return "RE_RUN", scope
    return "RESUME_REVALIDATE", scope


async def _execute_auto_run_for_model(
    model: ModelIdentity,
    hardware: HardwareInfo,
    action: Action,
    fingerprint_diff_fields: list[str] | None,
    profile: OptimizationProfile,
    quality_threshold: float,
    repetitions: int,
    validation_repetitions: int,
    style: str,
    workload: str,
    gpu_via_cli: bool,
    max_tokens_scale: float,
    min_speed: float,
    warmup_runs: int,
    max_context: int,
    min_gpu: float,
    max_gpu: float,
    optimize_context: bool,
    skip_phases: str | None,
    physical_batches: str | None,
    revalidate_fingerprint: bool,
    output_dir: Path,
) -> dict:
    """Execute auto-run for a single model. Returns result dict."""
    client = LMStudioClient()
    try:
        await client.connect()
    except Exception as e:
        return {"model": model.id, "action": "FAILED", "error": f"Connection failed: {e}"}

    benchmark_service = BenchmarkService(
        client,
        style=style,
        gpu_via_cli=gpu_via_cli,
    )
    quality_evaluator = QualityEvaluator(QualityConfig(minimum_score=quality_threshold))
    search_generator = SearchSpaceGenerator(client)
    optimizer = AdaptiveOptimizer(client, benchmark_service, quality_evaluator, search_generator)

    advanced = {
        "min_context": 2048,
        "max_context": max_context,
        "min_gpu_ratio": min_gpu,
        "max_gpu_ratio": max_gpu,
        "workload_type": workload,
        "gpu_via_cli": gpu_via_cli,
        "optimize_context": optimize_context,
        "min_speed": min_speed,
        "warmup_runs": warmup_runs,
        "selection_threshold": 0.05,
    }
    if skip_phases:
        advanced["skip_phases"] = [s.strip() for s in skip_phases.split(",")]
    if physical_batches:
        advanced["physical_batches"] = [int(x.strip()) for x in physical_batches.split(",")]

    try:
        if action == "RESUME":
            # Resume from checkpoint
            runs = run_repo.get_by_model(model.id, limit=1)
            if not runs:
                return {"model": model.id, "action": "FAILED", "error": "No run to resume"}
            run = runs[0]
            run = await optimizer.resume_from_checkpoint(
                run.id,
                revalidate=revalidate_fingerprint,
                revalidate_fingerprint=revalidate_fingerprint,
                fingerprint_diff_fields=fingerprint_diff_fields,
                style=style,
            )
        elif action == "RESUME_REVALIDATE":
            runs = run_repo.get_by_model(model.id, limit=1)
            if not runs:
                return {"model": model.id, "action": "FAILED", "error": "No run to resume"}
            run = runs[0]
            run = await optimizer.resume_from_checkpoint(
                run.id,
                revalidate_fingerprint=True,
                fingerprint_diff_fields=fingerprint_diff_fields,
                style=style,
            )
        else:
            # FULL or RE_RUN
            run = await optimizer.optimize(
                model=model,
                hardware=hardware,
                profile=profile,
                quality_threshold=quality_threshold,
                advanced_settings=advanced,
                style=style,
            )

        # Save report
        from lm_optimizer.services.reporting import save_best_report
        save_best_report(run, output_dir=output_dir)

        return {
            "model": model.id,
            "action": action,
            "run_id": str(run.id),
            "status": run.status.value,
            "best_score": run.get_best_config().score if run.get_best_config() else None,
        }

    except Exception as e:
        logger.exception(f"Auto-run failed for {model.id}")
        return {
            "model": model.id,
            "action": "FAILED",
            "error": f"{type(e).__name__}: {e}"[:300],
        }
    finally:
        await client.close()


def auto_run(
    models: Annotated[Optional[list[str]], typer.Argument(help="Model keys (default: all, small first)")] = None,
    profile: str = typer.Option("balanced", "--profile", "-p", help="Optimization profile (speed/balanced/context/quality)"),
    quality: float = typer.Option(0.97, "--quality", "-q", help="Minimum quality threshold (0.9-1.0)"),
    style: str = typer.Option("balanced", "--style", help="Benchmark style (precise/balanced/creative)"),
    repetitions: int = typer.Option(3, "--repetitions", "-r", help="Benchmark repetitions per config"),
    validation: int = typer.Option(5, "--validation", "-v", help="Validation repetitions for best config"),
    max_context: int = typer.Option(32768, "--max-context", help="Maximum context length"),
    min_gpu: float = typer.Option(0.0, "--min-gpu", help="Minimum GPU ratio"),
    max_gpu: float = typer.Option(1.0, "--max-gpu", help="Maximum GPU ratio"),
    workload: str = typer.Option("interactive", "--workload", help="Workload: interactive or throughput"),
    gpu_via_cli: bool = typer.Option(True, "--gpu-via-cli/--no-gpu-via-cli", help="Load GPU offload via lms CLI"),
    max_tokens_scale: float = typer.Option(1.0, "--max-tokens-scale", help="Scale case max-tokens 0.25-1.0"),
    min_speed: float = typer.Option(0.0, "--min-speed", help="Minimum generation tok/s to accept config"),
    warmup: int = typer.Option(1, "--warmup", help="Warmup runs per config 0-5"),
    skip: Optional[str] = typer.Option(None, "--skip", help="Comma-separated model id substrings to skip"),
    max_size_gb: Optional[float] = typer.Option(None, "--max-size-gb", help="Auto-pick only models up to this file size"),
    revalidate_fingerprint: bool = typer.Option(False, "--revalidate-fingerprint", help="Force re-check even if fingerprint matches"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without running"),
    output: Path = typer.Option(Path("results"), "--output", help="Report output directory"),
    yes: bool = typer.Option(False, "--yes", help="Accept plan without prompting"),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="Never prompt"),
    force_generation: bool = typer.Option(
        False,
        "--force-generation",
        help="Opt in: treat models with missing/unknown type as generation-capable",
    ),
):
    """Complete all incomplete model optimizations with fingerprint-aware decisions."""
    setup_logging()
    style = _require_style(style)
    workload = _normalize_workload(workload)

    output.mkdir(parents=True, exist_ok=True)

    # Serialize against every other optimizer command on this host. Acquired
    # BEFORE discovery so a second invocation cannot observe or mutate any
    # run state, and released on both the normal and the exception path.
    _acquire_auto_run_lock(len(models) if models else 0)
    try:
        _auto_run_body(
            models=models,
            profile=profile,
            quality=quality,
            style=style,
            repetitions=repetitions,
            validation=validation,
            max_context=max_context,
            min_gpu=min_gpu,
            max_gpu=max_gpu,
            workload=workload,
            gpu_via_cli=gpu_via_cli,
            max_tokens_scale=max_tokens_scale,
            min_speed=min_speed,
            warmup=warmup,
            skip=skip,
            max_size_gb=max_size_gb,
            revalidate_fingerprint=revalidate_fingerprint,
            dry_run=dry_run,
            output=output,
            yes=yes,
            non_interactive=non_interactive,
            force_generation=force_generation,
        )
    finally:
        release_lock()


def _acquire_auto_run_lock(target_count: int) -> None:
    """Claim the shared optimizer job lock or exit nonzero.

    Same lock domain as `optimize` / `auto` / `resume`: one optimizer job per
    host. Two auto-run processes must never drive the same LM Studio instance.
    """
    try:
        acquire_lock(f"auto-run:{target_count}")
    except JobBusyError as e:
        holder = e.holder or {}
        console.print("[red]Another optimizer job is already running:[/red]")
        console.print(
            f"  pid {holder.get('pid')} purpose {holder.get('purpose')!r}"
        )
        console.print("Run one job at a time; retry when it finishes.")
        raise typer.Exit(1)


def _auto_run_body(
    *,
    models,
    profile,
    quality,
    style,
    repetitions,
    validation,
    max_context,
    min_gpu,
    max_gpu,
    workload,
    gpu_via_cli,
    max_tokens_scale,
    min_speed,
    warmup,
    skip,
    max_size_gb,
    revalidate_fingerprint,
    dry_run,
    output,
    yes,
    non_interactive,
    force_generation,
) -> None:
    """Discovery -> eligibility -> plan -> (execute) -> report."""
    # Auto-reconcile stale runs from previous crashed/killed invocations.
    try:
        from lm_optimizer.services.run_reconcile import reconcile_stale_runs

        recon = reconcile_stale_runs(dry_run=False)
        if recon.get("changed"):
            console.print(f"[yellow]Reconciled {recon['changed']} stale run(s).[/yellow]")
    except Exception as e:
        logger.debug("Auto-reconcile failed", error=str(e))

    # One event loop for the whole discovery exchange. Calling asyncio.run()
    # three times reuses a client across closed loops and raises
    # "Event loop is closed" on the second call.
    async def _discover():
        client = LMStudioClient()
        try:
            # echo_probe=False is REQUIRED here: connect() otherwise performs a
            # real load of the smallest LLM (to verify applied configs). A plan
            # must never load a model -- discovery has to stay read-only.
            await client.connect(echo_probe=False)
            return await client.list_models()
        finally:
            try:
                await client.close()
            except Exception:
                pass

    try:
        lm_models = asyncio.run(_discover())
    except Exception as e:
        # A provider failure is NOT an empty inventory: say so and exit
        # nonzero so a broken host is never reported as "nothing to do".
        console.print(f"[red]Cannot reach LM Studio to list models: {e}[/red]")
        raise typer.Exit(1)

    if lm_models is None:
        console.print("[red]Model inventory unavailable (provider returned nothing).[/red]")
        raise typer.Exit(1)

    if not lm_models:
        console.print("[yellow]No models available from LM Studio; nothing to do.[/yellow]")
        return

    # Filter models
    if models:
        lm_models = [m for m in lm_models if any(sub in m.id for sub in models)]
    if skip:
        skip_patterns = [s.strip() for s in skip.split(",")]
        lm_models = [m for m in lm_models if not any(p in m.id for p in skip_patterns)]
    if max_size_gb and max_size_gb > 0:
        lm_models = [m for m in lm_models if (m.size_bytes or 0) <= max_size_gb * 1024**3]

    # Sort by size ascending
    lm_models.sort(key=lambda m: m.size_bytes or 0)

    # Eligibility: an embedding/reranker/encoder model is not a generation
    # benchmark target. Excluding it is NOT a failure.
    eligible_models, skipped_models, eligibility = partition_models(
        lm_models, force_generation=force_generation
    )
    console.print(
        f"[dim]Eligibility: discovered={eligibility.discovered} "
        f"eligible={eligibility.eligible} excluded={eligibility.excluded} "
        f"unknown={eligibility.unknown}[/dim]"
    )
    for m, verdict in skipped_models:
        console.print(f"  [dim]skip {getattr(m, 'id', m)}: {verdict.reason}[/dim]")
    lm_models = eligible_models

    if not lm_models:
        console.print(
            "[yellow]No generation-capable models to optimize "
            "(all discovered models were excluded as non-generative).[/yellow]"
        )
        return

    # Detect hardware
    from lm_optimizer.services.hardware import hardware_detector
    hardware = hardware_detector.detect()

    # Build plan
    plan = []
    for m in lm_models:
        # Get latest run
        runs = run_repo.get_by_model(m.id, limit=1)
        run = runs[0] if runs else None

        current_fp, current_components = _compute_current_fingerprint(
            m, hardware, cli_performs_loads=gpu_via_cli
        )
        action, fp_diff = _decide_action(run, current_fp, current_components)

        if revalidate_fingerprint:
            action = "RE_RUN" if action == "SKIP" else action
            if action in ("RESUME", "RESUME_REVALIDATE"):
                action = "RESUME_REVALIDATE"

        plan.append({
            "model": m,
            "run": run,
            "action": action,
            "fingerprint_match": (
                run is not None
                and (run.benchmark_params or {}).get("fingerprint") == current_fp
            ),
            "fp_diff": fp_diff,
            "current_fp": current_fp,
            "previous_run_id": str(run.id) if run is not None else None,
        })

    # Display plan
    table = Table(title="Auto-Run Plan")
    table.add_column("Model", style="cyan")
    table.add_column("Action", style="green")
    table.add_column("Fingerprint", style="dim")
    table.add_column("Details", style="yellow")

    for p in plan:
        details = ""
        if p["action"] == "SKIP":
            details = "Complete, fingerprint match"
        elif p["action"] == "RESUME":
            completed = len(p["run"].configurations) if p["run"] else 0
            details = f"Incomplete ({completed} configs done)"
        elif p["action"] == "RESUME_REVALIDATE":
            completed = len(p["run"].configurations) if p["run"] else 0
            diff_str = ", ".join(p["fp_diff"]) if p["fp_diff"] else "unknown"
            details = f"Incomplete + fingerprint mismatch ({diff_str})"
        elif p["action"] == "RE_RUN":
            diff_str = ", ".join(p["fp_diff"]) if p["fp_diff"] else "unknown"
            details = f"Complete but fingerprint mismatch ({diff_str})"
        elif p["action"] == "FULL":
            details = "No previous run"
        table.add_row(p["model"].id, p["action"], p["current_fp"][:16] + "...", details)

    console.print(table)

    # Confirm
    if not _confirm_plan(plan, dry_run, yes, non_interactive):
        return

    if dry_run:
        # A plan report describes intent only. It must never carry a run id,
        # score or status as if a measurement had happened.
        plan_rows = [
            {
                "model": p["model"].id,
                "action": p["action"],
                "fp_diff": p["fp_diff"],
                "previous_run_id": p["previous_run_id"],
            }
            for p in plan
        ]
        _generate_report(plan_rows, output, hardware, _get_lms_version(), kind="plan")
        console.print("[yellow]Dry run - no models were optimized.[/yellow]")
        return

    # Execute
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for p in plan:
            task = progress.add_task(f"Processing {p['model'].id}...", total=None)
            result = asyncio.run(_execute_auto_run_for_model(
                model=p["model"],
                hardware=hardware,
                action=p["action"],
                fingerprint_diff_fields=p["fp_diff"],
                profile=OptimizationProfile(profile),
                quality_threshold=quality,
                repetitions=repetitions,
                validation_repetitions=validation,
                style=style,
                workload=workload,
                gpu_via_cli=gpu_via_cli,
                max_tokens_scale=max_tokens_scale,
                min_speed=min_speed,
                warmup_runs=warmup,
                max_context=max_context,
                min_gpu=min_gpu,
                max_gpu=max_gpu,
                optimize_context=False,
                skip_phases=None,
                physical_batches=None,
                revalidate_fingerprint=revalidate_fingerprint,
                output_dir=output,
            ))
            results.append(result)
            progress.update(task, description=f"Done: {p['model'].id}")

    # Generate report (live execution: real run ids, scores and errors)
    _generate_report(results, output, hardware, _get_lms_version(), kind="results")
    ok = sum(1 for r in results if r["action"] != "FAILED")
    console.print(f"\n[green]Auto-run complete: {ok}/{len(results)} models processed.[/green]")
    console.print(f"[dim]Report directory: {output}[/dim]")


def _confirm_plan(plan: list, dry_run: bool, yes: bool, non_interactive: bool) -> bool:
    """Show plan and get confirmation."""
    if dry_run or yes or non_interactive:
        return True

    counts = {}
    for p in plan:
        counts[p["action"]] = counts.get(p["action"], 0) + 1

    console.print("\nSummary:")
    for action, count in counts.items():
        console.print(f"  {action}: {count}")

    return typer.confirm("\nProceed?", default=True)


ACTIONS = ("FULL", "SKIP", "RESUME", "RESUME_REVALIDATE", "RE_RUN", "FAILED")

_ACTION_ICONS = {
    "SKIP": "✅",
    "RESUME": "🔄",
    "RESUME_REVALIDATE": "🔁",
    "RE_RUN": "🔁",
    "FULL": "🆕",
    "FAILED": "❌",
}


def _unique_report_path(output_dir: Path, kind: str) -> Path:
    """Collision-free report path.

    Second-resolution timestamps collide when two runs land in the same second,
    silently overwriting the first report. Microseconds plus a numeric suffix
    guarantee a fresh file per invocation and never clobber an existing one.
    """
    stem = f"auto-run-{kind}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    candidate = output_dir / f"{stem}.md"
    n = 1
    while candidate.exists():
        candidate = output_dir / f"{stem}-{n}.md"
        n += 1
    return candidate


def _generate_report(
    results: list,
    output_dir: Path,
    hardware: HardwareInfo,
    lms_version: str,
    kind: str = "results",
) -> str:
    """Render the auto-run markdown report and write it to ``output_dir``.

    ``kind`` is the source of truth for what the document claims:

    - ``"plan"``: a dry run. Nothing was optimized, so no run id, score or
      status may be presented as an outcome.
    - ``"results"``: a live execution. Run ids / scores / errors are real
      outcomes and are rendered as such.

    A record carrying an error stays an error: a failure is never replaced by
    a fabricated score or run id.
    """
    if kind not in ("plan", "results"):
        raise ValueError(f"unknown report kind: {kind!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = _unique_report_path(output_dir, kind)
    now = datetime.now()

    summary = dict.fromkeys(ACTIONS, 0)
    for r in results:
        summary[r["action"]] = summary.get(r["action"], 0) + 1

    if kind == "plan":
        title = f"# Auto-Run Plan (dry run) - {now.strftime('%Y-%m-%d %H:%M:%S')}"
    else:
        title = f"# Auto-Run Report - {now.strftime('%Y-%m-%d %H:%M:%S')}"

    lines = [
        title,
        "",
        "## Summary",
        "| Action | Count |",
        "|--------|-------|",
    ]
    for k in ACTIONS:
        lines.append(f"| {k} | {summary[k]} |")
    lines.append(f"| **Total** | **{len(results)}** |")
    lines.append("")

    section = (
        "## Proposed Actions (nothing was executed)"
        if kind == "plan"
        else "## Per-Model Results"
    )
    lines.append(section)
    lines.append("")

    for r in results:
        action = r["action"]
        icon = _ACTION_ICONS.get(action, "❓")
        lines.append(f"### {icon} {r['model']} ({action})")

        if kind == "plan":
            # Only intent: never an outcome.
            if r.get("fp_diff"):
                lines.append(f"- **Fingerprint Diff**: {', '.join(r['fp_diff'])}")
            if r.get("previous_run_id"):
                lines.append(f"- **Existing Run**: `{r['previous_run_id']}`")
            lines.append("")
            continue

        if r.get("run_id"):
            lines.append(f"- **Run**: `{r['run_id']}`")
        if r.get("status"):
            lines.append(f"- **Status**: {r['status']}")
        if r.get("best_score") is not None:
            lines.append(f"- **Best Score**: {r['best_score']:.3f}")
        if r.get("error"):
            lines.append(f"- **Error**: {r['error']}")
        if r.get("fp_diff"):
            lines.append(f"- **Fingerprint Diff**: {', '.join(r['fp_diff'])}")
        lines.append("")

    if kind == "results":
        failed = [r for r in results if r.get("error") or r["action"] == "FAILED"]
        if failed:
            lines.append("## Failures")
            lines.append("")
            for r in failed:
                lines.append(f"- `{r['model']}`: {r.get('error') or r['action']}")
            lines.append("")

    fingerprint_changes = [r for r in results if r.get("fp_diff")]
    if fingerprint_changes:
        lines.append("## Fingerprint Changes Detected")
        lines.append("")
        lines.append("| Model | Changed Fields | Action |")
        lines.append("|-------|---------------|--------|")
        for r in fingerprint_changes:
            lines.append(f"| {r['model']} | {', '.join(r['fp_diff'])} | {r['action']} |")
        lines.append("")

    # Environment
    lines.append("## Environment")
    lines.append(f"- **LM Studio**: {lms_version}")
    lines.append(f"- **GPU**: {', '.join(f'{g.name} ({g.vram_gb:.1f}GB)' for g in hardware.gpus) or 'CPU-only'}")
    lines.append(f"- **CPU**: {hardware.cpu_name} ({hardware.cpu_cores_physical}P/{hardware.cpu_cores_logical}L)")
    lines.append(f"- **RAM**: {hardware.total_ram_gb:.1f} GB")
    lines.append(f"- **Benchmark Suite**: {BENCHMARK_SUITE_VERSION}")

    content = "\n".join(lines)
    report_path.write_text(content, encoding="utf-8")
    console.print(f"Report saved to {report_path}")
    return content