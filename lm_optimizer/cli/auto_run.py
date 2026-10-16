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
from lm_optimizer.services.lm_studio import LMStudioClient
from lm_optimizer.services.optimizer import AdaptiveOptimizer
from lm_optimizer.services.quality import QualityConfig, QualityEvaluator
from lm_optimizer.services.search_space import SearchSpaceGenerator
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
    """Get LM Studio version."""
    try:
        return _lms_cli.lms_version()
    except Exception:
        return "unknown"


def _compute_current_fingerprint(model: ModelIdentity, hardware: HardwareInfo) -> tuple[str, dict]:
    """Compute current fingerprint for a model."""
    lms_version = _get_lms_version()
    hw_fp = hardware_fingerprint(hardware)

    components = {
        "model_id": model.id,
        "quantization": model.quantization,
        "architecture": model.architecture,
        "lm_studio_version": lms_version,
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
    return fp, components


# Identity fields describe WHICH model a run belongs to. A run was looked up
# by model id, so these can never legitimately differ for the same model; a
# mismatch means the stored metadata is absent/stale, not that measurements
# are invalid. They are excluded from revalidation scope.
IDENTITY_FIELDS = ("model_id", "quantization", "architecture")


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
    if is_complete:
        return "RE_RUN", scope
    return ("RESUME_REVALIDATE", scope) if scope else ("RESUME", None)


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
        return {"model": model.id, "action": action, "error": str(e)}
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
):
    """Complete all incomplete model optimizations with fingerprint-aware decisions."""
    setup_logging()
    style = _require_style(style)
    workload = _normalize_workload(workload)

    output.mkdir(parents=True, exist_ok=True)

    # Discover models
    client = LMStudioClient()
    try:
        asyncio.run(client.connect())
    except Exception as e:
        console.print(f"[red]Cannot connect to LM Studio: {e}[/red]")
        raise typer.Exit(1)

    try:
        lm_models = asyncio.run(client.list_models())
    except Exception as e:
        console.print(f"[red]Failed to list models: {e}[/red]")
        raise typer.Exit(1)
    finally:
        asyncio.run(client.close())

    if not lm_models:
        console.print("[yellow]No models found[/yellow]")
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

    # Detect hardware
    from lm_optimizer.services.hardware import hardware_detector
    hardware = hardware_detector.detect()

    # Build plan
    plan = []
    for m in lm_models:
        # Get latest run
        runs = run_repo.get_by_model(m.id, limit=1)
        run = runs[0] if runs else None

        current_fp, current_components = _compute_current_fingerprint(m, hardware)
        action, fp_diff = _decide_action(run, current_fp, current_components)

        if revalidate_fingerprint:
            action = "RE_RUN" if action == "SKIP" else action
            if action in ("RESUME", "RESUME_REVALIDATE"):
                action = "RESUME_REVALIDATE"

        plan.append({
            "model": m,
            "run": run,
            "action": action,
            "fingerprint_match": run is not None and run.benchmark_params.get("fingerprint") == current_fp,
            "fp_diff": fp_diff,
            "current_fp": current_fp,
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
        console.print("[yellow]Dry run - no models were optimized[/yellow]")
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
                skip_phases=skip,
                physical_batches=physical_batches,
                revalidate_fingerprint=revalidate_fingerprint,
                output_dir=output,
            ))
            results.append(result)
            progress.update(task, description=f"Done: {p['model'].id}")

    # Generate report
    _generate_report(results, output, hardware, _get_lms_version())
    console.print(f"\n[green]Auto-run complete. Report: {output}/auto-run-{datetime.now():%Y%m%d-%H%M%S}.md[/green]")


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


def _generate_report(results: list, output_dir: Path, hardware: HardwareInfo, lms_version: str) -> str:
    """Generate markdown report. Returns the markdown content and writes to file."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = output_dir / f"auto-run-{timestamp}.md"

    summary = {
        "FULL": 0, "SKIP": 0, "RESUME": 0, "RESUME_REVALIDATE": 0, "RE_RUN": 0, "FAILED": 0
    }
    for r in results:
        summary[r["action"]] = summary.get(r["action"], 0) + 1

    lines = [
        f"# Auto-Run Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "| Metric | Count |",
        "|--------|-------|",
    ]
    for k, v in summary.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Per-Model Results")
    lines.append("")

    for r in results:
        action = r["action"]
        model_id = r["model"]
        icon = {
            "SKIP": "✅",
            "RESUME": "🔄",
            "RESUME_REVALIDATE": "🔁",
            "RE_RUN": "🔁",
            "FULL": "🆕",
            "FAILED": "❌",
        }.get(action, "❓")

        lines.append(f"### {icon} {model_id} ({action})")
        if r.get("run_id"):
            lines.append(f"- **Run**: `{r['run_id']}`")
        if r.get("best_score") is not None:
            lines.append(f"- **Best Score**: {r['best_score']:.3f}")
        if r.get("error"):
            lines.append(f"- **Error**: {r['error']}")
        if r.get("fp_diff"):
            lines.append(f"- **Fingerprint Diff**: {', '.join(r['fp_diff'])}")
        lines.append("")

    # Fingerprint changes table
    fingerprint_changes = [r for r in results if r.get("fp_diff")]
    if fingerprint_changes:
        lines.append("## Fingerprint Changes Detected")
        lines.append("| Model | Changed Fields | Action |")
        lines.append("|-------|---------------|--------|")
        for r in fingerprint_changes:
            diff_str = ", ".join(r["fp_diff"]) if r["fp_diff"] else "none"
            lines.append(f"| {r['model']} | {diff_str} | {r['action']} |")
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