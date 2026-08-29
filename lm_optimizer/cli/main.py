"""CLI commands for LM Studio Optimizer."""

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from lm_optimizer.config import config
from lm_optimizer.database.repositories import (
    preset_repo,
    run_repo,
)
from lm_optimizer.domain.models import (
    ConfigurationStatus,
    LoadConfiguration,
    OptimizationProfile,
)
from lm_optimizer.logging_config import get_logger, setup_logging
from lm_optimizer.services.benchmark import BenchmarkService
from lm_optimizer.services.hardware import hardware_detector
from lm_optimizer.services.optimizer import AdaptiveOptimizer
from lm_optimizer.services.quality import QualityConfig, QualityEvaluator
from lm_optimizer.services.search_space import SearchSpaceGenerator

app = typer.Typer(
    name="lm-optimizer",
    help="LM Studio Auto Optimizer - Automatically discover the best inference configuration",
    add_completion=False,
)

console = Console()
logger = get_logger(__name__)


def get_client(base_url: str | None = None) -> "LMStudioClient":
    """Create LM Studio client, optionally overriding URL (without persisting)."""
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


def _handle_connection_error(url: str, error: Exception) -> None:
    """User-friendly connection error without exposing secrets."""
    console.print("\n[red]Cannot connect to LM Studio at:[/red]")
    console.print(f"  {url}")
    console.print("\n[bold]Check:[/bold]")
    console.print("  - LM Studio is running")
    console.print("  - Developer API is enabled (LM Studio → Settings → Developer)")
    console.print("  - URL/port is correct")
    console.print("  - Network access is allowed (firewall/VPN)")
    console.print(f"\n[dim]Details: {type(error).__name__}[/dim]")
    # Do not log full URL with credentials or stack trace containing secrets


@app.callback()
def callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    config_file: Path | None = typer.Option(None, "--config", "-c", help="Config file path"),
):
    """LM Studio Auto Optimizer."""
    if verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)


@app.command()
def status(
    url: str | None = typer.Option(
        None, "--url", help="LM Studio API URL (overrides .env, not persisted)"
    ),
):
    """Show LM Studio connection status and hardware info."""
    setup_logging()

    # Resolve URL: CLI override > .env > default
    effective_url = url or config.lm_studio.base_url

    async def _status():
        client = get_client(base_url=effective_url)
        try:
            await client.connect()
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

            console.print("\n[bold]LM Studio[/bold]")
            console.print(f"  URL: {client.base_url}")
            console.print("  Status: [green]Connected[/green]")
            console.print(f"  API Version: {client.capabilities.version}")
            console.print(
                f"  Supported Parameters: {', '.join(client.capabilities.get_supported_load_params())}"
            )
            if url:
                console.print("  [dim]Using CLI override (not persisted to .env)[/dim]")

        except Exception as e:
            _handle_connection_error(effective_url, e)
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_status())


@app.command()
def models():
    """List all available models."""
    setup_logging()

    async def _models():
        client = get_client()
        try:
            await client.connect()
            models = await client.list_models()

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
                    "✓" if m.is_moe else "",
                    size_gb,
                    "✓" if client.get_loaded_model(m.id) else "",
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
    repetitions: int = typer.Option(
        3, "--repetitions", "-r", help="Number of benchmark repetitions"
    ),
):
    """Run benchmark with specific configuration."""
    setup_logging()

    async def _benchmark():
        client = get_client()
        try:
            await client.connect()

            load_config = LoadConfiguration(
                context_length=context,
                gpu_ratio=gpu_ratio,
                flash_attention=flash,
                offload_kv_cache_to_gpu=kv_gpu,
                eval_batch_size=batch,
            )

            benchmark_service = BenchmarkService(
                client,
                benchmark_config=type(
                    "obj",
                    (object,),
                    {
                        "repetitions": repetitions,
                        "warmup_repetitions": 1,
                        "timeout_seconds": 300,
                        "load_timeout_seconds": 60,
                        "generation_timeout_seconds": 120,
                    },
                )(),
            )

            with console.status("Running benchmark..."):
                result = await benchmark_service.run_benchmark(model, load_config, context)

            _display_benchmark_result(result)

        except Exception as e:
            logger.exception("Benchmark failed")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_benchmark())


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
    dry_run: bool = typer.Option(False, "--dry-run", help="Show search space without running"),
):
    """Optimize a model configuration."""
    setup_logging()

    async def _optimize():
        client = get_client()
        try:
            await client.connect()

            # Get model info
            model_info = await client.get_model(model)
            if not model_info:
                console.print(f"[red]Model not found: {model}[/red]")
                sys.exit(1)

            hardware = hardware_detector.detect()

            # Create services
            benchmark_service = BenchmarkService(
                client,
                benchmark_config=type(
                    "obj",
                    (object,),
                    {
                        "repetitions": repetitions,
                        "warmup_repetitions": 1,
                        "timeout_seconds": 300,
                        "load_timeout_seconds": 60,
                        "generation_timeout_seconds": 120,
                    },
                )(),
            )
            quality_evaluator = QualityEvaluator(QualityConfig(minimum_score=quality_threshold))
            search_generator = SearchSpaceGenerator(client)
            optimizer = AdaptiveOptimizer(
                client, benchmark_service, quality_evaluator, search_generator
            )

            # Advanced settings
            advanced = {
                "min_context": min_context,
                "max_context": max_context,
                "min_gpu_ratio": min_gpu,
                "max_gpu_ratio": max_gpu,
                "test_flash_on": True,
                "test_flash_off": True,
                "test_kv_gpu": True,
                "test_kv_cpu": True,
                "auto_batch": True,
            }

            if dry_run:
                # Just show search space
                search_space = search_generator.generate(
                    model_info, hardware, OptimizationProfile(profile), advanced
                )
                console.print(
                    f"[yellow]Dry run - would test {search_space.estimate_size()} configurations[/yellow]"
                )
                console.print_json(search_space.to_dict())
                return

            console.print(f"[bold]Starting optimization for {model}[/bold]")
            console.print(f"  Profile: {profile}")
            console.print(f"  Quality threshold: {quality_threshold}")
            console.print(f"  Repetitions: {repetitions}")
            console.print(f"  Validation repetitions: {validation_repetitions}")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Optimizing {model}...", total=100)

                result = await optimizer.optimize(
                    model_info,
                    hardware,
                    OptimizationProfile(profile),
                    quality_threshold=quality_threshold,
                    advanced_settings=advanced,
                )

                progress.update(task, completed=100)

            _display_optimization_result(result)

            # Save preset
            preset_data = {
                "model_id": model,
                "profile": profile,
                "config": result.recommended_config.to_dict()
                if hasattr(result, "recommended_config")
                else result.best_config.config.to_dict(),
                "metrics": {
                    "generation_tok_s": result.best_config.get_avg_generation_tok_s(),
                    "prompt_tok_s": result.best_config.get_avg_prompt_tok_s(),
                    "ttft_ms": result.best_config.get_avg_ttft_ms(),
                    "quality_score": result.best_config.quality_score.overall
                    if result.best_config.quality_score
                    else 0,
                },
                "quality": {
                    "overall": result.best_config.quality_score.overall
                    if result.best_config.quality_score
                    else 0,
                },
                "run_id": str(result.id),
                "optimizer_version": "1.0.0-beta",
            }

            preset_repo.save(preset_data)
            console.print("\n[green]Preset saved[/green]")

        except Exception as e:
            logger.exception("Optimization failed")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_optimize())


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
            console.print("Run 'lm-optimizer optimize' first to create a preset.")
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
                sys.exit(1)

        except Exception as e:
            logger.exception("Apply failed")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_apply())


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
            "running": "yellow",
            "failed": "red",
            "cancelled": "red",
        }.get(r.status.value, "white")

        table.add_row(
            r.created_at.strftime("%Y-%m-%d %H:%M"),
            r.model.name,
            r.profile.value,
            f"[{status_style}]{r.status.value}[/{status_style}]",
            f"{r.best_config.score:.3f}" if r.best_config_id else "—",
            f"{r.duration_seconds:.1f}s" if r.duration_seconds else "—",
            str(len(r.configurations)),
        )

    console.print(table)


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
            p["created_at"][:19] if p["created_at"] else "—",
            f"{metrics.get('generation_tok_s', 0):.1f}",
            f"{metrics.get('quality_score', 0):.3f}",
        )

    console.print(table)


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


def _display_benchmark_result(result):
    """Display benchmark results."""
    console.print(Panel.fit(f"[bold]Benchmark Result: {result.model_id}[/bold]"))

    table = Table(title="Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Context Length", str(result.context_length))
    table.add_row("GPU Ratio", f"{result.gpu_ratio:.2f}" if result.gpu_ratio else "Auto")
    table.add_row("Flash Attention", "Enabled" if result.flash_attention else "Disabled")
    table.add_row("KV Cache GPU", "Yes" if result.offload_kv_cache_to_gpu else "No")
    table.add_row(
        "Eval Batch Size", str(result.eval_batch_size) if result.eval_batch_size else "Auto"
    )

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
            "✓" if m.success else "✗",
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


def _display_optimization_result(result):
    """Display optimization results with baseline comparison and score breakdown."""
    console.print(Panel.fit(f"[bold green]Optimization Complete: {result.model_id}[/bold green]"))

    # Best config
    table = Table(title="Recommended Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    best = result.best_config
    table.add_row("Context Length", str(best.context_length))
    table.add_row("GPU Ratio", f"{best.gpu_ratio:.2f}" if best.gpu_ratio else "Auto")
    table.add_row("Flash Attention", "Enabled" if best.flash_attention else "Disabled")
    table.add_row("KV Cache GPU", "Yes" if best.offload_kv_cache_to_gpu else "No")
    table.add_row("Eval Batch Size", str(best.eval_batch_size) if best.eval_batch_size else "Auto")
    if best.num_experts:
        table.add_row("Experts", str(best.num_experts))
    if getattr(result, "is_experimental", False):
        table.add_row("Experimental", result.experimental_reason or "Yes")

    console.print(table)

    # Metrics with baseline comparison (percentage changes from actual measurements)
    metrics_table = Table(title="Performance Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Optimized", style="green")
    if result.baseline_metrics:
        metrics_table.add_column("Baseline", style="yellow")
        metrics_table.add_column("Change", style="magenta")
    else:
        metrics_table.add_column("Value", style="green")
        metrics_table.add_column("Note", style="dim")

    def fmt_change(new, old):
        if old and old != 0:
            pct = (new - old) / old * 100
            sign = "+" if pct >= 0 else ""
            return f"{sign}{pct:.1f}%"
        return "—"

    # Gather baseline if available
    baseline = result.baseline_metrics or {}
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
            metrics_table.add_row("Correctness / Quality", "N/A", "N/A", "—")
        metrics_table.add_row(
            "Stability Score",
            f"{best.stability_score:.3f}",
            f"{baseline.get('stability', 0):.3f}" if baseline.get("stability") else "N/A",
            "—",
        )
        metrics_table.add_row(
            "Context Length",
            str(best.context_length),
            str(baseline.get("context_length", "N/A")),
            fmt_change(best.context_length, baseline.get("context_length")),
        )
        metrics_table.add_row(
            "VRAM",
            f"{best.peak_vram_gb:.1f} GB" if best.peak_vram_gb else "—",
            f"{baseline.get('peak_vram_gb', 0):.1f} GB" if baseline.get("peak_vram_gb") else "—",
            fmt_change(best.peak_vram_gb or 0, baseline.get("peak_vram_gb") or 0),
        )
        metrics_table.add_row("Optimization Time", f"{result.duration_seconds:.1f} s", "—", "—")
    else:
        metrics_table.add_row(
            "Generation Speed", f"{best.get_avg_generation_tok_s():.1f} tok/s", "", ""
        )
        metrics_table.add_row(
            "Prompt Processing", f"{best.get_avg_prompt_tok_s():.0f} tok/s", "", ""
        )
        metrics_table.add_row(
            "Est. TTFT", f"{best.get_avg_estimated_ttft_ms():.0f} ms (estimated)", "", ""
        )
        if best.quality_score:
            qs = best.quality_score
            checks = (
                f"{qs.checks_passed}/{qs.checks_total} checks passed"
                if hasattr(qs, "checks_passed")
                else ""
            )
            metrics_table.add_row(
                "Correctness / Quality", f"{qs.overall:.3f} ({checks})", "heuristic checks", ""
            )
        metrics_table.add_row("Stability Score", f"{best.stability_score:.3f}", "", "")
        metrics_table.add_row("Context Length", str(best.context_length), "", "")
        metrics_table.add_row("Optimization Time", f"{result.duration_seconds:.1f} s", "", "")

    console.print(metrics_table)

    # Score breakdown explainability
    if best.score_breakdown:
        console.print("\n[bold]Score Breakdown (why this config won):[/bold]")
        bd_table = Table(title=f"{result.profile.value.capitalize()} score: {best.score:.3f}")
        bd_table.add_column("Component", style="cyan")
        bd_table.add_column("Normalized", justify="right")
        bd_table.add_column("Weight", justify="right")
        bd_table.add_column("Contribution", justify="right")
        weights = result.profile_weights if hasattr(result, "profile_weights") else {}
        for comp, norm in best.score_breakdown.items():
            w = weights.get(comp, 0)
            contrib = norm * w
            bd_table.add_row(comp, f"{norm:.3f}", f"{w:.2f}", f"{contrib:.3f}")
        console.print(bd_table)
        console.print(
            "[dim]Normalization: speeds/TTFT run-relative, context model-relative, memory hardware-relative. See docs/OPTIMIZATION_METHOD.md[/dim]"
        )

    # Pareto frontier
    if result.pareto_config_ids:
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

        for r in result.pareto_config_ids:
            pf_table.add_row(
                "Custom",
                str(r.context_length),
                f"{r.gpu_ratio:.0%}" if r.gpu_ratio else "N/A",
                "GPU" if r.offload_kv_cache_to_gpu else "CPU",
                str(r.eval_batch_size) if r.eval_batch_size else "N/A",
                f"{r.get_avg_generation_tok_s():.1f}",
                f"{r.peak_vram_gb:.1f}" if r.peak_vram_gb else "N/A",
                f"{r.quality_score.overall:.3f}" if r.quality_score else "N/A",
            )

        console.print(pf_table)


if __name__ == "__main__":
    app()
