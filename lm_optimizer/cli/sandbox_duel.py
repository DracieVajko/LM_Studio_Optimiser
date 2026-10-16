"""Sandbox duel: compare two models on the same prompt."""

import asyncio
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from lm_optimizer.database.repositories import config_repo, run_repo
from lm_optimizer.domain.models import (
    ConfigurationStatus,
    HardwareInfo,
    LoadConfiguration,
    RunStatus,
)
from lm_optimizer.logging_config import get_logger, setup_logging
from lm_optimizer.services.benchmark import BenchmarkService
from lm_optimizer.services.lm_studio import LMStudioClient
from lm_optimizer.services.quality import QualityConfig, QualityEvaluator

logger = get_logger(__name__)
console = Console()


@dataclass
class DuelRun:
    """One measured run of a prompt on one model."""

    ttft_ms: float
    generation_tok_s: float
    prompt_tok_s: float
    quality_overall: float | None
    passed: bool
    error: str | None = None


@dataclass
class DuelModelResult:
    """Measured results for one side of the duel."""

    model_id: str
    config: LoadConfiguration
    context_length: int
    runs: list[DuelRun] = field(default_factory=list)
    error: str | None = None

    def median_ttft(self) -> float:
        vals = [r.ttft_ms for r in self.runs if r.ttft_ms > 0]
        return statistics.median(vals) if vals else 0.0

    def median_gen(self) -> float:
        vals = [r.generation_tok_s for r in self.runs if r.generation_tok_s > 0]
        return statistics.median(vals) if vals else 0.0

    def median_quality(self) -> float:
        vals = [r.quality_overall for r in self.runs if r.quality_overall is not None]
        return statistics.median(vals) if vals else 0.0


@dataclass
class DuelResult:
    """Full duel outcome (both sides + verdict)."""

    prompt: str
    model_a: DuelModelResult
    model_b: DuelModelResult

    def verdict(self) -> dict:
        """Per-metric winner + margin.

        A metric with no measurement on either side (or a side that failed to
        load) yields `winner=None`: absent data must never read as a win.
        """
        out: dict = {}
        a_ttft, b_ttft = self.model_a.median_ttft(), self.model_b.median_ttft()
        a_gen, b_gen = self.model_a.median_gen(), self.model_b.median_gen()
        a_q, b_q = self.model_a.median_quality(), self.model_b.median_quality()

        out["TTFT"] = self._compare(a_ttft, b_ttft, lower_is_better=True)
        out["Generation speed"] = self._compare(a_gen, b_gen, lower_is_better=False)
        if a_q or b_q:
            out["Quality"] = self._compare(a_q, b_q, lower_is_better=False)
        return out

    @staticmethod
    def _compare(a: float, b: float, lower_is_better: bool) -> dict:
        """Compare one metric. Ties and missing data produce no winner."""
        if a <= 0.0 and b <= 0.0:
            return {"winner": None, "margin_pct": 0.0}
        if a <= 0.0:
            return {"winner": "b", "margin_pct": 0.0}
        if b <= 0.0:
            return {"winner": "a", "margin_pct": 0.0}

        if lower_is_better:
            winner = "a" if a < b else ("b" if b < a else None)
            lo, hi = min(a, b), max(a, b)
        else:
            winner = "a" if a > b else ("b" if b > a else None)
            lo, hi = min(a, b), max(a, b)
        margin = ((hi - lo) / hi * 100.0) if hi > 0 else 0.0
        return {"winner": winner, "margin_pct": round(margin, 1)}


def _require_style(style: str) -> str:
    valid = ["precise", "balanced", "creative"]
    if style not in valid:
        console.print(f"[red]Invalid style: {style}[/red]")
        console.print(f"Expected one of: {', '.join(valid)}")
        raise typer.Exit(2)
    return style


def _load_best_config(model_id: str) -> tuple[LoadConfiguration, int]:
    """Load the validated winner config for a model from its latest run.

    Only PASSED winners qualify: a run that never reached a validated winner
    cannot be dueled (refuses to guess a config).
    """
    runs = run_repo.get_by_model(model_id, limit=20)
    if not runs:
        raise ValueError(f"No runs for model {model_id!r}; run auto/optimize first.")

    for run in runs:
        if run.status not in (
            RunStatus.SUCCESS,
            RunStatus.COMPLETED,
            RunStatus.PARTIAL_SUCCESS,
        ):
            continue
        full = run_repo.get(str(run.id))
        if full is None or not full.best_config_id:
            continue
        best = config_repo.get(str(full.best_config_id))
        if best is None or best.status != ConfigurationStatus.PASSED:
            continue
        return best.config, best.context_length

    raise ValueError(
        f"No validated winner for {model_id!r}: every run failed or has no "
        f"passing winner. Run auto-run/optimize first."
    )


async def _measure_side(
    client: LMStudioClient,
    model_id: str,
    cfg: LoadConfiguration,
    ctx: int,
    prompt: str,
    repetitions: int,
    quality: QualityEvaluator,
) -> DuelModelResult:
    """Run the prompt N times against one model's validated config.

    `BenchmarkService.run_cases` owns the load/verify/unload skeleton, so each
    repetition is a fresh full cycle: the host never holds two models at once
    and a failed repetition cannot poison the next one.
    """
    from lm_optimizer.benchmark.suite import BenchmarkCase

    result = DuelModelResult(model_id=model_id, config=cfg, context_length=ctx)
    service = BenchmarkService(client, style="balanced")

    for _i in range(repetitions):
        case = BenchmarkCase(
            name="duel_prompt",
            category="duel",
            prompt=prompt,
            max_tokens=256,
            temperature=0.7,
        )
        try:
            res = await service.run_cases(model_id, cfg, ctx, [case])
        except Exception as e:
            result.runs.append(
                DuelRun(
                    ttft_ms=0.0,
                    generation_tok_s=0.0,
                    prompt_tok_s=0.0,
                    quality_overall=None,
                    passed=False,
                    error=f"{type(e).__name__}: {e}"[:200],
                )
            )
            continue

        m = res.metrics[0] if getattr(res, "metrics", None) else None
        q_overall = None
        if m is not None:
            try:
                agg = quality.aggregate_quality(quality.evaluate_all(model_id, res.metrics))
                q_overall = agg.overall
            except Exception:
                q_overall = None
        result.runs.append(
            DuelRun(
                ttft_ms=getattr(m, "estimated_ttft_ms", 0.0) if m else 0.0,
                generation_tok_s=getattr(m, "generation_tok_s", 0.0) if m else 0.0,
                prompt_tok_s=getattr(m, "prompt_tok_s", 0.0) if m else 0.0,
                quality_overall=q_overall,
                passed=getattr(res.status, "value", str(getattr(res, "status", ""))) == "passed",
                error=getattr(res, "error", None),
            )
        )

    return result


def generate_duel_report(result: DuelResult, output_dir: Path, env: dict) -> str:
    """Markdown duel report: configs, per-run tables, medians, verdict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_a = result.model_a.model_id.replace("/", "_")
    safe_b = result.model_b.model_id.replace("/", "_")
    path = output_dir / f"duel-{safe_a}-vs-{safe_b}-{stamp}.md"

    a, b = result.model_a, result.model_b
    lines = [
        f"# Duel: {a.model_id} vs {b.model_id}",
        "",
        "## Configurations",
        "| Model | Context | GPU | Flash | KV | Eval batch | Physical batch | Parallel | Checkpoints |",
        "|-------|---------|-----|-------|-----|------------|----------------|----------|-------------|",
    ]
    for side in (a, b):
        c = side.config
        lines.append(
            f"| {side.model_id} | {side.context_length} | {c.gpu_ratio} | "
            f"{c.flash_attention} | {c.offload_kv_cache_to_gpu} | {c.eval_batch_size} | "
            f"{c.physical_batch_size} | {c.parallel} | {c.context_checkpoints} |"
        )
    lines.extend(["", "## Prompt", "", "> " + result.prompt.replace("\n", " "), ""])

    lines.extend(
        [
            "## Results",
            "",
            "| Model | Run | TTFT (ms) | Gen tok/s | Prompt tok/s | Quality | Pass |",
            "|-------|-----|-----------|-----------|--------------|---------|------|",
        ]
    )
    for side in (a, b):
        if side.error:
            lines.append(f"| {side.model_id} | - | - | - | - | - | ERROR: {side.error} |")
            continue
        for idx, r in enumerate(side.runs, 1):
            q = f"{r.quality_overall:.3f}" if r.quality_overall is not None else "n/a"
            lines.append(
                f"| {side.model_id} | {idx} | {r.ttft_ms:.0f} | {r.generation_tok_s:.1f} | "
                f"{r.prompt_tok_s:.1f} | {q} | {'yes' if r.passed else 'no'} |"
            )

    lines.extend(
        [
            "",
            "## Medians",
            "",
            "| Model | TTFT (ms) | Gen tok/s | Quality |",
            "|-------|-----------|-----------|---------|",
            f"| {a.model_id} | {a.median_ttft():.0f} | {a.median_gen():.1f} | {a.median_quality():.3f} |",
            f"| {b.model_id} | {b.median_ttft():.0f} | {b.median_gen():.1f} | {b.median_quality():.3f} |",
            "",
            "## Verdict",
            "",
            "| Metric | Winner | Margin |",
            "|--------|--------|--------|",
        ]
    )
    verdict = result.verdict()
    label = {"a": a.model_id, "b": b.model_id}
    for metric, v in verdict.items():
        who = label.get(v["winner"], "n/a (no data)")
        lines.append(f"| {metric} | {who} | {v['margin_pct']}% |")

    speed_wins = [
        m for m in ("TTFT", "Generation speed")
        if verdict.get(m, {}).get("winner") is not None
    ]
    if speed_wins:
        wins_a = sum(1 for m in speed_wins if verdict[m]["winner"] == "a")
        overall = a.model_id if wins_a > len(speed_wins) / 2 else b.model_id
        lines.extend(["", f"**Overall speed winner: {overall}**"])
    else:
        lines.extend(["", "**Overall speed winner: undetermined (insufficient measurements)**"])

    lines.extend(
        [
            "",
            "## Environment",
            f"- **GPU**: {env.get('gpu', 'unknown')}",
            f"- **CPU**: {env.get('cpu', 'unknown')}",
            f"- **LM Studio**: {env.get('lms', 'unknown')}",
            "",
        ]
    )

    content = "\n".join(lines)
    path.write_text(content, encoding="utf-8")
    return content


def sandbox_duel(
    model_a: str = typer.Option(..., "--model-a", help="First model ID"),
    model_b: str = typer.Option(..., "--model-b", help="Second model ID"),
    prompt: str = typer.Option(..., "--prompt", help="Prompt to run on both models"),
    repetitions: int = typer.Option(3, "--repetitions", "-r", help="Runs per model"),
    style: str = typer.Option("balanced", "--style", help="Benchmark style"),
    output: Path = typer.Option(Path("results/duels"), "--output", help="Output directory"),
    quality: float = typer.Option(0.97, "--quality", "-q", help="Quality threshold"),
):
    """Compare two models on the same prompt using each one's validated winner."""
    setup_logging()
    style = _require_style(style)

    try:
        cfg_a, ctx_a = _load_best_config(model_a)
        cfg_b, ctx_b = _load_best_config(model_b)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Duel:[/bold] {model_a} (ctx {ctx_a}) vs {model_b} (ctx {ctx_b})")

    async def _run():
        client = LMStudioClient()
        try:
            await client.connect()
        except Exception as e:
            console.print(f"[red]Cannot connect to LM Studio: {e}[/red]")
            raise typer.Exit(1)

        quality_eval = QualityEvaluator(QualityConfig(minimum_score=quality))
        try:
            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
            ) as progress:
                t1 = progress.add_task(f"Measuring {model_a}...", total=repetitions)
                side_a = await _measure_side(
                    client, model_a, cfg_a, ctx_a, prompt, repetitions, quality_eval
                )
                progress.update(t1, completed=repetitions)
                t2 = progress.add_task(f"Measuring {model_b}...", total=repetitions)
                side_b = await _measure_side(
                    client, model_b, cfg_b, ctx_b, prompt, repetitions, quality_eval
                )
                progress.update(t2, completed=repetitions)
        finally:
            await client.close()

        return DuelResult(prompt=prompt, model_a=side_a, model_b=side_b)

    result = asyncio.run(_run())

    table = Table(title="Duel Result")
    table.add_column("Metric", style="cyan")
    table.add_column(model_a, style="green")
    table.add_column(model_b, style="green")
    table.add_column("Winner", style="yellow")
    table.add_row(
        "TTFT (ms)", f"{result.model_a.median_ttft():.0f}", f"{result.model_b.median_ttft():.0f}", ""
    )
    table.add_row(
        "Gen tok/s",
        f"{result.model_a.median_gen():.1f}",
        f"{result.model_b.median_gen():.1f}",
        "",
    )
    table.add_row(
        "Quality",
        f"{result.model_a.median_quality():.3f}",
        f"{result.model_b.median_quality():.3f}",
        "",
    )
    label = {"a": model_a, "b": model_b}
    for metric, v in result.verdict().items():
        who = label.get(v["winner"], "n/a (no data)")
        table.add_row(metric, "", "", f"{who} (+{v['margin_pct']}%)")
    console.print(table)

    from lm_optimizer.services.hardware import hardware_detector

    hw = hardware_detector.detect()
    env = {
        "gpu": ", ".join(f"{g.name} {g.vram_gb:.1f}GB" for g in hw.gpus) or "CPU-only",
        "cpu": f"{hw.cpu_name} ({hw.cpu_cores_physical}P/{hw.cpu_cores_logical}L)",
        "lms": _safe_lms_version(),
    }
    generate_duel_report(result, output, env)
    console.print(f"\n[green]Report: {output}[/green]")


def _safe_lms_version() -> str:
    from lm_optimizer.services import lms_cli

    try:
        return lms_cli.lms_version()
    except Exception:
        return "unknown"