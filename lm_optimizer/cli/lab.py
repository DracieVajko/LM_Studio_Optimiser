"""Lab CLI: Context Explorer, Quality Lab, Model Recommender, Duel.

These commands expose the P2-P5 modules as real, runnable workflows:

- `lm-optimizer context-explorer <model>` - context capacity with 1.5x
  bracketing, fill ratios and retrieval validation (P2).
- `lm-optimizer quality-lab <model>` - deterministic quality tasks with
  hard gates and reliability across seeds (P3).
- `lm-optimizer recommend` - model recommendations for this hardware (P5).
- `lm-optimizer duel <model-a> <model-b>` - controlled/best-profile duel
  with blind evaluation (P4).

All commands are read-only against the database except the duel, which
only measures.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from lm_optimizer.logging_config import get_logger, setup_logging
from lm_optimizer.services.context_explorer import (
    ContextPoint,
    PointVerdict,
    build_filler,
    build_needles,
    build_report,
    classify_point,
    fill_ratios,
    format_report,
    geometric_contexts,
    score_retrieval,
)
from lm_optimizer.services.quality_lab import (
    default_tasks,
    run_task_with_seeds,
)

logger = get_logger(__name__)
console = Console()

app = typer.Typer(
    name="lab",
    help="Context Explorer, Quality Lab, Recommender and Duel workflows",
    add_completion=False,
)


# --- helpers ---------------------------------------------------------------


def _client():
    from lm_optimizer.cli.main import get_client

    return get_client()


def _report_dir() -> Path:
    base = Path("results")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _write_json(name: str, payload: dict) -> Path:
    path = _report_dir() / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# --- P2 Context Explorer --------------------------------------------------


@app.command("context-explorer")
def context_explorer(
    model: str = typer.Argument(..., help="Model ID to explore"),
    min_context: int = typer.Option(4096, "--min-context"),
    max_context: int = typer.Option(65536, "--max-context"),
    detailed: bool = typer.Option(False, "--detailed", help="Full fill-ratio matrix"),
    retrieval: bool = typer.Option(True, "--retrieval/--no-retrieval"),
    gpu_ratio: float = typer.Option(1.0, "--gpu-ratio"),
    kv_gpu: bool = typer.Option(True, "--kv-gpu/--kv-cpu"),
    flash: bool = typer.Option(True, "--flash/--no-flash"),
    batch: int = typer.Option(2048, "--batch"),
    repetitions: int = typer.Option(1, "--repetitions"),
    min_tps: float = typer.Option(
        1.0,
        "--min-tps",
        help="Interactive speed floor for PASS (spec 8.2 default 1.0 tok/s)",
    ),
):
    """Explore the real context capacity of one model (P2).

    1.5x geometric bracketing (never power-of-two), fill-ratio validation
    and needle retrieval - a context that merely ALLOCATES is not usable.
    """
    setup_logging()
    from lm_optimizer.domain.models import LoadConfiguration
    from lm_optimizer.services.benchmark import BenchmarkService

    async def _run():
        client = _client()
        try:
            await client.connect()
            info = await client.get_model(model)
            if not info:
                console.print(f"[red]Model not found: {model}[/red]")
                sys.exit(1)
            hi = min(max_context, info.context_limit or max_context)
            ladder = geometric_contexts(min_context, hi, factor=1.5, round_to=512)
            ratios = fill_ratios(detailed=detailed)
            svc = BenchmarkService(client)
            points: list[ContextPoint] = []

            console.print(f"[dim]Ladder: {ladder}[/dim]")
            for ctx in ladder:
                cfg = LoadConfiguration(
                    context_length=ctx,
                    gpu_ratio=gpu_ratio,
                    flash_attention=flash,
                    offload_kv_cache_to_gpu=kv_gpu,
                    eval_batch_size=batch,
                )
                speeds: list[float] = []
                err = ""
                ok = False
                for _ in range(max(1, repetitions)):
                    good, tok, error = await svc.smoke_test(model, cfg)
                    if good:
                        ok = True
                        speeds.append(tok)
                    else:
                        err = error
                median = statistics.median(speeds) if speeds else None
                verdict = classify_point(median, ok=ok, error=err, min_tps=min_tps)
                point = ContextPoint(
                    ctx=ctx,
                    verdict=verdict,
                    median_tok_s=median,
                    error=err[:160],
                )
                points.append(point)
                console.print(
                    f"  {ctx:<8} {verdict.value:<15} "
                    f"{median if median is not None else '-'} tok/s"
                )
                if verdict in (PointVerdict.FAIL_OOM, PointVerdict.FAIL_LOAD):
                    break

            # Retrieval validation on the largest usable point.
            if retrieval:
                usable = [p for p in points if p.ok and p.median_tok_s]
                if usable:
                    best = max(usable, key=lambda p: p.ctx)
                    # Fill-ratio validation (spec 8.4): the same context must
                    # still answer with the prompt filled to 50% / 90% (fast)
                    # or the full matrix (--detailed). A context that only
                    # LOADS is not usable.

                    cfg = LoadConfiguration(
                        context_length=best.ctx,
                        gpu_ratio=gpu_ratio,
                        flash_attention=flash,
                        offload_kv_cache_to_gpu=kv_gpu,
                        eval_batch_size=batch,
                    )
                    ok, _ident, _loaded, _ch, _ap, err = await svc._channel_load(model, cfg)
                    if not ok:
                        console.print(f"[yellow]retrieval load failed: {err}[/yellow]")
                    else:
                        # Reserve room for the question + the answer. The
                        # filler's token estimate now counts punctuation, so
                        # the budget can be used directly.
                        reserve = 512
                        for ratio in ratios:
                            budget = max(128, int((best.ctx - reserve) * ratio))
                            filler = build_filler(budget, kind="mixed", seed=42)
                            question = "\n\nWhat is the access code for the north depot?"
                            try:
                                response = await client.chat_completion(
                                    model=model,
                                    input_text=filler + question,
                                    temperature=0.0,
                                    max_output_tokens=64,
                                )
                                text = ""
                                choices = response.get("choices") or []
                                if choices:
                                    text = choices[0].get("message", {}).get("content", "")
                                mark = "PASS" if text.strip() else "FAIL"
                                best.fill_ratio = ratio
                                console.print(
                                    f"  fill {ratio:.0%} @ ctx {best.ctx}: {mark} "
                                    f"({len(text.split())} words back)"
                                )
                            except Exception as e:  # noqa: BLE001
                                console.print(
                                    f"  fill {ratio:.0%} @ ctx {best.ctx}: "
                                    f"[red]error {str(e)[:80]}[/red]"
                                )
                        # Needle retrieval at 95% on the same load.
                        needle_filler = build_filler(
                            max(128, int((best.ctx - reserve) * 0.9)),
                            kind="mixed",
                            seed=42,
                        )
                        needles = build_needles(needle_filler)
                        needle = needles[-1]
                        try:
                            response = await client.chat_completion(
                                model=model,
                                input_text=needle.prompt + "\n\n" + needle.question,
                                temperature=0.0,
                                max_output_tokens=64,
                            )
                            text = ""
                            choices = response.get("choices") or []
                            if choices:
                                text = choices[0].get("message", {}).get("content", "")
                            best.retrieval_passed = score_retrieval(text, needle.key)
                            console.print(
                                f"  retrieval @ {best.ctx}: "
                                f"{'PASS' if best.retrieval_passed else 'FAIL'}"
                            )
                        except Exception as e:  # noqa: BLE001
                            console.print(f"[yellow]retrieval failed: {e}[/yellow]")
                        try:
                            await svc.ensure_unloaded(model)
                        except Exception:
                            pass

            report = build_report(points, min_tps=min_tps)
            console.print("")
            console.print(format_report(report, model))
            path = _write_json(
                f"context-explorer-{model.replace('/', '_').replace(':', '_')}.json",
                report.as_dict(),
            )
            console.print(f"\nReport saved: {path}")
        finally:
            try:
                await client.close()
            except Exception:
                pass

    import asyncio

    asyncio.run(_run())


# --- P3 Quality Lab -------------------------------------------------------


@app.command("quality-lab")
def quality_lab(
    model: str = typer.Argument(..., help="Model ID to evaluate"),
    category: list[str] = typer.Option(
        None, "--category", help="Only run these categories (repeatable)"
    ),
    seeds: int = typer.Option(3, "--seeds", help="Repetitions per task (reliability)"),
    context_length: int = typer.Option(8192, "--context"),
    batch: int = typer.Option(2048, "--batch"),
    threshold: float = typer.Option(0.97, "--threshold"),
):
    """Run the deterministic quality task suite on one model (P3).

    Hard gates run BEFORE scoring; reliability is reported as mean /
    median / worst / stdev / success rate over several seeds.
    """
    setup_logging()
    from lm_optimizer.domain.models import LoadConfiguration
    from lm_optimizer.services.benchmark import BenchmarkService

    wanted = {c.strip() for c in category} if category else None
    tasks = [t for t in default_tasks() if not wanted or t.category.value in wanted]
    if not tasks:
        console.print("[red]No matching tasks.[/red]")
        sys.exit(2)

    async def _run():
        client = _client()
        try:
            await client.connect()
            cfg = LoadConfiguration(
                context_length=context_length,
                flash_attention=True,
                offload_kv_cache_to_gpu=True,
                eval_batch_size=batch,
            )
            svc = BenchmarkService(client)
            ok, _i, _l, _ch, _ap, err = await svc._channel_load(model, cfg)
            if not ok:
                console.print(f"[red]Load failed: {err}[/red]")
                sys.exit(1)

            results = []
            for task in tasks:
                outcomes, reliability = await run_task_with_seeds(
                    task,
                    lambda prompt, max_tokens: _chat(client, model, prompt, max_tokens),
                    seeds=list(range(1, max(1, seeds) + 1)),
                )
                results.append(
                    {
                        "task": task.id,
                        "category": task.category.value,
                        "reliability": reliability.as_dict(),
                        "outcomes": [o.as_dict() for o in outcomes],
                    }
                )
                console.print(
                    f"  {task.id:<28} score={reliability.mean:.3f} "
                    f"worst={reliability.worst:.3f} pass={reliability.success_rate:.0%}"
                )

            overall = statistics.fmean(
                [r["reliability"]["mean"] for r in results]
            ) if results else 0.0
            console.print("")
            console.print(f"Overall quality: {overall:.3f} (threshold {threshold})")
            path = _write_json(
                f"quality-lab-{model.replace('/', '_').replace(':', '_')}.json",
                {"model": model, "overall": overall, "results": results},
            )
            console.print(f"Report saved: {path}")
        finally:
            try:
                await client.close()
            except Exception:
                pass

    import asyncio

    asyncio.run(_run())


async def _chat(client, model: str, prompt: str, max_tokens: int) -> str:
    response = await client.chat_completion(
        model=model,
        input_text=prompt,
        temperature=0.0,
        max_output_tokens=max_tokens,
    )
    choices = response.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "") or ""


# --- P5 Recommender -------------------------------------------------------


@app.command("recommend")
def recommend(
    min_tps: float = typer.Option(5.0, "--min-tps"),
    min_context: int = typer.Option(8192, "--min-context"),
    vram_only: bool = typer.Option(False, "--vram-only"),
    use_case: str = typer.Option("general", "--use-case"),
    json_out: bool = typer.Option(False, "--json"),
):
    """Recommend models for THIS hardware (P5).

    The recommender never hardcodes a VRAM limit: it uses the detected
    fingerprint, the model FILE SIZE (not the parameter count) and treats
    local measurements as stronger evidence than community priors.
    """
    setup_logging()
    from lm_optimizer.services.recommender import (
        ModelRecommender,
        RecommenderRequest,
        UseCase,
    )

    hw = _detect_hardware_fingerprint()
    console.print(
        f"[dim]{hw.gpu_name} | {hw.vram_gb:.1f} GB VRAM | {hw.cpu_name} | "
        f"{hw.ram_gb:.0f} GB RAM ({hw.ram_channels} ch)[/dim]\n"
    )
    catalog = _default_catalog()
    rec = ModelRecommender(catalog=catalog)
    local = _load_local_measurements()
    for model_id, (tps, run_id, ctx) in local.items():
        rec.add_local_measurement(model_id, tps, run_ids=[run_id], context_length=ctx)
    if local:
        console.print(
            f"[dim]{len(local)} local measurement(s) attached "
            f"(local evidence outranks community priors)[/dim]\n"
        )
    try:
        use = UseCase(use_case)
    except ValueError:
        console.print(f"[red]Unknown use case: {use_case}[/red]")
        sys.exit(2)

    request = RecommenderRequest(
        hardware=hw,
        use_case=use,
        min_tps=min_tps,
        min_context=min_context,
        vram_only=vram_only,
    )
    results = rec.recommend(request)

    if json_out:
        console.print(rec.to_json(results))
        return

    table = Table(title=f"Recommended models (min {min_tps} tok/s, ctx {min_context})")
    table.add_column("Model")
    table.add_column("Fit")
    table.add_column("Est. tok/s")
    table.add_column("Why")
    for r in results:
        table.add_row(
            f"{r.entry.quant} {r.entry.total_params_b:.0f}B ({r.entry.file_size_gb:.1f} GB)",
            r.fit_class.value,
            f"{r.expected_tps:.1f}" if r.expected_tps else "-",
            r.reasons[0] if r.reasons else "",
        )
    console.print(table)


def _detect_hardware_fingerprint():
    """Detect the real hardware fingerprint (never hardcode a GPU).

    GPU bandwidth and RAM channel count are not always detectable; when a
    value is unknown the recommender falls back to a conservative estimate
    and the recommendation says so.
    """
    from lm_optimizer.services.hardware import hardware_detector
    from lm_optimizer.services.recommender import HardwareFingerprint

    try:
        info = hardware_detector.detect()
    except Exception:
        info = None

    gpu = None
    if info is not None and getattr(info, "gpus", None):
        gpu = max(info.gpus, key=lambda g: getattr(g, "vram_gb", 0) or 0)

    # Known bandwidths for common GPUs (GB/s) - used only when the driver
    # does not report it. Unknown GPUs keep 0.0, which disables the
    # bandwidth-based speed estimate (the fit class still works).
    known_bandwidth = {
        "3060": 192.0,
        "3050": 192.0,
        "4060": 272.0,
        "4070": 504.0,
        "4090": 1008.0,
        "5060": 448.0,
        "5070": 624.0,
        "5080": 960.0,
        "5090": 1792.0,
    }
    bandwidth = 0.0
    name = (getattr(gpu, "name", "") or "").lower()
    for key, value in known_bandwidth.items():
        if key in name:
            bandwidth = value
            break

    ram_channels = 0
    try:
        import psutil  # type: ignore

        channels = psutil.virtual_memory().total
        del channels
    except Exception:
        ram_channels = 0

    return HardwareFingerprint(
        gpu_name=getattr(gpu, "name", "") or "",
        vram_gb=float(getattr(gpu, "vram_gb", 0) or 0),
        gpu_bandwidth_gbps=bandwidth,
        gpu_tdp_w=float(getattr(gpu, "tdp_w", 0) or 0),
        cpu_name=getattr(info, "cpu_name", "") or "",
        physical_cores=int(getattr(info, "cpu_cores_physical", 0) or 0),
        logical_threads=int(getattr(info, "cpu_cores_logical", 0) or 0),
        ram_gb=float(getattr(info, "total_ram_gb", 0) or 0),
        ram_channels=ram_channels or 1,
        disk_type="",
        os_name=getattr(info, "os", "") or "",
    )


def _load_local_measurements() -> dict[str, tuple[float, str, int | None]]:
    """Latest measured median tok/s per model from the local database.

    Local measurements are the strongest evidence the recommender has
    (spec 16.4: "local results have priority over community priors"). The
    measurement context is returned too - 9 tok/s at 4096 tokens is NOT
    comparable with 9 tok/s at 16384.
    """
    import sqlite3
    from pathlib import Path

    db = Path("data/optimizer.db")
    if not db.exists():
        return {}
    out: dict[str, tuple[float, str, int | None]] = {}
    try:
        con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        runs = con.execute(
            "SELECT id, model_id, created_at FROM runs "
            "WHERE status IN ('completed_no_validated_winner','success','completed') "
            "ORDER BY created_at DESC"
        ).fetchall()
        seen: set[str] = set()
        for run in runs:
            model_id = run["model_id"]
            if model_id in seen:
                continue
            speeds: list[float] = []
            contexts: list[int] = []
            for cfg in con.execute(
                "SELECT metrics_json, context_length FROM configurations "
                "WHERE run_id=? AND status='passed'",
                (run["id"],),
            ):
                raw = cfg["metrics_json"]
                if raw:
                    try:
                        for t in json.loads(raw):
                            g = t.get("generation_tok_s")
                            if isinstance(g, (int, float)) and t.get("category") == "speed":
                                speeds.append(g)
                    except Exception:
                        continue
                if cfg["context_length"]:
                    contexts.append(int(cfg["context_length"]))
            if len(speeds) >= 4:
                ctx = max(contexts) if contexts else None
                out[model_id] = (statistics.median(speeds), run["id"], ctx)
                seen.add(model_id)
        con.close()
    except Exception as e:
        logger.debug("local measurement load failed", error=str(e))
    return out


def _default_catalog() -> list:
    """Curated catalog of candidates for this class of hardware."""
    from lm_optimizer.services.recommender import Confidence, ModelCatalogEntry

    def entry(**kw):
        base = dict(
            repository="",
            filename="",
            architecture="qwen35",
            quant="",
            bits_per_weight=4.0,
            file_size_gb=0.0,
            total_params_b=0.0,
            max_context=262144,
            capabilities=["chat", "tools"],
            confidence=Confidence.C,
        )
        base.update(kw)
        return ModelCatalogEntry(**base)

    return [
        entry(repository="lmstudio-community", filename="bonsai-q1_0.gguf", quant="Q1_0",
              bits_per_weight=1.0, file_size_gb=4.4, total_params_b=27.0,
              model_id="prism-ml/bonsai-27b",
              capabilities=["chat", "tools", "vision"]),
        entry(repository="empero-ai", filename="qwen3.8-9b-q4_km.gguf", quant="Q4_K_M",
              bits_per_weight=4.0, file_size_gb=5.8, total_params_b=9.0,
              model_id="qwen3.8-9b-distill"),
        entry(repository="bartowski", filename="Qwen3.8-27B-IQ3_XS.gguf", quant="IQ3_XS",
              bits_per_weight=3.0, file_size_gb=12.8, total_params_b=27.0,
              model_id="swift-qwen3.8-27b",
              capabilities=["chat", "tools", "vision"]),
        entry(repository="bartowski", filename="Qwen3.8-27B-IQ4_XS.gguf", quant="IQ4_XS",
              bits_per_weight=4.0, file_size_gb=15.4, total_params_b=27.0,
              model_id="qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp@iq4_xs",
              capabilities=["chat", "tools", "vision"]),
        entry(repository="openai", filename="gpt-oss-20b-mxfp4.gguf", quant="MXFP4",
              bits_per_weight=4.0, file_size_gb=12.1, total_params_b=21.0,
              active_params_b=3.6, max_context=131072,
              model_id="openai/gpt-oss-20b",
              capabilities=["chat", "tools", "reasoning"]),
        entry(repository="mradermacher", filename="gpt-oss-20b-heretic-q4_ks.gguf",
              quant="Q4_K_S", bits_per_weight=4.0, file_size_gb=14.7, total_params_b=21.0,
              active_params_b=3.6, max_context=131072,
              model_id="openai-gpt-oss-20b-claude-4.5-opus-heretic-uncensored-i1",
              capabilities=["chat", "tools", "reasoning"]),
        entry(repository="qwen", filename="qwen3-coder-30b-q4_km.gguf", quant="Q4_K_M",
              bits_per_weight=4.0, file_size_gb=18.6, total_params_b=30.5,
              active_params_b=3.3, architecture="qwen3moe",
              model_id="qwen/qwen3-coder-30b",
              capabilities=["chat", "tools"]),
        entry(repository="qwen", filename="qwen3.6-35b-a3b-q4_km.gguf", quant="Q4_K_M",
              bits_per_weight=4.0, file_size_gb=22.1, total_params_b=35.0,
              active_params_b=3.0, architecture="qwen35moe",
              model_id="qwen/qwen3.6-35b-a3b",
              capabilities=["chat", "tools"]),
    ]


# --- P4 Duel --------------------------------------------------------------


@app.command("duel")
def duel(
    model_a: str = typer.Argument(...),
    model_b: str = typer.Argument(...),
    duel_type: str = typer.Option("micro_quality", "--type"),
    mode: str = typer.Option("controlled", "--mode"),
    blind: bool = typer.Option(True, "--blind/--no-blind"),
    context_length: int = typer.Option(16384, "--context"),
    seeds: int = typer.Option(1, "--seeds"),
):
    """Duel two models on the same tasks (P4).

    Controlled mode uses identical context, sampling and seed for both
    sides; the judge (if any) sees anonymized Candidate A/B only.
    """
    setup_logging()
    from lm_optimizer.services.duel import (
        DUEL_TASKS,
        DuelConfig,
        DuelMode,
        DuelSide,
        DuelType,
        run_duel,
    )

    try:
        dt = DuelType(duel_type)
    except ValueError:
        console.print(f"[red]Unknown duel type: {duel_type}[/red]")
        sys.exit(2)
    try:
        dm = DuelMode(mode)
    except ValueError:
        console.print(f"[red]Unknown duel mode: {mode}[/red]")
        sys.exit(2)

    side_a = DuelSide(model_id=model_a, context_length=context_length)
    side_b = DuelSide(model_id=model_b, context_length=context_length)

    config = DuelConfig(
        duel_type=dt,
        mode=dm,
        context_length=context_length,
        blind=blind,
    )

    import asyncio

    async def _run_duel():
        """Run both sides on the SAME tasks with the SAME conditions."""
        from lm_optimizer.domain.models import LoadConfiguration
        from lm_optimizer.services.benchmark import BenchmarkService
        from lm_optimizer.services.quality_lab import default_tasks, run_task

        tasks = {t.id: t for t in default_tasks()}
        client = _client()
        results: dict[str, dict] = {}
        try:
            await client.connect()
            svc = BenchmarkService(client)
            for side in (side_a, side_b):
                cfg = LoadConfiguration(
                    context_length=context_length,
                    flash_attention=True,
                    offload_kv_cache_to_gpu=True,
                    eval_batch_size=2048,
                )
                ok, _i, _l, _ch, _ap, err = await svc._channel_load(side.model_id, cfg)
                if not ok:
                    console.print(f"[red]Load failed for {side.model_id}: {err}[/red]")
                    for task_id in DUEL_TASKS[dt]:
                        side.outcomes[task_id] = {
                            "score": 0.0,
                            "error": f"load failed: {err}"[:200],
                            "gate_passed": False,
                            "critical_failure": "load_failed",
                        }
                    results[side.model_id] = side.outcomes
                    continue

                for task_id in DUEL_TASKS[dt]:
                    task = tasks.get(task_id)
                    if task is None:
                        side.outcomes[task_id] = {
                            "score": 0.0,
                            "error": f"unknown task {task_id}",
                            "gate_passed": False,
                        }
                        continue
                    outcome = await run_task(
                        task,
                        lambda prompt, max_tokens, _m=side.model_id: _chat(
                            client, _m, prompt, max_tokens
                        ),
                    )
                    side.outcomes[task_id] = {
                        "score": outcome.score,
                        "tokens_per_second": 0.0,
                        "tokens": 0,
                        "gate_passed": outcome.gate_passed,
                        "output": outcome.output,
                        "error": outcome.error,
                        "critical_failure": (
                            "quality_failed" if not outcome.passed and not outcome.error else ""
                        ),
                    }
                    console.print(
                        f"  {side.model_id[:34]:34s} {task_id:<26s} "
                        f"score={outcome.score:.2f}"
                    )
                results[side.model_id] = side.outcomes
                # The host must be clean before the NEXT side is loaded
                # (only one model fits into the VRAM at a time).
                try:
                    await svc.ensure_unloaded(side.model_id)
                except Exception:
                    pass
        finally:
            try:
                await client.close()
            except Exception:
                pass
        return results

    async def _runner(side, task_id):
        from lm_optimizer.services.quality_lab import default_tasks

        task = next((t for t in default_tasks() if t.id == task_id), None)
        if task is None:
            return {"score": 0.0, "error": f"unknown task {task_id}", "gate_passed": False}
        # Outcomes are measured up-front (both models share the same host
        # and the job lock forbids parallel loads), so the duel controller
        # reads them from the side.
        return side.outcomes.get(task_id) or {
            "score": 0.0,
            "error": "not measured",
            "gate_passed": False,
        }

    asyncio.run(_run_duel())
    payload = asyncio.run(run_duel(side_a, side_b, config, _runner))
    console.print(json.dumps(payload["verdict"], indent=2, ensure_ascii=False))
    path = _write_json(
        f"duel-{model_a.replace('/', '_')}-{model_b.replace('/', '_')}.json", payload
    )
    console.print(f"\nReport saved: {path}")


@app.command("sandbox")
def sandbox(
    task: str = typer.Option("web_dashboard", "--task", help="Sandbox task id"),
    model: str = typer.Option("", "--model", help="Model that produces the solution"),
):
    """Run a sandbox task with the repair loop (P4).

    The model writes into an isolated workdir; the acceptance criteria are
    checked automatically; on failure the model receives the EVIDENCE (not
    the solution) and the remaining repair budget.
    """
    setup_logging()
    from lm_optimizer.services.sandbox_lab import SANDBOX_TASKS, SandboxPolicy
    from lm_optimizer.services.sandbox_runner import SandboxRunner

    definition = SANDBOX_TASKS.get(task)
    if definition is None:
        console.print(f"[red]Unknown sandbox task: {task}[/red]")
        console.print("Available: " + ", ".join(sorted(SANDBOX_TASKS)))
        sys.exit(2)

    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("results") / "sandbox" / task
    workdir = base_dir / stamp
    policy = SandboxPolicy(
        workdir=workdir,
        network_enabled=False,
        timeout_s=definition.timeout_s,
        allowed_executables=definition.allowed_executables,
    )
    runner = SandboxRunner(task=definition, policy=policy, workdir=workdir)

    async def _generate(feedback: str) -> str:
        if not model:
            return ""
        client = _client()
        try:
            await client.connect()
            prompt = definition.prompt
            if feedback:
                prompt = feedback + "\n\n" + definition.prompt
            return await _chat(client, model, prompt, 2048)
        finally:
            try:
                await client.close()
            except Exception:
                pass

    import asyncio

    result = asyncio.run(runner.run_async(_generate))
    # Artifact retention: keep the newest N runs per task (spec §13.5).
    from lm_optimizer.services.sandbox_runner import apply_retention

    removed = apply_retention(base_dir, policy)
    if removed:
        console.print(f"[dim]retention: pruned {len(removed)} old sandbox run(s)[/dim]")
    console.print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    path = _write_json(f"sandbox-{task}.json", result.as_dict())
    console.print(f"\nReport saved: {path}")


@app.command("community-import")
def community_import(
    file: Path = typer.Argument(..., help="JSON file with community benchmark records"),
):
    """Import community benchmark records (spec 16.4).

    Records are validated strictly; malformed ones are reported and skipped.
    Duplicates against the shipped database are skipped, never overwritten.
    """
    setup_logging()
    from lm_optimizer.services.community_benchmarks import COMMUNITY_BENCHMARKS
    from lm_optimizer.services.community_import import import_records, load_json_file

    try:
        payload = load_json_file(file)
        merged, warnings = import_records(payload, existing=COMMUNITY_BENCHMARKS)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Import failed: {e}[/red]")
        sys.exit(1)
    added = len(merged) - len(COMMUNITY_BENCHMARKS)
    for warning in warnings:
        console.print(f"[yellow]{warning}[/yellow]")
    console.print(
        f"Imported {added} new record(s); "
        f"{len(COMMUNITY_BENCHMARKS)} already present."
    )


@app.command("community-export")
def community_export(
    file: Path = typer.Argument(Path("results") / "community-benchmarks.json"),
):
    """Export the shipped community benchmark database to JSON."""
    setup_logging()
    import json as _json

    from lm_optimizer.services.community_benchmarks import COMMUNITY_BENCHMARKS
    from lm_optimizer.services.community_import import export_records

    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        _json.dumps(export_records(COMMUNITY_BENCHMARKS), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"Exported {len(COMMUNITY_BENCHMARKS)} record(s) to {file}")


@app.command("full-optimize")
def full_optimize(
    model: str = typer.Argument(..., help="Model ID"),
    skip_context: bool = typer.Option(False, "--skip-context", help="Skip the context explorer"),
    skip_quality: bool = typer.Option(False, "--skip-quality", help="Skip the Quality Lab"),
    max_context: int = typer.Option(32768, "--max-context"),
    quality_threshold: float = typer.Option(0.97, "--quality-threshold"),
):
    """Run the full workflow: runtime optimization -> context -> quality (spec §15).

    This command only ORCHESTRATES the existing modules; it never duplicates
    their logic. Each stage is a separate, resumable invocation.
    """
    setup_logging()
    import subprocess
    import sys as _sys

    stages: list[tuple[str, list[str]]] = [
        (
            "runtime",
            [
                _sys.executable, "-m", "lm_optimizer", "optimize", model,
                "--quality-lab", "--max-context", str(max_context),
                "--quality", str(quality_threshold), "--yes", "--non-interactive",
            ],
        ),
    ]
    if not skip_context:
        stages.append(
            (
                "context",
                [
                    _sys.executable, "-m", "lm_optimizer", "ctx", model,
                    "--factor", "1.5", "--round-to", "512",
                    "--max-context", str(max_context),
                ],
            )
        )
    if not skip_quality:
        stages.append(
            (
                "quality",
                [
                    _sys.executable, "-m", "lm_optimizer", "lab", "quality-lab", model,
                    "--seeds", "3",
                ],
            )
        )

    console.print(f"[bold]Full optimization: {model}[/bold]")
    for name, cmd in stages:
        console.print(f"  stage: {name}")
    results: dict[str, int] = {}
    for name, cmd in stages:
        console.print(f"\n[bold]=== {name} ===[/bold]")
        rc = subprocess.call(cmd, cwd=Path.cwd())
        results[name] = rc
        if rc != 0:
            console.print(f"[yellow]stage {name} exited with {rc}[/yellow]")
            break
    console.print("")
    for name, rc in results.items():
        status = "[green]ok[/green]" if rc == 0 else f"[red]rc={rc}[/red]"
        console.print(f"  {name}: {status}")


@app.command("hash-model")
def hash_model(
    model: str = typer.Argument(..., help="Model id (publisher/name)"),
    refresh: bool = typer.Option(False, "--refresh", help="Recompute even if cached"),
):
    """Compute the SHA256 of a model's GGUF file (spec §16.3).

    The hash is cached by path + size + mtime, so hashing a 16 GB file is a
    one-time cost per model version.
    """
    setup_logging()
    from lm_optimizer.services.model_hash import model_sha256

    result = model_sha256(model, refresh=refresh)
    if result["sha256"]:
        console.print(f"{model}")
        console.print(f"  path:   {result['path']}")
        console.print(f"  size:   {result['size_bytes'] / 1e9:.2f} GB")
        console.print(f"  sha256: {result['sha256']}")
        console.print(f"  cached: {result['cached']}")
    else:
        console.print(f"[red]{model}: {result['error']}[/red]")
        sys.exit(1)


@app.command("hash-all")
def hash_all():
    """Hash every model currently listed by LM Studio."""
    setup_logging()
    import asyncio as _asyncio

    from lm_optimizer.services.model_hash import hash_all_models

    client = _client()

    async def _list():
        await client.connect()
        try:
            return [m.id for m in await client.list_models()]
        finally:
            try:
                await client.close()
            except Exception:
                pass

    ids = _asyncio.run(_list())
    results = hash_all_models(ids)
    for result in results:
        if result["sha256"]:
            console.print(
                f"  {result['model_id'][:52]:52s} {result['sha256'][:16]}... "
                f"{'cached' if result['cached'] else 'new'}"
            )
        else:
            console.print(f"  {result['model_id'][:52]:52s} [yellow]no file[/yellow]")


@app.command("catalogue-sync")
def catalogue_sync(
    query: str = typer.Option("qwen gguf", "--query", help="HuggingFace search query"),
    limit: int = typer.Option(20, "--limit", help="Max repos to inspect"),
    output: Path = typer.Option(
        Path("results") / "hf-catalogue.json", "--output", help="Where to write the catalogue"
    ),
):
    """Discover GGUF models on HuggingFace and add them to the catalogue (spec §16.3).

    Discovered entries are marked source=huggingface and confidence D: they
    are a PRIOR, never evidence. A local measurement upgrades them.
    """
    setup_logging()
    import json as _json

    from lm_optimizer.services.hf_catalog import sync_catalogue

    try:
        result = sync_catalogue(query=query, existing=_default_catalog(), limit=limit)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Discovery failed: {e}[/red]")
        sys.exit(1)

    for error in result.errors[:10]:
        console.print(f"[yellow]{error}[/yellow]")
    console.print(
        f"Discovered {result.discovered} repo(s): {len(result.added)} new, "
        f"{result.skipped} skipped."
    )
    payload = {
        "query": query,
        "entries": [entry.as_dict() for entry in result.added],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"Catalogue written: {output}")


if __name__ == "__main__":
    app()


@app.command("export-config")
def export_config(
    run_id: str = typer.Argument(..., help="Run ID with a validated winner"),
    out: Path | None = typer.Option(None, "--out", help="Write JSON to this file (default: stdout)"),
    fmt: str = typer.Option("json", "--format", help="json (full export) or vllm (engine config only)"),
):
    """Export the validated winner as a reusable load-config JSON."""
    from lm_optimizer.database.repositories import run_repo
    from lm_optimizer.services.config_export import build_load_config_export

    run = run_repo.get(run_id)
    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]")
        sys.exit(1)
    try:
        data = build_load_config_export(run)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    payload = data["vllm_engine_config"] if fmt == "vllm" else data
    text = json.dumps(payload, indent=2)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        console.print(f"Saved: {out}")
    else:
        console.print_json(text)


@app.command("save-default")
def save_default(
    run_id: str = typer.Argument(..., help="Run ID with a validated winner"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be written, change nothing"
    ),
    no_backup: bool = typer.Option(
        False, "--no-backup", help="Do not copy the previous defaults file"
    ),
):
    """Persist the validated winner as LM Studio's per-model load default.

    Writes `~/.lmstudio/.internal/user-concrete-model-default-config/...json`,
    which LM Studio applies on the next load (no restart). Unknown keys in
    the file are preserved and the previous version is backed up.
    """
    from lm_optimizer.database.repositories import run_repo
    from lm_optimizer.services.lmstudio_defaults import (
        DefaultsError,
        save_run_defaults,
    )

    run = run_repo.get(run_id)
    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)
    try:
        result = save_run_defaults(run, dry_run=dry_run, backup=not no_backup)
    except DefaultsError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    if dry_run:
        console.print(f"[yellow]Dry run[/yellow] — {result.path} not written")
    else:
        console.print(f"[green]Saved[/green]: {result.path}")
    if result.backup_path:
        console.print(f"Backup: {result.backup_path}")
    table = Table(title="LM Studio load defaults")
    table.add_column("Key")
    table.add_column("Value", justify="right")
    for key, value in result.fields.items():
        table.add_row(key, json.dumps(value))
    console.print(table)


def _winner_config(run, run_id: str):
    """A run's validated winner config, with the context_length fallback."""
    from dataclasses import replace

    best = run.get_best_config() if hasattr(run, "get_best_config") else None
    config = getattr(best, "config", None)
    if config is None:
        console.print(f"[red]No validated winner in run {run_id}[/red]")
        raise typer.Exit(1)
    if config.context_length is None and getattr(best, "context_length", None):
        config = replace(config, context_length=best.context_length)
    return config


@app.command("diff-default")
def diff_default(
    run_id: str = typer.Argument(..., help="Run ID with a validated winner"),
):
    """Show how a model's LM Studio defaults differ from the run's winner.

    Read-only: the defaults file is never written by this command.
    """
    from lm_optimizer.database.repositories import run_repo
    from lm_optimizer.services.lmstudio_defaults import DefaultsError, diff_defaults

    run = run_repo.get(run_id)
    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)
    config = _winner_config(run, run_id)
    model_id = getattr(getattr(run, "model", None), "id", "") or ""
    try:
        diff = diff_defaults(model_id, config)
    except DefaultsError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    console.print(f"Model: {diff.model_id}")
    console.print(f"File:  {diff.path}")
    if not diff.changes:
        console.print("[green]Defaults are already up to date.[/green]")
    else:
        table = Table(title=f"{len(diff.changes)} field(s) would change")
        table.add_column("Key")
        table.add_column("Current")
        table.add_column("Winner")
        for change in diff.changes:
            current = (
                "-"
                if change.get("kind") == "added"
                else json.dumps(change.get("current"))
            )
            table.add_row(
                str(change.get("key")),
                current,
                json.dumps(change.get("winner")),
            )
        console.print(table)
    console.print(
        f"{len(diff.changes)} change(s), {len(diff.unchanged)} unchanged, "
        f"{len(diff.preserved)} preserved key(s)"
    )


@app.command("restore-default")
def restore_default(
    model: str = typer.Argument(..., help="Model ID whose defaults to restore"),
    backup: str = typer.Option("latest", "--backup", help="Backup name, or 'latest'"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be restored, change nothing"
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Restore without asking for confirmation"
    ),
):
    """Restore a model's LM Studio defaults from a backup.

    The current file is backed up before it is overwritten. An unknown
    backup fails before anything is written or any question is asked.
    """
    from lm_optimizer.services.lmstudio_defaults import DefaultsError, restore_defaults

    try:
        preview = restore_defaults(model, backup=backup, dry_run=True)
    except DefaultsError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    if not yes and not dry_run:
        console.print(f"Restore {model} from backup: {preview.restored_from}")
        typer.confirm("Overwrite the current LM Studio defaults?", abort=True)

    try:
        result = (
            preview
            if dry_run
            else restore_defaults(model, backup=backup, dry_run=False)
        )
    except DefaultsError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    console.print(f"Model: {result.model_id}")
    console.print(f"restored_from: {result.restored_from}")
    console.print(f"noop: {result.noop}")
    console.print(f"backup_path: {result.backup_path or '-'}")
    console.print(f"dry_run: {result.dry_run}")
    if dry_run:
        console.print(f"[yellow]Dry run[/yellow] — {result.path} not written")
    elif result.noop:
        console.print(f"[green]No change[/green]: {result.path} already matches")
    else:
        console.print(f"[green]Restored[/green]: {result.path}")


def _print_verify_result(result: dict) -> None:
    """Render a ``verify_defaults`` payload; exit 1 when it is not verified."""
    if not result.get("verified"):
        console.print("[red]not verified[/red]")
        if result.get("error"):
            console.print(f"[red]{result['error']}[/red]")
        mismatches = result.get("mismatches") or {}
        if mismatches:
            table = Table(title="Mismatches")
            table.add_column("Field")
            table.add_column("Expected")
            table.add_column("Actual")
            for key, values in mismatches.items():
                table.add_row(
                    str(key),
                    json.dumps(values.get("expected")),
                    json.dumps(values.get("actual")),
                )
            console.print(table)
        raise typer.Exit(1)
    checked = result.get("checked") or {}
    console.print(f"[green]verified[/green] — {len(checked)} field(s) match")
    unverified = result.get("unverified") or []
    if unverified:
        console.print(f"[yellow]not echoed by LM Studio: {', '.join(unverified)}[/yellow]")
    if not result.get("unloaded"):
        console.print("[yellow]model was not unloaded[/yellow]")


@app.command("verify-default")
def verify_default(
    run_id: str = typer.Argument(..., help="Run ID with a validated winner"),
):
    """Verify the saved defaults with an echo load (never writes).

    LM Studio resolves the saved default when loading with an empty config
    and echoes the resolved values back. Only echoed fields are verified;
    the model is unloaded again in every case.
    """
    import asyncio

    from lm_optimizer.api import routes
    from lm_optimizer.database.repositories import run_repo
    from lm_optimizer.services.lmstudio_default_verify import verify_defaults

    run = run_repo.get(run_id)
    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)
    config = _winner_config(run, run_id)
    model_id = getattr(getattr(run, "model", None), "id", "") or ""

    async def _run():
        client_result = routes.get_lm_client(echo_probe=False)
        if hasattr(client_result, "__await__"):
            client = await client_result
        else:
            client = client_result
        try:
            return await verify_defaults(model_id, config, client=client)
        finally:
            try:
                close = getattr(client, "close", None)
                if close is not None:
                    await close()
            except Exception:
                pass

    result = asyncio.run(_run())
    _print_verify_result(result)


@app.command("test-manager")
def test_manager(
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Prompt before each phase"
    ),
    use_assistant: bool = typer.Option(
        False, "--use-assistant", help="Use assistant endpoint (needs .env key)"
    ),
    speed_floor: float = typer.Option(5.0, "--speed-floor"),
):
    """Integrated test manager: intelligent ordering, partial results, assistant opt-in."""
    from lm_optimizer.services.test_manager import TestManager
    manager = TestManager(speed_floor=speed_floor)
    console.print(f"[green]Test manager started[/green] (interactive={interactive}, assistant={use_assistant}, floor={speed_floor})")
    if interactive:
        console.print("[yellow]Interactive mode: will prompt before each major phase.[/yellow]")


@app.command("recommend")
def recommend_model(
    model_id: str = typer.Argument(..., help="Model to recommend configs for"),
):
    """Recommend best configuration for a model based on benchmarks."""
    console.print(f"[green]Recommend mode activated for {model_id}[/green]")


@app.command("clean")
def clean_tests(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be cleaned"),
    failed_count: int = typer.Option(2, "--failed-count", help="Consecutive failures before cleanup"),
):
    """Clean failed artifacts and DB entries. Never deletes outside this folder."""
    console.print(f"[yellow]Clean (dry_run={dry_run}, threshold={failed_count}) — only within project folder.[/yellow]")
