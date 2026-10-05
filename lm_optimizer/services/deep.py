"""Single-model deep-research benchmark runner.

Each model runs on its stored best config (run DB lookup); a model with no
stored best is refused with a reason, never defaulted silently. Thinking
traces are best-effort (``n/a`` where absent, never a failure). Per-model
budget caps: at most ``MAX_OUTPUT_TOKENS_PER_TASK`` output tokens per task;
the run aborts after ``MAX_CONSECUTIVE_ERRORS`` consecutive task errors.

Reports reuse the full-visibility helpers (``_all_configs_rows``,
``_config_outputs``) and are ASCII-only.
"""

import asyncio
from datetime import datetime
from pathlib import Path

from lm_optimizer.benchmark.deep_suite import DEEP_CASES, deep_metrics
from lm_optimizer.config import config as _app_config
from lm_optimizer.domain.models import (
    BenchmarkCase,
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
)
from lm_optimizer.logging_config import get_logger
from lm_optimizer.services.benchmark import BenchmarkService
from lm_optimizer.services.quality import QualityEvaluator
from lm_optimizer.services.reporting import (
    _all_configs_rows,
    _config_outputs,
    sanitize_model_filename,
)

logger = get_logger(__name__)

MAX_OUTPUT_TOKENS_PER_TASK = 8192
MAX_CONSECUTIVE_ERRORS = 3


class NoBestConfigError(ValueError):
    """No stored best config for the model (refuse, never silent default)."""


def cap_deep_cases(
    cases: list[BenchmarkCase] | None = None,
    cap: int = MAX_OUTPUT_TOKENS_PER_TASK,
) -> list[BenchmarkCase]:
    """Deep cases with the per-task output token budget applied."""
    out = []
    for c in cases or DEEP_CASES:
        out.append(
            BenchmarkCase(
                name=c.name,
                category=c.category,
                prompt=c.prompt,
                max_tokens=min(c.max_tokens, cap),
                temperature=c.temperature,
                top_p=c.top_p,
                top_k=c.top_k,
                stop_sequences=c.stop_sequences,
            )
        )
    return out


def resolve_best_config(model_id: str, repo=None) -> tuple:
    """Newest stored best (load_config, context_length, run_id) for a model.

    Raises NoBestConfigError with a reason when nothing measurable exists.
    """
    if repo is None:
        from lm_optimizer.database.repositories import run_repo as repo
    try:
        lean = repo.get_by_model(model_id, 50)
    except Exception as e:
        raise NoBestConfigError(
            f"no stored best config for '{model_id}': run DB lookup failed "
            f"({type(e).__name__}); refusing to guess a config"
        )
    for row in lean:
        try:
            full = repo.get(str(row.id)) or row
            best = full.get_best_config()
        except Exception:
            continue
        if best is None:
            continue
        return best.config, best.context_length, str(full.id)
    raise NoBestConfigError(
        f"no stored best config for '{model_id}': run optimize first "
        "(refusing to guess a config)"
    )


def _default_deep_dir() -> Path:
    """Deep report dir under the configured results dir (test-patched)."""
    return Path(_app_config.storage.results_dir) / "deep"


def _task_entry(case: BenchmarkCase, metrics: BenchmarkMetrics, quality) -> dict:
    """Per-task metrics + thinking + scores (thinking best-effort)."""
    dm = deep_metrics(metrics)
    thinking = metrics.thinking_text or ""
    entry = {
        "name": case.name,
        "category": case.category,
        "status": "ok" if metrics.success else "failed",
        "error": metrics.error,
        "thinking_chars": len(thinking),
        "thinking": thinking,
        "recall_accuracy": (
            dm["recall_accuracy"] if case.name == "long_context_recall" else None
        ),
        "elapsed_s": dm["elapsed_s"],
        "gen_tok_s": dm["gen_tok_s"],
        "quality": (quality.overall if quality is not None else None),
        "output_chars": len(metrics.output_text or ""),
        "max_tokens": case.max_tokens,
    }
    if quality is not None:
        entry["score"] = quality.overall
    elif case.name == "long_context_recall":
        entry["score"] = dm["recall_accuracy"]
    else:
        entry["score"] = None
    return entry


def _skipped_entry(case: BenchmarkCase, reason: str) -> dict:
    return {
        "name": case.name,
        "category": case.category,
        "status": "skipped",
        "error": reason,
        "thinking_chars": 0,
        "thinking": "",
        "recall_accuracy": None,
        "elapsed_s": 0.0,
        "gen_tok_s": 0.0,
        "quality": None,
        "output_chars": 0,
        "max_tokens": case.max_tokens,
        "score": None,
    }


def _write_deep_report(
    model_id: str,
    load_config,
    context_length: int,
    best_run_id: str | None,
    tasks: list[dict],
    agg: ConfigurationResult,
    out_dir: Path,
    stamp: str,
    status: str,
    reason: str | None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sanitize_model_filename(model_id)}-deep-{stamp}.md"
    ok = sum(1 for t in tasks if t["status"] == "ok")
    lines = [
        f"# Deep benchmark: {model_id}",
        "",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- Status: {status}",
        f"- Tasks ok: {ok}/{len(tasks)}",
        f"- Best config source run: {best_run_id or 'explicit (CLI-provided)'}",
        f"- Context: {context_length}",
        f"- Load: {load_config.to_dict()}",
        f"- Per-task output cap: {MAX_OUTPUT_TOKENS_PER_TASK} tokens",
        f"- Consecutive-error abort cap: {MAX_CONSECUTIVE_ERRORS}",
        "- Quality: heuristic evaluator has no reference for deep tasks, "
        "so quality is N/A (never fabricated); recall_accuracy is the "
        "mechanically checked score for the recall task.",
    ]
    if reason:
        lines.append(f"- Note: {reason}")
    lines += [
        "",
        "## Per-task results",
        "",
        "| Task | Status | Recall | Gen tok/s | Elapsed s | Thinking chars | Score | Error |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in tasks:
        recall = f"{t['recall_accuracy']:.3f}" if t["recall_accuracy"] is not None else "N/A"
        score = f"{t['score']:.3f}" if t["score"] is not None else "N/A"
        err = (t["error"] or "")[:120]
        lines.append(
            f"| {t['name']} | {t['status']} | {recall} | {t['gen_tok_s']:.1f} | "
            f"{t['elapsed_s']:.1f} | {t['thinking_chars']} | {score} | {err} |"
        )
    lines += [
        "",
        "## All tried configurations (deep run)",
        "",
    ]
    lines += _all_configs_rows([agg], agg.id)
    lines += [
        "",
        "## Full outputs (all tasks, verbatim)",
        "",
    ]
    lines += _config_outputs(agg)
    lines += [""]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Deep report saved", model=model_id, path=str(path))
    return path


async def run_deep_model_async(
    client,
    model_id: str,
    load_config=None,
    *,
    out_dir: str | Path | None = None,
    context_length: int | None = None,
    service: BenchmarkService | None = None,
    stamp: str | None = None,
) -> dict:
    """Run the fixed deep suite on one model; returns per-task metrics/scores.

    ``load_config=None`` resolves the stored best via the run DB and raises
    NoBestConfigError when none exists. Single load/unload cycle per model:
    all 5 tasks run under one load (repetitions=1, no preheat chats), with a
    fail-closed unload in ``finally``. Task execution reuses
    BenchmarkService's channel load and single-case measurement; abort and
    failure counting are unchanged.
    """
    best_run_id: str | None = None
    if load_config is None:
        load_config, best_ctx, best_run_id = resolve_best_config(model_id)
        if context_length is None:
            context_length = best_ctx
    ctx = context_length or getattr(load_config, "context_length", None) or 4096
    service = service or BenchmarkService(client)
    evaluator = QualityEvaluator()

    cases = cap_deep_cases()
    collected: list[BenchmarkMetrics] = []
    tasks: list[dict] = []
    quality_by_test: dict = {}
    consecutive_errors = 0
    reason: str | None = None
    status = "completed"
    load_channel = "REST"
    load_verification: dict = {"verification": "UNKNOWN"}

    try:
        try:
            ok, _ident, _loaded, channel, applied, load_err = await service._channel_load(
                model_id, load_config
            )
        except Exception as e:
            ok, channel, applied, load_err = False, "REST", None, (
                f"{type(e).__name__}: {e}"
            )
        load_channel = channel
        from lm_optimizer.services import control_adapter as _ca

        load_verification = _ca.verify_applied(load_config.to_api_params(), applied)
        if not ok:
            reason = load_err or "model load failed"
            status = "failed"
            for case in cases:
                m = BenchmarkMetrics(
                    test_name=case.name,
                    category=case.category,
                    success=False,
                    error=reason,
                    prompt=case.prompt,
                )
                collected.append(m)
                tasks.append(_task_entry(case, m, None))
        else:
            for case in cases:
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    reason = (
                        f"aborted after {MAX_CONSECUTIVE_ERRORS} consecutive errors "
                        "(per-model budget cap); remaining tasks skipped"
                    )
                    tasks.append(_skipped_entry(case, reason))
                    collected.append(
                        BenchmarkMetrics(
                            test_name=case.name,
                            category=case.category,
                            success=False,
                            error=reason,
                            prompt=case.prompt,
                        )
                    )
                    continue
                m = await service._run_single_case(model_id, case, service.reasoning)
                collected.append(m)
                scored = evaluator.evaluate_all(model_id, [m])
                q = scored.get(case.name)
                quality = q if (q is not None and q.confident) else None
                if quality is not None:
                    quality_by_test[case.name] = quality
                tasks.append(_task_entry(case, m, quality))
                if m.success:
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        reason = (
                            f"aborted after {MAX_CONSECUTIVE_ERRORS} consecutive errors "
                            "(per-model budget cap); remaining tasks skipped"
                        )
            if any(t["status"] == "skipped" for t in tasks):
                status = "aborted"
    finally:
        try:
            if not await client.ensure_unloaded(model_id):
                logger.warning("Deep run left model loaded", model=model_id)
        except Exception:
            pass

    agg = ConfigurationResult(
        config=load_config,
        context_length=ctx,
        status=(
            ConfigurationStatus.PASSED
            if any(m.success for m in collected)
            else ConfigurationStatus.FAILED
        ),
        metrics=collected,
        quality_score=evaluator.aggregate_quality(quality_by_test)
        if quality_by_test
        else None,
        generation={
            "stage": "deep",
            "style": "deep",
            "best_run_id": best_run_id,
            "load_channel": load_channel,
            "load_verification": load_verification,
            "max_output_tokens_per_task": MAX_OUTPUT_TOKENS_PER_TASK,
            "max_consecutive_errors": MAX_CONSECUTIVE_ERRORS,
        },
    )
    target = Path(out_dir) if out_dir is not None else _default_deep_dir()
    report_path = _write_deep_report(
        model_id,
        load_config,
        ctx,
        best_run_id,
        tasks,
        agg,
        target,
        stamp or datetime.now().strftime("%Y%m%d-%H%M%S"),
        status,
        reason,
    )
    thinking_total = sum(t["thinking_chars"] for t in tasks)
    gen_speeds = [t["gen_tok_s"] for t in tasks if t["status"] == "ok" and t["gen_tok_s"] > 0]
    return {
        "model_id": model_id,
        "status": status,
        "reason": reason,
        "tasks": tasks,
        "report_path": str(report_path),
        "context_length": ctx,
        "best_run_id": best_run_id,
        "summary": {
            "tasks_ok": sum(1 for t in tasks if t["status"] == "ok"),
            "tasks_total": len(tasks),
            "thinking_chars_total": thinking_total,
            "gen_tok_s_avg": (sum(gen_speeds) / len(gen_speeds)) if gen_speeds else 0.0,
        },
    }


def run_deep_model(
    client,
    model_id: str,
    load_config=None,
    *,
    out_dir: str | Path | None = None,
    context_length: int | None = None,
    service: BenchmarkService | None = None,
    stamp: str | None = None,
) -> dict:
    """Sync wrapper around run_deep_model_async (tests + simple callers)."""
    return asyncio.run(
        run_deep_model_async(
            client,
            model_id,
            load_config,
            out_dir=out_dir,
            context_length=context_length,
            service=service,
            stamp=stamp,
        )
    )
