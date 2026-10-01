"""API routes for the optimizer."""

import asyncio
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from lm_optimizer.api.schemas import (
    AdvancedSettingsSchema,
    ApplyConfigRequest,
    CompareRequest,
    CompareResponse,
    SandboxRequest,
    SandboxResponse,
    ConfigurationResultResponse,
    GPUInfoSchema,
    HardwareInfoSchema,
    LoadConfigSchema,
    ModelIdentitySchema,
    OptimizationProfileSchema,
    OptimizationRequest,
    OptimizationRunResponse,
    PresetSchema,
    ProfileWeightsSchema,
    SettingsSchema,
)
from lm_optimizer.database.repositories import (
    config_repo,
    duel_repo,
    preset_repo,
    run_repo,
    settings_repo,
)
from lm_optimizer.domain.models import (
    ConfigurationResult,
    ConfigurationStatus,
    HardwareInfo,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    ProfileWeights,
    RunStatus,
)
from lm_optimizer.services.benchmark import BenchmarkService
from lm_optimizer.services.hardware import hardware_detector
from lm_optimizer.services.lm_studio import LMStudioClient, create_client
from lm_optimizer.services.optimizer import AdaptiveOptimizer
from lm_optimizer.services.quality import QualityConfig, QualityEvaluator
from lm_optimizer.services.search_space import SearchSpaceGenerator

router = APIRouter()

# Global optimizer instance for current run
_current_optimizer: AdaptiveOptimizer | None = None
_current_run_id: UUID | None = None

# In-memory A/B compare jobs (short-lived; polling via GET /compare/{id}).
_compare_jobs: dict[str, dict] = {}

# In-memory sandbox duels (files persist under results/sandbox/{job_id}/).
_sandbox_jobs: dict[str, dict] = {}


def get_lm_studio_url() -> str:
    """Get LM Studio URL from settings."""
    return settings_repo.get("lm_studio_url", "http://127.0.0.1:1234")


async def warm_capability_cache() -> bool:
    """One load-free connect at startup so later requests hit the caps cache.

    Never raises: a down LM Studio must not prevent server startup.
    """
    try:
        client = await get_lm_client(echo_probe=False)
        try:
            return True
        finally:
            try:
                await client.close()
            except Exception:
                pass
    except Exception as e:
        logger.warning("Startup capability warmup skipped", error=str(e)[:160])
        return False


async def get_lm_client(echo_probe: bool = True) -> LMStudioClient:
    """Get or create LM Studio client.

    Read-only endpoints (status/models/connect-test) must pass
    echo_probe=False: the echo phase loads a real model, which takes
    far longer than the UI timeout and disturbs loaded models.
    """
    client = await create_client(get_lm_studio_url(), echo_probe=echo_probe)
    return client


def _convert_model(m: ModelIdentity) -> ModelIdentitySchema:
    return ModelIdentitySchema(
        id=m.id,
        name=m.name,
        architecture=m.architecture,
        parameter_count=m.parameter_count,
        quantization=m.quantization,
        context_limit=m.context_limit,
        is_moe=m.is_moe,
        num_experts=m.num_experts,
        size_bytes=m.size_bytes,
    )


def _convert_hardware(h: HardwareInfo) -> HardwareInfoSchema:
    return HardwareInfoSchema(
        os=h.os,
        cpu_name=h.cpu_name,
        cpu_cores_physical=h.cpu_cores_physical,
        cpu_cores_logical=h.cpu_cores_logical,
        total_ram_gb=h.total_ram_gb,
        gpu_count=h.gpu_count,
        gpus=[
            GPUInfoSchema(
                index=g.index,
                name=g.name,
                vram_gb=g.vram_gb,
                vendor=g.vendor,
                driver_version=g.driver_version,
                compute_capability=g.compute_capability,
                shared_memory_gb=g.shared_memory_gb,
            )
            for g in h.gpus
        ],
        cuda_version=h.cuda_version,
        metal_available=h.metal_available,
        vulkan_available=h.vulkan_available,
    )


def _convert_config(c: LoadConfiguration) -> LoadConfigSchema:
    return LoadConfigSchema(
        context_length=c.context_length,
        flash_attention=c.flash_attention,
        offload_kv_cache_to_gpu=c.offload_kv_cache_to_gpu,
        eval_batch_size=c.eval_batch_size,
        physical_batch_size=c.physical_batch_size,
        parallel=c.parallel,
        context_checkpoints=c.context_checkpoints,
        reasoning_budget_message=c.reasoning_budget_message,
        num_experts=c.num_experts,
        speculative_draft_max_tokens=c.speculative_draft_max_tokens,
        speculative_draft_min_tokens=c.speculative_draft_min_tokens,
        speculative_draft_min_continue_probability=c.speculative_draft_min_continue_probability,
        speculative_draft_mtp=c.speculative_draft_mtp,
        speculative_draft_simple=c.speculative_draft_simple,
        speculative_draft_model=c.speculative_draft_model,
    )


STALE_NONTERMINAL = frozenset({"running", "resumed"})
STALE_AFTER_SECONDS = 2 * 3600


def _is_stale_run(run_id, status_value: str) -> bool:
    """A running/resumed run with no fresh checkpoint is dead (killed process)."""
    if str(status_value or "").lower() not in STALE_NONTERMINAL:
        return False
    try:
        from lm_optimizer.storage.run_checkpoint import checkpoint_path as _cp

        p = _cp(str(run_id))
        if not p.exists():
            return True
        import time as _time

        return (_time.time() - p.stat().st_mtime) > STALE_AFTER_SECONDS
    except Exception:
        return False


def _convert_run(r, config_count: int = 0, stale: bool = False) -> OptimizationRunResponse:
    pab = (r.benchmark_params or {}).get("phase_ab") or {}
    raw_id = pab.get("raw_fastest_id")
    try:
        raw_uuid = UUID(str(raw_id)) if raw_id else None
    except Exception:
        raw_uuid = None
    return OptimizationRunResponse(
        id=r.id,
        model=_convert_model(r.model),
        hardware=_convert_hardware(r.hardware),
        profile=OptimizationProfileSchema(r.profile.value),
        profile_weights=ProfileWeightsSchema(**r.profile_weights),
        quality_threshold=r.quality_threshold,
        status=r.status.value,
        stage=r.stage.value,
        search_space=r.search_space,
        created_at=r.created_at,
        started_at=r.started_at,
        completed_at=r.completed_at,
        duration_seconds=r.duration_seconds,
        error=r.error,
        baseline_config_id=r.baseline_config_id,
        baseline_metrics=r.baseline_metrics,
        best_config_id=r.best_config_id,
        pareto_config_ids=r.pareto_config_ids,
        is_experimental=r.is_experimental,
        experimental_reason=r.experimental_reason,
        benchmark_params=r.benchmark_params,
        phase=str(pab.get("phase") or getattr(r, "phase", "speed")),
        raw_fastest_config_id=raw_uuid,
        phase_b_enabled=bool(pab.get("phase_b_enabled", getattr(r, "phase_b_enabled", False))),
        phase_b_result=pab.get("phase_b_result", getattr(r, "phase_b_result", None)),
        pause=pab.get("pause"),
        config_count=int(config_count),
        stale=bool(stale),
    )


def _convert_config_result(c: ConfigurationResult) -> ConfigurationResultResponse:
    # Build quality dict with correctness terminology
    quality_dict = None
    if c.quality_score:
        quality_dict = {
            "overall": c.quality_score.overall,
            "task_completion": c.quality_score.task_completion,
            "factual_consistency": c.quality_score.factual_consistency,
            "format_compliance": c.quality_score.format_compliance,
            "coding_correctness": c.quality_score.coding_correctness,
            "no_truncation": c.quality_score.no_truncation,
            "no_malformed": c.quality_score.no_malformed,
            "confident": c.quality_score.confident,
            "details": c.quality_score.details,
            "checks_passed": c.quality_score.checks_passed,
            "checks_total": c.quality_score.checks_total,
            "checks_str": c.quality_score.as_checks_str()
            if hasattr(c.quality_score, "as_checks_str")
            else f"{c.quality_score.checks_passed}/{c.quality_score.checks_total} checks passed",
            "label": "Correctness / Quality Score (heuristic checks)",
        }
    return ConfigurationResultResponse(
        id=c.id,
        run_id=c.run_id,
        config=_convert_config(c.config),
        context_length=c.context_length,
        status=c.status.value,
        metrics=[
            {
                "test_name": m.test_name,
                "category": m.category,
                "success": m.success,
                "load_time_ms": m.load_time_ms,
                "estimated_ttft_ms": m.estimated_ttft_ms,
                "ttft_ms": m.estimated_ttft_ms,  # backward compat
                "prompt_tokens": m.prompt_tokens,
                "completion_tokens": m.completion_tokens,
                "total_tokens": m.total_tokens,
                "prompt_processing_ms": m.prompt_processing_ms,
                "generation_ms": m.generation_ms,
                "prompt_tok_s": m.prompt_tok_s,
                "generation_tok_s": m.generation_tok_s,
                "error": m.error,
                "output_text": m.output_text[:200] if m.output_text else "",
                "thinking_text": (m.thinking_text or "")[:2000],
                "prompt": (m.prompt or "")[:4000],
            }
            for m in c.metrics
        ],
        quality=quality_dict,
        stability_score=c.stability_score,
        peak_vram_gb=c.peak_vram_gb,
        peak_ram_gb=c.peak_ram_gb,
        error=c.error,
        score=c.score,
        score_breakdown=c.score_breakdown,
        generation=c.generation,
        score_weights=(c.generation or {}).get("score_weights"),
        tested_at=c.tested_at,
        duration_ms=c.duration_ms,
        avg_generation_tok_s=c.get_avg_generation_tok_s(),
        avg_prompt_tok_s=c.get_avg_prompt_tok_s(),
        avg_estimated_ttft_ms=c.get_avg_estimated_ttft_ms(),
        avg_ttft_ms=c.get_avg_ttft_ms(),
    )


def _convert_preset(p: dict) -> PresetSchema:
    return PresetSchema(
        id=UUID(p["id"]),
        model_id=p["model_id"],
        profile=OptimizationProfileSchema(p["profile"]),
        name=p["name"],
        config=LoadConfigSchema(**p["config"]),
        metrics=p["metrics"],
        quality=p["quality"],
        run_id=UUID(p["run_id"]) if p["run_id"] else None,
        created_at=datetime.fromisoformat(p["created_at"]),
        updated_at=datetime.fromisoformat(p["updated_at"]),
        optimizer_version=p["optimizer_version"],
    )


# Import datetime
from datetime import datetime

# ============================================================
# Hardware & LM Studio Status
# ============================================================


@router.get("/status")
async def get_status():
    """Get system status (degrades to disconnected when LM Studio is down)."""
    hardware = hardware_detector.detect()
    url = get_lm_studio_url()

    def _offline():
        return {
            "lm_studio": {"connected": False, "url": url, "loaded_model": None},
            "hardware": _convert_hardware(hardware),
        }

    try:
        client = await get_lm_client(echo_probe=False)
    except Exception as e:
        logger.warning("LM Studio unreachable for /status", error=str(e)[:160])
        return _offline()
    try:
        try:
            lm_connected = await client.health_check()
        except Exception:
            lm_connected = False

        # Get currently loaded model
        loaded_model = None
        if lm_connected:
            try:
                models = await client.list_models()
                loaded = [m for m in models if m.id in client._loaded_models]
                if loaded:
                    loaded_model = _convert_model(loaded[0])
            except Exception:
                pass

        return {
            "lm_studio": {
                "connected": bool(lm_connected),
                "url": url,
                "loaded_model": loaded_model,
            },
            "hardware": _convert_hardware(hardware),
        }
    finally:
        try:
            await client.close()
        except Exception:
            pass


@router.post("/connect")
async def connect_lm_studio(url: str):
    """Test connection to LM Studio."""
    try:
        client = await create_client(url, echo_probe=False)
        await client.close()
        # Update setting
        settings_repo.set("lm_studio_url", url)
        return {"success": True, "message": "Connected successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# Models
# ============================================================


@router.get("/models")
async def list_models():
    """List all available models from LM Studio (empty when unreachable)."""
    try:
        client = await get_lm_client(echo_probe=False)
    except Exception as e:
        logger.warning("LM Studio unreachable for /models", error=str(e)[:160])
        return {"models": []}
    try:
        try:
            models = await client.list_models()
        except Exception:
            return {"models": []}
        return {"models": [_convert_model(m) for m in models]}
    finally:
        try:
            await client.close()
        except Exception:
            pass


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get model details."""
    client = await get_lm_client(echo_probe=False)
    try:
        model = await client.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        return _convert_model(model)
    finally:
        await client.close()


# ============================================================
# Optimization
# ============================================================


@router.post("/optimize", response_model=OptimizationRunResponse)
async def start_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Start an optimization run."""
    global _current_optimizer, _current_run_id

    if (
        _current_optimizer
        and _current_optimizer.state
        and not _current_optimizer.state.run.completed_at
    ):
        raise HTTPException(status_code=409, detail="Optimization already running")

    # Validate model exists
    client = await get_lm_client()
    model = await client.get_model(request.model_id)
    if not model:
        await client.close()
        raise HTTPException(status_code=404, detail="Model not found")

    # Fail-closed start: never optimize onto a dirty host (two models
    # sharing the GPU silently invalidates speed measurements).
    try:
        from lm_optimizer.services.unload_guard import UnloadNotClean, assert_unloaded

        await assert_unloaded(client, purpose=f"optimize:{request.model_id}")
    except UnloadNotClean as e:
        try:
            await client.close()
        except Exception:
            pass
        raise HTTPException(status_code=409, detail=str(e)[:300])

    # Get hardware
    hardware = hardware_detector.detect()

    # Create services
    benchmark_service = BenchmarkService(client)
    quality_evaluator = QualityEvaluator(QualityConfig(minimum_score=request.quality_threshold))
    search_generator = SearchSpaceGenerator(client)
    optimizer = AdaptiveOptimizer(client, benchmark_service, quality_evaluator, search_generator)

    # Merge top-level workload into advanced settings (canonical source).
    from lm_optimizer.services.workload import normalize_workload

    adv = request.advanced_settings.dict() if request.advanced_settings else {}
    adv.setdefault("workload_type", normalize_workload(request.workload_type))
    adv["workload_type"] = normalize_workload(adv.get("workload_type"))
    weights = ProfileWeights(**request.custom_weights.dict()) if request.custom_weights else None
    profile = OptimizationProfile(request.profile.value)

    # Prepare run + state synchronously so the response carries a real run id
    # (state used to appear only inside the background task -> HTTP 500).
    try:
        run, baseline_config, adv = await optimizer.prepare_run(
            model, hardware, profile, weights, request.quality_threshold, adv,
        )
    except Exception as e:
        try:
            await client.close()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Could not start run: {str(e)[:200]}")

    _current_optimizer = optimizer
    _current_run_id = run.id

    # Run in background from the prepared state (never re-prepares).
    background_tasks.add_task(
        run_optimization_task,
        optimizer,
        model,
        profile,
        run,
        baseline_config,
        adv,
    )

    return _convert_run(run, config_count=0, stale=False)


async def run_optimization_task(
    optimizer: AdaptiveOptimizer,
    model: ModelIdentity,
    profile: OptimizationProfile,
    run,
    baseline_config,
    advanced_settings: dict,
):
    """Background task for optimization (shutdown-safe: checkpoint first).

    Continues from the synchronously prepared run (prepare_run); never
    re-prepares, so the returned run id stays the executed one.
    """
    run_id = run.id
    try:
        run = await optimizer.execute(model, profile, run, baseline_config, advanced_settings)
        run_id = run.id
        try:
            from lm_optimizer.api.websocket import broadcast_complete

            await broadcast_complete(run.id, {"status": run.status.value, "stage": run.stage.value})
        except Exception:
            pass
    except asyncio.CancelledError:
        try:
            if optimizer.state is not None:
                optimizer.checkpoint_now(reason="task-cancelled")
                run_id = optimizer.state.run.id
        except Exception:
            pass
    except Exception as e:
        logger.error("Optimization task failed", error=str(e))
        try:
            if optimizer.state is not None:
                optimizer.checkpoint_now(reason="task-failed")
        except Exception:
            pass
    finally:
        global _current_optimizer, _current_run_id
        _current_optimizer = None
        _current_run_id = None


@router.post("/optimize/{run_id}/pause")
async def pause_optimization(run_id: UUID):
    """Pause current optimization."""
    global _current_optimizer
    if (
        _current_optimizer
        and _current_optimizer.state
        and _current_optimizer.state.run.id == run_id
    ):
        _current_optimizer.pause()
        return {"success": True}
    raise HTTPException(status_code=404, detail="Run not found or not running")


@router.post("/optimize/{run_id}/resume")
async def resume_optimization(run_id: UUID):
    """Resume paused optimization."""
    global _current_optimizer
    if (
        _current_optimizer
        and _current_optimizer.state
        and _current_optimizer.state.run.id == run_id
    ):
        _current_optimizer.resume()
        return {"success": True}
    raise HTTPException(status_code=404, detail="Run not found or not running")


@router.post("/optimize/{run_id}/cancel")
async def cancel_optimization(run_id: UUID):
    """Cancel current optimization (checkpoint is written before ack)."""
    global _current_optimizer
    if (
        _current_optimizer
        and _current_optimizer.state
        and _current_optimizer.state.run.id == run_id
    ):
        _current_optimizer.checkpoint_now(reason="api-cancel")
        _current_optimizer.cancel()
        return {"success": True, "checkpoint": True}
    # Also allow cancelling a persisted RESUMED/RUNNING run record.
    run = run_repo.get(str(run_id))
    if run and run.status.value in ("running", "resumed", "paused"):
        run.status = RunStatus.CANCELLED
        from lm_optimizer.domain.models import OptimizationStage as _Stage

        run.stage = _Stage.CANCELLED
        run_repo.save(run)
        return {"success": True, "checkpoint": False}
    raise HTTPException(status_code=404, detail="Run not found or not running")


@router.post("/optimize/{run_id}/resume-run")
async def resume_run_from_checkpoint(run_id: UUID, background_tasks: BackgroundTasks):
    """Resume an interrupted/cancelled run from its checkpoint (no re-runs)."""
    global _current_optimizer
    if _current_optimizer and _current_optimizer.state:
        raise HTTPException(status_code=409, detail="Another optimization is running")
    from lm_optimizer.storage.run_checkpoint import load_checkpoint

    try:
        ckpt = load_checkpoint(run_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Corrupted checkpoint: {e}")
    if not ckpt:
        raise HTTPException(status_code=404, detail="No checkpoint for run")
    run = run_repo.get(str(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    client = await get_lm_client()
    hardware = hardware_detector.detect()
    benchmark_service = BenchmarkService(client)
    quality_evaluator = QualityEvaluator(QualityConfig(minimum_score=run.quality_threshold))
    search_generator = SearchSpaceGenerator(client)
    optimizer = AdaptiveOptimizer(client, benchmark_service, quality_evaluator, search_generator)
    _current_optimizer = optimizer
    background_tasks.add_task(resume_run_task, optimizer, run_id)
    completed = len(ckpt.get("completed_candidate_ids", []))
    return {"success": True, "run_id": str(run_id), "completed": completed}


async def resume_run_task(optimizer: AdaptiveOptimizer, run_id: UUID):
    """Background resume task."""
    global _current_optimizer
    try:
        await optimizer.resume_from_checkpoint(run_id)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error("Resume task failed", error=str(e))
    finally:
        _current_optimizer = None


@router.get("/checkpoints")
async def list_checkpoints():
    """List interrupt checkpoints."""
    from lm_optimizer.storage.run_checkpoint import list_checkpoints as _list

    return {"checkpoints": _list()}


@router.get("/optimize/{run_id}/progress")
async def get_optimization_progress(run_id: UUID):
    """Get live optimization progress (never fakes ETA)."""
    from datetime import datetime as _dt

    global _current_optimizer
    if (
        _current_optimizer
        and _current_optimizer.state
        and _current_optimizer.state.run.id == run_id
    ):
        from lm_optimizer.services.failure_states import is_terminal_failure

        from lm_optimizer.services.optimizer import _display_total

        state = _current_optimizer.state
        total = _display_total(state)
        tested = len(state.tested_configs)
        passed = sum(1 for c in state.tested_configs if c.status == ConfigurationStatus.PASSED)
        failed = sum(
            1 for c in state.tested_configs
            if is_terminal_failure(c.status)
            and c.status not in (ConfigurationStatus.OOM, ConfigurationStatus.TIMEOUT)
        )
        oom = sum(1 for c in state.tested_configs if c.status == ConfigurationStatus.OOM)
        timeouts = sum(
            1 for c in state.tested_configs if c.status == ConfigurationStatus.TIMEOUT
        )

        current_config = None
        current_metrics = None
        if state.current_config_id:
            for c in state.tested_configs:
                if c.id == state.current_config_id:
                    current_config = _convert_config(c.config)
                    if c.metrics:
                        m = c.metrics[0]
                        current_metrics = {
                            "generation_tok_s": m.generation_tok_s,
                            "prompt_tok_s": m.prompt_tok_s,
                            "ttft_ms": m.estimated_ttft_ms,
                            "vram_gb": c.peak_vram_gb,
                            "quality": c.quality_score.overall if c.quality_score else None,
                        }
                    break

        best_score = 0.0
        best_config = None
        current_speed = 0.0
        vram = None
        ram = None
        if state.best_config:
            best_score = state.best_config.score
            best_config = _convert_config(state.best_config.config)
            current_speed = round(state.best_config.get_avg_generation_tok_s(), 1)
            vram = state.best_config.peak_vram_gb
            ram = state.best_config.peak_ram_gb
        elapsed = (
            (_dt.now() - state.started_at).total_seconds()
            if getattr(state, "started_at", None)
            else 0.0
        )
        return {
            "run_id": run_id,
            "stage": state.run.stage.value,
            "model": state.run.model.id,
            "progress": min(100, int(tested / max(total, 1) * 100)) if total else 0,
            "configs_tested": tested,
            "configs_total": total,
            "configs_passed": passed,
            "configs_failed": failed,
            "configs_oom": oom,
            "configs_timeouts": timeouts,
            "current_config": current_config,
            "current_metrics": current_metrics,
            "best_score": best_score,
            "best_config": best_config,
            "current_speed": current_speed,
            "vram_gb": vram,
            "ram_gb": ram,
            "context_progress": sorted({c.context_length for c in state.tested_configs}),
            "elapsed_s": round(elapsed, 1),
            "remaining": "adaptive / unknown",
            "events": list(getattr(state, "event_log", []))[-10:],
        }

    # Return saved progress (distinguish terminal states explicitly).
    run = run_repo.get(str(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    done_states = ("completed", "success", "partial_success", "failed", "cancelled", "interrupted")
    return {
        "run_id": run_id,
        "stage": run.stage.value,
        "status": run.status.value,
        "model": run.model.id,
        "progress": 100 if run.status.value in done_states else 0,
        "configs_tested": len(run.configurations),
        "configs_total": len(run.configurations),
        "configs_passed": sum(
            1 for c in run.configurations if c.status == ConfigurationStatus.PASSED
        ),
        "configs_failed": sum(
            1 for c in run.configurations
            if is_terminal_failure(c.status)
            and c.status not in (ConfigurationStatus.OOM, ConfigurationStatus.TIMEOUT)
        ),
        "configs_oom": sum(1 for c in run.configurations if c.status == ConfigurationStatus.OOM),
        "configs_timeouts": sum(
            1 for c in run.configurations if c.status == ConfigurationStatus.TIMEOUT
        ),
        "remaining": "adaptive / unknown",
        "events": [],
    }


# ============================================================
# Results
# ============================================================


def _safe_convert_run(r):
    """Convert one run; unreadable rows are skipped (never kill the list)."""
    try:
        return _convert_run(r)
    except Exception as e:
        logger.warning("Skipping unreadable run row", error=str(e)[:160])
        return None


def _safe_convert_config(c):
    """Convert one config; unreadable rows are skipped (never kill the list)."""
    try:
        return _convert_config_result(c)
    except Exception as e:
        logger.warning("Skipping unreadable config row", error=str(e)[:160])
        return None


def _enrich_config_detail(conv: dict, config, model_id: str) -> dict:
    """Single-config detail: full outputs + recomputed per-test quality.

    Heuristic quality is deterministic, so per-test scores are recomputed
    from stored outputs (only aggregates are persisted). Never raises:
    on any failure the truncated list view is kept.
    """
    try:
        from lm_optimizer.services.quality import QualityEvaluator

        full = {m.test_name: m.output_text or "" for m in (config.metrics or [])}
        ev = QualityEvaluator()
        for m in conv.get("metrics", []):
            try:
                name = m.get("test_name", "")
                if name in full:
                    m["output_text"] = full[name]
                    if full[name]:
                        q = ev.evaluate(model_id, name, full[name])
                        m["quality_overall"] = round(q.overall, 3)
                        try:
                            if q.details.get("thinking_outside_json"):
                                m["thinking_outside_json"] = True
                        except Exception:
                            pass
                src = next((x for x in (config.metrics or []) if x.test_name == name), None)
                if src is not None:
                    if getattr(src, "thinking_text", ""):
                        m["thinking_text"] = src.thinking_text
                    if getattr(src, "prompt", ""):
                        m["prompt"] = src.prompt
            except Exception:
                continue
    except Exception as e:
        logger.warning("Detail enrich failed", error=str(e)[:160])
    return conv


@router.get("/runs")
async def list_runs(limit: int = 50, offset: int = 0, model_id: str | None = None):
    """List optimization runs (corrupt rows skipped, counted)."""
    if model_id:
        runs = run_repo.get_by_model(model_id, limit)
    else:
        runs = run_repo.list_all(limit, offset)
    kept, skipped = [], 0
    for r in runs:
        try:
            n = run_repo.count_configurations(str(r.id))
        except Exception:
            n = 0
        stale = _is_stale_run(getattr(r, "id", None), getattr(getattr(r, "status", None), "value", ""))
        c = _safe_convert_run(r)
        if c is None:
            skipped += 1
        else:
            c.config_count = n
            c.stale = stale
            kept.append(c)
    return {"runs": kept, "skipped": skipped}


@router.get("/runs/{run_id}", response_model=OptimizationRunResponse)
async def get_run(run_id: UUID):
    """Get optimization run details."""
    run = run_repo.get(str(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        n = run_repo.count_configurations(str(run_id))
    except Exception:
        n = len(run.configurations or [])
    stale = _is_stale_run(str(run_id), getattr(getattr(run, "status", None), "value", ""))
    return _convert_run(run, config_count=n, stale=stale)


@router.get("/runs/{run_id}/configurations")
async def list_configurations(run_id: UUID):
    """List all configurations for a run (corrupt rows skipped, counted)."""
    run = run_repo.get(str(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    kept, skipped = [], 0
    for c in run.configurations:
        conv = _safe_convert_config(c)
        if conv is None:
            skipped += 1
        else:
            kept.append(conv)
    return {"configurations": kept, "skipped": skipped}


@router.get(
    "/runs/{run_id}/configurations/{config_id}", response_model=ConfigurationResultResponse
)
async def get_configuration(run_id: UUID, config_id: UUID):
    """Get configuration result details (full outputs, per-test quality)."""
    config = config_repo.get(str(config_id))
    if not config or str(config.run_id) != str(run_id):
        raise HTTPException(status_code=404, detail="Configuration not found")
    conv = _convert_config_result(config)
    model_id = ""
    try:
        run = run_repo.get(str(run_id))
        if run is not None and getattr(run, "model", None) is not None:
            model_id = run.model.id or ""
    except Exception:
        pass
    return _enrich_config_detail(conv.model_dump(), config, model_id)


@router.get("/runs/{run_id}/pareto")
async def get_pareto_frontier(run_id: UUID):
    """Get Pareto frontier configurations."""
    run = run_repo.get(str(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    pareto = run.get_pareto_configs()
    kept = [c for c in (_safe_convert_config(x) for x in pareto) if c is not None]
    return {"configurations": kept}


# ============================================================
# Apply Configuration
# ============================================================


@router.post("/apply")
async def apply_configuration(request: ApplyConfigRequest):
    """Apply a configuration to LM Studio (one-shot load, not a saved default)."""
    client = await get_lm_client(echo_probe=False)
    try:
        try:
            model = await client.get_model(request.model_id)
        except Exception:
            model = None
        if not model:
            raise HTTPException(
                status_code=404, detail=f"Model '{request.model_id}' not found in LM Studio"
            )
        result = await client.load_model(
            request.model_id, LoadConfiguration(**request.config.dict())
        )
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error or "Failed to load model")
        return {"success": True, "identifier": result.identifier}
    finally:
        try:
            await client.close()
        except Exception:
            pass


@router.post("/restore/{model_id}")
async def restore_previous(model_id: str):
    """Restore previous model configuration."""
    # This would restore the baseline configuration
    # For now, just unload the model
    client = await get_lm_client()
    try:
        await client.unload_model(model_id=model_id)
        return {"success": True, "message": "Model unloaded"}
    finally:
        await client.close()


# ============================================================
# Presets
# ============================================================


@router.get("/presets")
async def list_presets(model_id: str | None = None):
    """List saved presets."""
    if model_id:
        presets = preset_repo.list_by_model(model_id)
    else:
        presets = preset_repo.list_all()
    return {"presets": [_convert_preset(p) for p in presets]}


@router.post("/presets")
async def save_preset(preset: PresetSchema):
    """Save a preset."""
    preset_data = preset.dict()
    preset_data["id"] = str(preset.id)
    preset_id = preset_repo.save(preset_data)
    return {"id": preset_id, "success": True}


@router.delete("/presets/{preset_id}")
async def delete_preset(preset_id: UUID):
    """Delete a preset."""
    success = preset_repo.delete(str(preset_id))
    if not success:
        raise HTTPException(status_code=404, detail="Preset not found")
    return {"success": True}


@router.post("/presets/{preset_id}/apply")
async def apply_preset(preset_id: UUID):
    """Apply a saved preset."""
    preset = preset_repo.get(str(preset_id))
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")

    client = await get_lm_client()
    try:
        result = await client.load_model(preset["model_id"], LoadConfiguration(**preset["config"]))
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error or "Failed to load model")
        return {"success": True, "identifier": result.identifier}
    finally:
        await client.close()


# ============================================================
# Settings
# ============================================================


@router.get("/settings", response_model=SettingsSchema)
async def get_settings():
    """Get all settings."""
    settings = settings_repo.get_all()
    return SettingsSchema(**{k: v["value"] for k, v in settings.items()})


@router.put("/settings")
async def update_settings(settings: SettingsSchema):
    """Update settings."""
    for key, value in settings.dict().items():
        settings_repo.set(key, str(value))
    return {"success": True}


# ============================================================
# Export/Import
# ============================================================


@router.get("/runs/{run_id}/export")
async def export_run(run_id: UUID, format: str = "json"):
    """Export run results."""
    run = run_repo.get(str(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    data = _convert_run(run).dict()
    data["configurations"] = [_convert_config_result(c).dict() for c in run.configurations]

    if format == "json":
        return JSONResponse(
            content=data, headers={"Content-Disposition": f"attachment; filename=run_{run_id}.json"}
        )
    if format == "markdown":
        # Generate markdown report
        md = generate_markdown_report(run)
        return JSONResponse(
            content=md,
            headers={"Content-Disposition": f"attachment; filename=run_{run_id}.md"},
            media_type="text/markdown",
        )

    raise HTTPException(status_code=400, detail="Unsupported format")


def generate_markdown_report(run) -> str:
    """Generate markdown report for a run."""
    lines = [
        "# Optimization Run Report",
        "",
        f"**Run ID:** {run.id}",
        f"**Model:** {run.model.name} ({run.model.id})",
        f"**Profile:** {run.profile.value}",
        f"**Status:** {run.status.value}",
        f"**Created:** {run.created_at}",
        f"**Duration:** {run.duration_seconds:.1f}s",
        "",
        "## Hardware",
        f"- **OS:** {run.hardware.os}",
        f"- **CPU:** {run.hardware.cpu_name}",
        f"- **RAM:** {run.hardware.total_ram_gb:.1f} GB",
        f"- **GPU:** {', '.join(g.name for g in run.hardware.gpus)}",
        "",
        "## Best Configuration",
    ]

    if run.best_config_id:
        best = next((c for c in run.configurations if c.id == run.best_config_id), None)
        if best:
            bc = best.config
            lines.extend(
                [
                    f"- **Context:** {best.context_length}",
                    f"- **Flash Attention:** {bc.flash_attention}",
                    f"- **KV Cache:** {'GPU' if bc.offload_kv_cache_to_gpu else 'CPU'}",
                    f"- **Batch Size:** {bc.eval_batch_size}",
                    f"- **Physical Batch:** {bc.physical_batch_size}",
                    f"- **Parallel:** {bc.parallel}",
                    f"- **Checkpoints:** {bc.context_checkpoints}",
                    f"- **Generation:** {best.get_avg_generation_tok_s():.1f} tok/s",
                    f"- **Prompt:** {best.get_avg_prompt_tok_s():.0f} tok/s",
                    f"- **TTFT:** {best.get_avg_ttft_ms():.0f} ms",
                    f"- **VRAM:** {best.peak_vram_gb:.1f} GB"
                    if best.peak_vram_gb
                    else "- **VRAM:** N/A",
                    f"- **Quality:** {best.quality_score.overall:.3f}"
                    if best.quality_score
                    else "- **Quality:** N/A",
                ]
            )

    lines.append("")
    lines.append("## All Configurations")
    for c in run.configurations:
        q = f"{c.quality_score.overall:.3f}" if c.quality_score else "N/A"
        flag = "PASS" if c.status == ConfigurationStatus.PASSED else "FAIL"
        lines.append(
            f"- {c.context_length} ctx, KV "
            f"{'GPU' if c.config.offload_kv_cache_to_gpu else 'CPU'}, "
            f"{c.get_avg_generation_tok_s():.1f} tok/s, Q={q} {flag}"
        )

    return "\n".join(lines)


@router.post("/compare", response_model=CompareResponse)
async def start_compare(request: CompareRequest, background_tasks: BackgroundTasks):
    """Start an A/B comparison of two configs on the SAME tests."""
    import uuid

    from lm_optimizer.services.compare import parse_compare_cases

    if request.tests is not None:
        try:
            parse_compare_cases({"tests": [t.model_dump() for t in request.tests]})
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    compare_id = str(uuid.uuid4())
    _compare_jobs[compare_id] = {"status": "running", "verdict": None, "result": None, "error": None}
    background_tasks.add_task(_run_compare_task, compare_id, request)
    return CompareResponse(compare_id=compare_id, status="running")


@router.get("/compare/{compare_id}", response_model=CompareResponse)
async def get_compare(compare_id: str):
    """Poll an A/B comparison result."""
    job = _compare_jobs.get(compare_id)
    if not job:
        raise HTTPException(status_code=404, detail="Compare job not found")
    return CompareResponse(compare_id=compare_id, **job)


async def _run_compare_task(compare_id: str, request: CompareRequest) -> None:
    """Background A/B task: same tests on A then B, partial-safe."""
    try:
        from lm_optimizer.services.compare import CompareCase, run_ab_compare

        client = await get_lm_client()
        try:
            cfg_a = LoadConfiguration(**request.config_a.model_dump(exclude_none=True))
            cfg_b = LoadConfiguration(**request.config_b.model_dump(exclude_none=True))
            cases = (
                [CompareCase(**t.model_dump()) for t in request.tests]
                if request.tests
                else None
            )
            res = await run_ab_compare(
                client,
                request.model_id,
                cfg_a,
                cfg_b,
                cases,
                repetitions=request.repetitions,
                context_length=request.context_length,
            )
            _compare_jobs[compare_id] = {
                "status": "done",
                "verdict": res.verdict,
                "result": res.to_dict(),
                "error": None,
            }
        finally:
            try:
                await client.close()
            except Exception:
                pass
    except Exception as e:
        _compare_jobs[compare_id] = {
            "status": "failed",
            "verdict": None,
            "result": None,
            "error": f"{type(e).__name__}: {e}"[:300],
        }


@router.post("/sandbox", response_model=SandboxResponse)
async def start_sandbox(request: SandboxRequest, background_tasks: BackgroundTasks):
    """Start a model-vs-model duel (text / html / scene). Sequential A then B."""
    import uuid

    try:
        client = await get_lm_client(echo_probe=False)
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"LM Studio unreachable: {str(e)[:160]}"
        )
    try:
        for mid in (request.model_a, request.model_b):
            try:
                found = await client.get_model(mid)
            except Exception:
                found = None
            if not found:
                raise HTTPException(status_code=404, detail=f"Model '{mid}' not found")
    finally:
        try:
            await client.close()
        except Exception:
            pass

    job_id = uuid.uuid4().hex[:12]
    _sandbox_jobs[job_id] = {"status": "running", "kind": request.kind.value, "result": None, "error": None}
    try:
        duel_repo.save({
            "id": job_id, "kind": request.kind.value, "model_a": request.model_a,
            "model_b": request.model_b, "prompt": request.prompt, "status": "running",
            "faster": None, "result_json": None, "error": None, "completed_at": None,
        })
    except Exception as e:
        logger.warning("Duel history save failed (job still runs)", error=str(e)[:160])
    background_tasks.add_task(_run_sandbox_task, job_id, request)
    return SandboxResponse(job_id=job_id, status="running", kind=request.kind.value)


@router.get("/generation-profiles")
async def generation_profiles(model_id: str, architecture: str | None = None):
    """Measured benchmark settings + publisher recommendations per purpose."""
    from lm_optimizer.services.generation_profiles import profiles_for

    return {"model_id": model_id, "profiles": profiles_for(model_id, architecture)}


@router.get("/sandbox/{job_id}", response_model=SandboxResponse)
async def get_sandbox(job_id: str):
    """Poll a sandbox duel result (in-memory first, history fallback)."""
    import json as _json

    job = _sandbox_jobs.get(job_id)
    if job:
        return SandboxResponse(job_id=job_id, **job)
    try:
        row = duel_repo.get(job_id)
    except Exception:
        row = None
    if not row:
        raise HTTPException(status_code=404, detail="Sandbox job not found")
    result = None
    if row.get("result_json"):
        try:
            result = _json.loads(row["result_json"])
        except Exception:
            result = None
    return SandboxResponse(
        job_id=job_id, status=row.get("status", "done"),
        kind=row.get("kind", "text"), result=result, error=row.get("error"),
    )


@router.get("/duels")
async def list_duels(limit: int = 50, offset: int = 0):
    """Duel history (summaries; full result via single get)."""
    import time as _time

    def _stale(r: dict) -> bool:
        if str(r.get("status", "")).lower() not in ("running",):
            return False
        try:
            from datetime import datetime as _dt

            created = _dt.fromisoformat(str(r.get("created_at", "")).replace("Z", ""))
            return (_time.time() - created.timestamp()) > 6 * 3600
        except Exception:
            return False

    rows = duel_repo.list_all(limit, offset)
    out = []
    for r in rows:
        d = {k: r[k] for k in (
            "id", "kind", "model_a", "model_b", "status", "faster",
            "error", "created_at", "completed_at",
        ) if k in r}
        d["prompt"] = (r.get("prompt") or "")[:300]
        d["stale"] = _stale(r)
        out.append(d)
    return {"duels": out}


@router.get("/duels/{job_id}")
async def get_duel(job_id: str):
    """Full duel record including result."""
    try:
        row = duel_repo.get(job_id)
    except Exception:
        row = None
    if not row:
        raise HTTPException(status_code=404, detail="Duel not found")
    return row


def _duel_files_dir(job_id: str):
    """Sandbox file root (module attr: monkeypatchable in tests)."""
    from lm_optimizer.services import sandbox as _sb

    return _sb.DEFAULT_SANDBOX_ROOT / job_id


@router.delete("/duels/{job_id}")
async def delete_duel(job_id: str):
    """Delete a duel record plus its generated files."""
    import shutil

    try:
        shutil.rmtree(_duel_files_dir(job_id), ignore_errors=True)
    except Exception:
        pass
    try:
        gone = duel_repo.delete(job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)[:160]}")
    _sandbox_jobs.pop(job_id, None)
    if not gone:
        raise HTTPException(status_code=404, detail="Duel not found")
    return {"success": True, "id": job_id}


@router.delete("/runs/{run_id}")
async def delete_run(run_id: UUID):
    """Delete a run (refuses live or fresh non-terminal runs)."""
    try:
        run = run_repo.get(str(run_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lookup failed: {str(e)[:160]}")
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    status_value = getattr(getattr(run, "status", None), "value", "")
    if str(run_id) == str(_current_run_id) or (
        str(status_value).lower() in ("running", "resumed")
        and not _is_stale_run(str(run_id), status_value)
    ):
        raise HTTPException(status_code=409, detail="Run is active; pause/cancel first")
    try:
        ok = run_repo.delete(str(run_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)[:160]}")
    if not ok:
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        from lm_optimizer.storage.run_checkpoint import remove_checkpoint as _rm

        _rm(str(run_id))
    except Exception:
        pass
    return {"success": True, "id": str(run_id)}


@router.post("/runs/{run_id}/abandon")
async def abandon_run(run_id: UUID, force: bool = False):
    """Mark a dead non-terminal run INTERRUPTED (keeps it in history).

    Refuses live runs (fresh checkpoint or current) and terminal states
    unless force=1. Stale running/resumed processes are the main target.
    """
    from datetime import datetime

    try:
        run = run_repo.get(str(run_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lookup failed: {str(e)[:160]}")
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    status_value = str(getattr(getattr(run, "status", None), "value", "")).lower()
    terminal = status_value not in ("running", "resumed", "paused", "pending")
    if terminal and not force:
        raise HTTPException(status_code=409, detail=f"Run is already {status_value}")
    if not force and (
        str(run_id) == str(_current_run_id)
        or (
            status_value in ("running", "resumed")
            and not _is_stale_run(str(run_id), status_value)
        )
    ):
        raise HTTPException(status_code=409, detail="Run looks alive; pause/cancel first")
    try:
        run.status = RunStatus.INTERRUPTED
        run.completed_at = datetime.now()
        run.error = ((run.error + "; " if getattr(run, "error", None) else "")
                     + "abandoned: process gone, no fresh checkpoint")
        run_repo.save(run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Abandon failed: {str(e)[:160]}")
    try:
        from lm_optimizer.storage.run_checkpoint import remove_checkpoint as _rm

        _rm(str(run_id))
    except Exception:
        pass
    return {"success": True, "id": str(run_id), "status": "interrupted"}


def _persist_duel(job_id: str, request: SandboxRequest, status: str,
                  result_json: str | None, error: str | None,
                  faster: str | None = None) -> None:
    """Best-effort duel history write (never breaks the duel itself)."""
    try:
        from datetime import datetime

        duel_repo.save({
            "id": job_id, "kind": request.kind.value, "model_a": request.model_a,
            "model_b": request.model_b, "prompt": request.prompt, "status": status,
            "faster": faster, "result_json": result_json, "error": error,
            "completed_at": datetime.now().isoformat() if status != "running" else None,
        })
    except Exception as e:
        logger.warning("Duel history update failed", error=str(e)[:160])


async def _run_sandbox_task(job_id: str, request: SandboxRequest) -> None:
    """Background duel: A, unload, B. Files land in results/sandbox/{job}/."""
    import json as _json

    try:
        from lm_optimizer.services.sandbox import run_duel

        client = await get_lm_client(echo_probe=False)
        try:
            res = await run_duel(
                client,
                request.model_a,
                request.model_b,
                request.prompt,
                kind=request.kind.value,
                timeout_s=request.timeout_s,
                job_id=job_id,
            )
            _sandbox_jobs[job_id] = {
                "status": "done",
                "kind": request.kind.value,
                "result": res.to_dict(),
                "error": None,
            }
            _persist_duel(job_id, request, "done", _json.dumps(res.to_dict()), None,
                          faster=res.faster)
        finally:
            try:
                await client.close()
            except Exception:
                pass
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:300]
        _sandbox_jobs[job_id] = {
            "status": "failed",
            "kind": request.kind.value,
            "result": None,
            "error": err,
        }
        _persist_duel(job_id, request, "failed", None, err)


# Import logger
import structlog

logger = structlog.get_logger(__name__)
