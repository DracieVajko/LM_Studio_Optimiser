"""API routes for the optimizer."""

import asyncio
import sqlite3
import statistics
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
    RestoreDefaultsRequest,
    ProfileWeightsSchema,
    SettingsSchema,
    StatsResponse,
    JudgeRequest,
    JudgeBackendSchema,
    JudgeVerdictResponse,
    JudgeHistoryItem,
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
from lm_optimizer.services.joblock import JobBusyError, acquire_lock, release_lock
from lm_optimizer.services.lm_studio import LMStudioClient, create_client
from lm_optimizer.storage.run_checkpoint import checkpoint_dir, load_checkpoint
from pathlib import Path
from datetime import datetime
import orjson
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


def _winner_metrics(r) -> dict:
    """Best-candidate metrics for a run (history lists must not load configs).

    Uses the run's architectural winner (final_recommendation) when configs are
    loaded, else the persisted ``best_config_id``, else the highest-score passed
    config. Missing => None (never a fake 0 that looks measured).
    """
    best = None
    configs = list(getattr(r, "configurations", []) or [])
    if configs:
        try:
            from lm_optimizer.services.run_summary import final_recommendation

            best = final_recommendation(configs, r)
        except Exception:
            best = None
    if best is None and getattr(r, "best_config_id", None):
        try:
            best = config_repo.get(str(r.best_config_id))
        except Exception:
            best = None
    if best is None:
        try:
            best = config_repo.get_best_for_run(str(r.id))
        except Exception:
            best = None
    if best is None:
        return {
            "best_gen_tok_s": None,
            "best_quality": None,
            "best_context_length": None,
            "has_validated_winner": False,
        }
    q = getattr(best, "quality_score", None)
    gen = None
    try:
        g = best.get_avg_generation_tok_s()
        gen = round(float(g), 3) if g else None
    except Exception:
        gen = None
    return {
        "best_gen_tok_s": gen,
        "best_quality": float(q.overall) if q is not None else None,
        "best_context_length": getattr(best, "context_length", None),
        "has_validated_winner": True,
    }


def _convert_run(r, config_count: int = 0, stale: bool = False) -> OptimizationRunResponse:
    pab = (r.benchmark_params or {}).get("phase_ab") or {}
    raw_id = pab.get("raw_fastest_id")
    try:
        raw_uuid = UUID(str(raw_id)) if raw_id else None
    except Exception:
        raw_uuid = None
    winner = _winner_metrics(r)
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
        best_gen_tok_s=winner["best_gen_tok_s"],
        best_quality=winner["best_quality"],
        best_context_length=winner["best_context_length"],
        has_validated_winner=winner["has_validated_winner"],
        artifact_test=(r.benchmark_params or {}).get("artifact_test"),
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


def _is_embedding_model(m) -> bool:
    """Embedding-only models cannot be text-optimized (excluded from Optimize)."""
    mid = (getattr(m, "id", "") or "").lower()
    mt = (getattr(m, "model_type", "") or "").lower()
    return "embed" in mid or mt in ("embeddings", "embedding")


async def _read_inventory() -> tuple[list, str | None]:
    """Live provider inventory plus an error string (never a silent empty).

    Distinguishes a genuinely empty inventory from a failed provider call so
    the dashboard can show an error state instead of a misleading 0.
    """
    try:
        client = await get_lm_client(echo_probe=False)
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"[:160]
    try:
        return list(await client.list_models()), None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"[:160]
    finally:
        try:
            await client.close()
        except Exception:
            pass


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Dashboard quick-stats with documented, tested meanings."""
    models, inv_err = await _read_inventory()
    total = len(models)
    embedding_only = sum(1 for m in models if _is_embedding_model(m))
    optimized = total_runs = runs_today = 0
    try:
        optimized = run_repo.count_models_with_validated_winner()
        total_runs = run_repo.count_all()
        runs_today = run_repo.count_created_on(datetime.now().strftime("%Y-%m-%d"))
    except Exception as e:
        logger.warning("Stats DB read failed", error=str(e)[:160])
    return StatsResponse(
        models_available=total - embedding_only,
        models_total=total,
        models_embedding_only=embedding_only,
        inventory_ok=inv_err is None,
        inventory_error=inv_err,
        optimized_models=optimized,
        runs_today=runs_today,
        total_runs=total_runs,
        as_of=datetime.now().isoformat(timespec="seconds"),
    )


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get model details."""
    try:
        client = await get_lm_client(echo_probe=False)
    except Exception:
        raise HTTPException(status_code=503, detail="LM Studio unreachable")
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

    # Cross-process job lock (shared with the CLI): one optimizer job per
    # machine. PID-liveness means a stale lock never blocks forever; two
    # workers racing resolve to exactly one winner (atomic create).
    try:
        acquire_lock(f"optimize:{request.model_id}")
    except JobBusyError as e:
        holder = e.holder or {}
        raise HTTPException(
            status_code=409,
            detail=(
                "Optimizer job already running "
                f"(pid {holder.get('pid')}, purpose {holder.get('purpose')!r})."
            ),
        )
    try:
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

        # Merge top-level workload into advanced settings (canonical source).
        from lm_optimizer.services.workload import normalize_workload

        adv = request.advanced_settings.dict() if request.advanced_settings else {}
        adv.setdefault("workload_type", normalize_workload(request.workload_type))
        adv["workload_type"] = normalize_workload(adv.get("workload_type"))
        # Top-level request fields are authoritative for the speed floor and
        # preset; advanced_settings defaults must not silently override them.
        adv["min_generation_tok_s"] = float(request.min_generation_tok_s)
        adv["enforce_min_generation_speed"] = bool(
            request.enforce_min_generation_speed
        )
        adv["run_preset"] = str(request.run_preset)

        # Granular modes: compose the skip set the same way the CLI does, so
        # the web endpoint honours --speed-only/--quality-only/... too.
        from lm_optimizer.services.optimizer import compose_granular_skips

        if any(
            adv.get(key)
            for key in ("speed_only", "quality_only", "reasoning_sweep", "mtp_test")
        ):
            adv["skip_phases"] = compose_granular_skips(adv)
        if adv.get("reasoning_sweep"):
            adv["include_categories"] = {"reasoning"}

        # Scheduled artifact test: validate before any run is created so an
        # unknown task fails fast instead of being silently disabled.
        if adv.get("artifact_test_id"):
            from lm_optimizer.services.optimizer import validate_artifact_test_id

            try:
                adv["artifact_test_id"] = validate_artifact_test_id(
                    adv["artifact_test_id"]
                )
            except ValueError as e:
                try:
                    await client.close()
                except Exception:
                    pass
                raise HTTPException(status_code=422, detail=str(e)[:300]) from None

        # Create services
        benchmark_service = BenchmarkService(
            client,
            speed_only=bool(adv.get("speed_only")),
            quality_only=bool(adv.get("quality_only")),
            speed_context=adv.get("speed_context"),
            mtp_test=bool(adv.get("mtp_test")),
            reasoning_sweep=bool(adv.get("reasoning_sweep")),
            quality_lab=bool(adv.get("quality_lab")),
        )
        if adv.get("warmup_runs") is not None:
            try:
                benchmark_service.config.warmup_repetitions = min(
                    5, max(0, int(adv["warmup_runs"]))
                )
            except (TypeError, ValueError):
                pass
        quality_evaluator = QualityEvaluator(QualityConfig(minimum_score=request.quality_threshold))
        search_generator = SearchSpaceGenerator(client)
        optimizer = AdaptiveOptimizer(
            client, benchmark_service, quality_evaluator, search_generator
        )
        weights = (
            ProfileWeights(**request.custom_weights.dict())
            if request.custom_weights
            else None
        )
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
    except Exception:
        # Start failed after the lock was claimed: never leak it.
        try:
            release_lock()
        except Exception:
            pass
        raise

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
        try:
            release_lock()
        except Exception:
            pass


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


@router.post("/runs/{run_id}/pause")
async def pause_run(run_id: UUID):
    """Pause any running optimization by run ID (creates pause flag for next safe boundary)."""
    run = run_repo.get(str(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    status_value = run.status.value if hasattr(run.status, 'value') else run.status
    if status_value not in ("running", "resumed"):
        raise HTTPException(status_code=409, detail=f"Run is {status_value}, not running")
    
    flag = Path(str(checkpoint_dir())) / f"pause_{run_id}.flag"
    try:
        payload = {
            "requested_at": datetime.now().isoformat(timespec="seconds"),
            "reason": "user_requested",
        }
        flag.write_bytes(orjson.dumps(payload))
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Cannot write pause flag: {e}")
    
    return {"success": True, "message": "Pause requested; will take effect at next safe boundary"}


@router.post("/runs/{run_id}/resume")
async def resume_run(run_id: UUID):
    """Resume a paused run from its checkpoint (no re-runs)."""
    run = run_repo.get(str(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    status_value = run.status.value if hasattr(run.status, 'value') else run.status
    # Finishable states: paused, partial_success, interrupted. Live runs
    # (running/resumed) and explicit cancels stay manual.
    if status_value not in ("paused", "partial_success", "interrupted"):
        raise HTTPException(status_code=409, detail=f"Run is {status_value}, not resumable")
    
    ckpt = load_checkpoint(run_id)
    if not ckpt:
        raise HTTPException(status_code=404, detail="No checkpoint for run")
    
    # Resume via the existing background task mechanism
    from fastapi import BackgroundTasks
    
    background_tasks = BackgroundTasks()
    result = await resume_run_from_checkpoint(run_id, background_tasks)
    
    return {"success": True, "run_id": str(run_id), "completed": len(ckpt.get("completed_candidate_ids", []))}


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
    try:
        acquire_lock(f"resume:{run_id}")
    except JobBusyError as e:
        holder = e.holder or {}
        raise HTTPException(
            status_code=409,
            detail=(
                "Optimizer job already running "
                f"(pid {holder.get('pid')}, purpose {holder.get('purpose')!r})."
            ),
        )
    from lm_optimizer.storage.run_checkpoint import load_checkpoint

    try:
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
        optimizer = AdaptiveOptimizer(
            client, benchmark_service, quality_evaluator, search_generator
        )
        _current_optimizer = optimizer
        background_tasks.add_task(resume_run_task, optimizer, run_id)
    except Exception:
        try:
            release_lock()
        except Exception:
            pass
        raise
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
        try:
            release_lock()
        except Exception:
            pass


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

    done_states = (
        "completed",
        "success",
        "partial_success",
        "completed_no_validated_winner",
        "failed",
        "cancelled",
        "interrupted",
    )
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


@router.get("/runs/{run_id}/export-config")
async def export_lmstudio_config(run_id: UUID):
    """Export the validated winner as a reusable load-config JSON."""
    from lm_optimizer.services.config_export import build_load_config_export

    run = run_repo.get(str(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        return build_load_config_export(run)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None


@router.get("/model-default")
async def get_model_default(model_id: str):
    """Inspect a model's current LM Studio defaults, fields and backups."""
    from lm_optimizer.services.lmstudio_defaults import (
        DefaultsError,
        ModelNotFoundError,
        list_backups,
        resolve_defaults_path,
    )

    try:
        path = resolve_defaults_path(model_id)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except Exception as e:  # noqa: BLE001 - surface resolution errors
        raise HTTPException(status_code=400, detail=str(e)) from None

    existed = path.exists()
    fields: dict = {}
    if existed:
        try:
            from lm_optimizer.services.lmstudio_defaults import (
                _normalize_backup_fields,
                _read_document,
            )

            document, _ = _read_document(path)
            fields = _normalize_backup_fields(document)
        except DefaultsError as e:
            raise HTTPException(status_code=409, detail=str(e)) from None

    backups = [entry.name for entry in list_backups(path)]
    return {
        "model_id": model_id,
        "path": str(path),
        "exists": existed,
        "fields": fields,
        "backups": backups,
    }


@router.post("/model-default/restore")
async def restore_model_default(request: RestoreDefaultsRequest):
    """Restore a model's LM Studio defaults from a backup."""
    from lm_optimizer.services.lmstudio_defaults import DefaultsError, restore_defaults

    try:
        result = restore_defaults(
            request.model_id, backup=request.backup, dry_run=request.dry_run
        )
    except DefaultsError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    return result.to_dict()


@router.get("/runs/{run_id}/default-diff")
async def get_run_default_diff(run_id: UUID):
    """Diff a run's validated winner against the model's current defaults."""
    from dataclasses import replace as _replace

    run = run_repo.get(str(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    best = run.get_best_config()
    if best is None:
        raise HTTPException(status_code=404, detail="No validated winner in this run")
    config = best.config
    if config.context_length is None and getattr(best, "context_length", None):
        config = _replace(config, context_length=best.context_length)
    model_id = getattr(getattr(run, "model", None), "id", "") or ""

    from lm_optimizer.services.lmstudio_defaults import DefaultsError, diff_defaults

    try:
        diff = diff_defaults(model_id, config)
    except DefaultsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from None
    return diff.to_dict()


@router.post("/runs/{run_id}/verify-default")
async def verify_run_default(run_id: UUID):
    """Verify a run's saved defaults via an echo load (never writes)."""
    from dataclasses import replace as _replace

    run = run_repo.get(str(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    best = run.get_best_config()
    if best is None:
        raise HTTPException(status_code=404, detail="No validated winner in this run")
    config = best.config
    if config.context_length is None and getattr(best, "context_length", None):
        config = _replace(config, context_length=best.context_length)
    model_id = getattr(getattr(run, "model", None), "id", "") or ""

    _client_result = get_lm_client(echo_probe=False)
    if hasattr(_client_result, "__await__"):
        client = await _client_result
    else:
        client = _client_result
    try:
        from lm_optimizer.services.lmstudio_default_verify import verify_defaults

        result = await verify_defaults(model_id, config, client=client)
    finally:
        try:
            if hasattr(client, "close"):
                await client.close()
        except Exception:  # noqa: BLE001
            pass
    return result


@router.post("/runs/{run_id}/save-default")
async def save_lmstudio_default(run_id: UUID, dry_run: bool = False, verify: bool = False):
    """Persist the validated winner as LM Studio's per-model load default."""
    from lm_optimizer.services.lmstudio_defaults import (
        DefaultsError,
        ModelNotFoundError,
        NoWinnerError,
        save_run_defaults,
    )

    run = run_repo.get(str(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        result = save_run_defaults(run, dry_run=dry_run)
    except (NoWinnerError, ModelNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except DefaultsError as e:
        # Unreadable/conflicting local file: never overwrite user data.
        raise HTTPException(status_code=409, detail=str(e)) from None

    payload = result.to_dict()

    if verify:
        if dry_run:
            payload["verification"] = {"skipped": "dry_run"}
        else:
            from dataclasses import replace as _replace

            best = run.get_best_config()
            config = best.config if best is not None else None
            if config is not None and config.context_length is None and getattr(
                best, "context_length", None
            ):
                config = _replace(config, context_length=best.context_length)
            model_id = getattr(getattr(run, "model", None), "id", "") or ""
            _client_result = get_lm_client(echo_probe=False)
            if hasattr(_client_result, "__await__"):
                client = await _client_result
            else:
                client = _client_result
            try:
                from lm_optimizer.services.lmstudio_default_verify import verify_defaults

                verification = await verify_defaults(model_id, config, client=client)
            finally:
                try:
                    if hasattr(client, "close"):
                        await client.close()
                except Exception:
                    pass
            payload["verification"] = verification

    return payload


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
    # Checkpoint is intentionally preserved: history + diagnostics + a possible
    # later resume must survive an abandon. Abandon is not deletion.
    return {"success": True, "id": str(run_id), "status": "interrupted"}


@router.post("/runs/reconcile")
async def reconcile_runs(apply: bool = False):
    """Preview/apply stale-running reconciliation (idempotent, non-destructive).

    Default preview only; pass ``?apply=true`` to mark stale runs interrupted.
    Keeps all candidate results, reports and checkpoints.
    """
    from lm_optimizer.services.run_reconcile import reconcile_stale_runs

    return reconcile_stale_runs(dry_run=not apply, current_run_id=_current_run_id)


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


# ==================== JUDGE ENDPOINTS ====================

# In-memory job tracking for async judge requests
_judge_jobs: dict[str, dict] = {}


async def _run_judge_async(judge_id: str, request: JudgeRequest):
    """Background task to run judge evaluation."""
    from lm_optimizer.services.judge_service import JudgeService
    from lm_optimizer.database.manager import DatabaseManager
    from lm_optimizer.config import config

    try:
        db = DatabaseManager(config.storage.database_path)
        svc = JudgeService(db_path=config.storage.database_path)
        verdict = await svc.run_judge(request)

        _judge_jobs[judge_id] = {
            "status": "completed",
            "verdict": verdict.to_dict(),
            "error_message": None,
        }
    except Exception as e:
        _judge_jobs[judge_id] = {
            "status": "failed",
            "verdict": None,
            "error_message": str(e),
        }


@router.post("/judge", response_model=JudgeVerdictResponse)
async def judge_submit(request: JudgeRequest, background: BackgroundTasks, wait: bool = False):
    """Submit a judge evaluation request."""
    import uuid

    # Validate request
    if not request.run_ids and not request.model_ids:
        raise HTTPException(status_code=400, detail="Either run_ids or model_ids must be provided")

    if request.backend == JudgeBackendSchema.API and not request.api_key:
        raise HTTPException(status_code=400, detail="API key required for API backend")

    judge_id = str(uuid.uuid4())
    _judge_jobs[judge_id] = {"status": "pending", "verdict": None, "error_message": None}

    if wait:
        # Run synchronously
        from lm_optimizer.services.judge_service import JudgeService
        from lm_optimizer.database.manager import DatabaseManager
        from lm_optimizer.config import config

        db = DatabaseManager(config.storage.database_path)
        svc = JudgeService(db_path=config.storage.database_path)
        verdict = await svc.run_judge(request)

        _judge_jobs[judge_id] = {
            "status": "completed",
            "verdict": verdict.to_dict(),
            "error_message": None,
        }
        return JudgeVerdictResponse(
            judge_id=judge_id,
            status="completed",
            verdict=verdict.to_dict(),
            error_message=None,
            poll_url=f"/api/judge/{judge_id}",
        )
    else:
        # Run in background - return 202 Accepted
        from fastapi import Response
        _judge_jobs[judge_id] = {"status": "pending", "verdict": None, "error_message": None}
        background.add_task(_run_judge_async, judge_id, request)
        return Response(
            content=JudgeVerdictResponse(
                judge_id=judge_id,
                status="pending",
                verdict=None,
                error_message=None,
                poll_url=f"/api/judge/{judge_id}",
            ).model_dump_json(),
            status_code=202,
            media_type="application/json",
        )


@router.get("/judge/{judge_id}", response_model=JudgeVerdictResponse)
async def judge_get(judge_id: str):
    """Get judge verdict by ID."""
    # First check in-memory jobs (for pending/running)
    job = _judge_jobs.get(judge_id)
    if job:
        return JudgeVerdictResponse(
            judge_id=judge_id,
            status=job["status"],
            verdict=job.get("verdict"),
            error_message=job.get("error_message"),
            poll_url=f"/api/judge/{judge_id}",
        )

    # Fallback to database (for completed verdicts)
    from lm_optimizer.database.manager import DatabaseManager
    from lm_optimizer.config import config
    import json

    db = DatabaseManager(config.storage.database_path)
    with db.get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, run_ids_json, config_ids_json, use_case, custom_prompt, priority, backend, judge_model, verdict_json, prompt_tokens, completion_tokens, latency_ms, status, error_message, created_at "
            "FROM judge_verdicts WHERE id = ?",
            (judge_id,)
        ).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Judge job not found")

        verdict = json.loads(row["verdict_json"]) if row["verdict_json"] else {}

        return JudgeVerdictResponse(
            judge_id=row["id"],
            status=row["status"],
            verdict=verdict,
            error_message=row["error_message"],
            poll_url=f"/api/judge/{judge_id}",
        )


@router.get("/judge/history", response_model=list[JudgeHistoryItem])
async def judge_history(limit: int = 20):
    """Get recent judge verdicts from database."""
    from lm_optimizer.database.manager import DatabaseManager
    from lm_optimizer.config import config
    import json

    db = DatabaseManager(config.storage.database_path)
    with db.get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, run_ids_json, use_case, backend, judge_model, verdict_json, created_at, status "
            "FROM judge_verdicts ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()

        results = []
        for row in rows:
            verdict = json.loads(row["verdict_json"]) if row["verdict_json"] else {}
            results.append(JudgeHistoryItem(
                id=row["id"],
                run_ids=json.loads(row["run_ids_json"]),
                use_case=row["use_case"],
                backend=row["backend"],
                judge_model=row["judge_model"],
                created_at=row["created_at"],
                status=row["status"],
                confidence=verdict.get("confidence"),
                overall_winner_model=verdict.get("overall_winner", {}).get("model_id"),
            ))
        return results


# --- Lab workflows (P2-P5) ------------------------------------------------


@router.get("/lab/recommend")
async def lab_recommend(
    min_tps: float = 5.0,
    min_context: int = 8192,
    vram_only: bool = False,
    use_case: str = "general",
):
    """Model recommendations for the detected hardware.

    Fit is decided by file size and bits/weight (never parameter count) and
    local measurements in the database outrank community priors.
    """
    from lm_optimizer.services.recommender import (
        Confidence,
        HardwareFingerprint,
        ModelCatalogEntry,
        ModelRecommender,
        RecommenderRequest,
        UseCase,
    )
    from lm_optimizer.services.hardware import hardware_detector

    try:
        hw_info = hardware_detector.detect()
    except Exception:
        hw_info = None

    gpu = None
    if hw_info is not None and getattr(hw_info, "gpus", None):
        gpu = max(hw_info.gpus, key=lambda g: getattr(g, "vram_gb", 0) or 0)

    hardware = HardwareFingerprint(
        gpu_name=(getattr(gpu, "name", "") or ""),
        vram_gb=float(getattr(gpu, "vram_gb", 0) or 0),
        gpu_bandwidth_gbps=float(getattr(gpu, "bandwidth_gbps", 0) or 0),
        gpu_tdp_w=float(getattr(gpu, "tdp_w", 0) or 0),
        cpu_name=(getattr(hw_info, "cpu_name", "") or ""),
        physical_cores=int(getattr(hw_info, "cpu_cores_physical", 0) or 0),
        logical_threads=int(getattr(hw_info, "cpu_cores_logical", 0) or 0),
        ram_gb=float(getattr(hw_info, "total_ram_gb", 0) or 0),
        ram_channels=int(getattr(hw_info, "ram_channels", 1) or 1),
        disk_type=(getattr(hw_info, "disk_type", "") or ""),
        os_name=(getattr(hw_info, "os", "") or ""),
    )

    catalog = _lab_catalog()
    recommender = ModelRecommender(catalog=catalog)
    for model_id, (tps, run_id, ctx) in _local_measurements().items():
        recommender.add_local_measurement(model_id, tps, run_ids=[run_id], context_length=ctx)

    try:
        use = UseCase(use_case)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown use case: {use_case}")

    request = RecommenderRequest(
        hardware=hardware,
        use_case=use,
        min_tps=min_tps,
        min_context=min_context,
        vram_only=vram_only,
    )
    results = recommender.recommend(request)
    return {
        "hardware": hardware.as_dict(),
        "recommendations": [r.as_dict() for r in results],
    }


@router.get("/lab/quality-tasks")
async def lab_quality_tasks():
    """The deterministic quality task library (P3)."""
    from lm_optimizer.services.quality_lab import default_tasks

    return {
        "tasks": [
            {
                "id": t.id,
                "category": t.category.value,
                "prompt": t.prompt,
                "validators": [v.kind.value for v in t.validators],
                "hard_gates": [g.kind.value for g in t.hard_gates],
                "max_tokens": t.max_tokens,
            }
            for t in default_tasks()
        ]
    }


@router.get("/lab/duel-types")
async def lab_duel_types():
    """Available duel types and their task sets (P4)."""
    from lm_optimizer.services.duel import DUEL_TASKS, DuelType

    return {
        "types": [
            {"id": t.value, "tasks": list(DUEL_TASKS[t])} for t in DuelType
        ]
    }


@router.get("/lab/sandbox-tasks")
async def lab_sandbox_tasks():
    """Sandbox task definitions with their acceptance criteria (P4)."""
    from lm_optimizer.services.sandbox_lab import SANDBOX_TASKS

    return {
        "tasks": [
            {
                "id": task.id,
                "kind": task.kind.value,
                "title": task.title,
                "prompt": task.prompt,
                "checks": [
                    {
                        "name": check.name,
                        "kind": check.kind,
                        "critical": check.critical,
                        "negate": check.negate,
                    }
                    for check in task.checks
                ],
                "max_repair_attempts": task.max_repair_attempts,
                "timeout_s": task.timeout_s,
                "network_allowed": task.network_allowed,
                "artifacts": list(task.artifacts),
            }
            for task in SANDBOX_TASKS.values()
        ]
    }


@router.post("/lab/sandbox-run")
async def lab_sandbox_run(payload: dict, background: BackgroundTasks):
    """Start a sandbox task in the background (P4).

    Body: {"task": "web_dashboard", "model": "publisher/model"}.
    Returns a job id; poll GET /lab/sandbox-run/{job_id}.
    """
    import uuid as _uuid

    from lm_optimizer.services.sandbox_lab import SANDBOX_TASKS

    task_id = str(payload.get("task") or "")
    model = str(payload.get("model") or "")
    if task_id not in SANDBOX_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown sandbox task: {task_id}")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    job_id = str(_uuid.uuid4())
    _sandbox_jobs[job_id] = {"status": "pending", "result": None, "error": None}
    background.add_task(_run_sandbox_job, job_id, task_id, model)
    return {"job_id": job_id, "status": "pending", "poll_url": f"/lab/sandbox-run/{job_id}"}


@router.get("/lab/sandbox-run/{job_id}")
async def lab_sandbox_status(job_id: str):
    """Poll a sandbox job."""
    job = _sandbox_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return {"job_id": job_id, **job}


async def _run_sandbox_job(job_id: str, task_id: str, model: str) -> None:
    """Background worker: run the sandbox task with the repair loop."""
    from datetime import datetime as _dt

    from lm_optimizer.services.sandbox_lab import SANDBOX_TASKS, SandboxPolicy
    from lm_optimizer.services.sandbox_runner import SandboxRunner, apply_retention

    definition = SANDBOX_TASKS[task_id]
    stamp = _dt.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("results") / "sandbox" / task_id
    workdir = base_dir / stamp
    policy = SandboxPolicy(
        workdir=workdir,
        network_enabled=False,
        timeout_s=definition.timeout_s,
        allowed_executables=definition.allowed_executables,
    )
    runner = SandboxRunner(task=definition, policy=policy, workdir=workdir)
    client = None
    try:
        _sandbox_jobs[job_id]["status"] = "running"
        client = await get_lm_client(echo_probe=False)

        async def _generate(feedback: str) -> str:
            prompt = definition.prompt
            if feedback:
                prompt = feedback + "\n\n" + definition.prompt
            response = await client.chat_completion(
                model=model,
                input_text=prompt,
                temperature=0.2,
                max_output_tokens=2048,
            )
            choices = response.get("choices") or []
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content", "") or ""

        result = await runner.run(_generate)
        apply_retention(base_dir, policy)
        _sandbox_jobs[job_id] = {
            "status": "completed",
            "result": result.as_dict(),
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        _sandbox_jobs[job_id] = {
            "status": "failed",
            "result": None,
            "error": f"{type(e).__name__}: {e}"[:300],
        }
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass


#: In-memory sandbox jobs (same lifetime semantics as the judge jobs).
_sandbox_jobs: dict[str, dict] = {}


def _local_measurements() -> dict[str, tuple[float, str, int | None]]:
    """Latest measured median tok/s per model (local evidence).

    The context the measurement was taken at is returned as well: a 9 tok/s
    result at 4096 tokens is NOT comparable with a 9 tok/s result at 16384.
    """
    import json
    import sqlite3

    from lm_optimizer.config import config

    db_path = config.storage.database_path
    if not db_path or not db_path.exists():
        return {}
    out: dict[str, tuple[float, str, int | None]] = {}
    try:
        con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
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
    except Exception as e:  # noqa: BLE001
        logger.debug("lab local measurement load failed", error=str(e))
    return out


def _lab_catalog() -> list:
    """Curated catalog for the recommender (mirrors the CLI catalogue)."""
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


# Import logger
import structlog

logger = structlog.get_logger(__name__)
