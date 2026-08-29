"""API routes for the optimizer."""

import asyncio
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from lm_optimizer.api.schemas import (
    AdvancedSettingsSchema,
    ApplyConfigRequest,
    ConfigurationResultResponse,
    HardwareInfoSchema,
    LoadConfigSchema,
    ModelIdentitySchema,
    OptimizationProfileSchema,
    OptimizationRequest,
    OptimizationRunResponse,
    PresetSchema,
    RunProgressResponse,
    SettingsSchema,
)
from lm_optimizer.database.repositories import (
    config_repo,
    preset_repo,
    run_repo,
    settings_repo,
)
from lm_optimizer.domain.models import (
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


def get_lm_studio_url() -> str:
    """Get LM Studio URL from settings."""
    return settings_repo.get("lm_studio_url", "http://127.0.0.1:1234")


async def get_lm_client() -> LMStudioClient:
    """Get or create LM Studio client."""
    client = await create_client(get_lm_studio_url())
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
        gpu_ratio=c.gpu_ratio,
        flash_attention=c.flash_attention,
        offload_kv_cache_to_gpu=c.offload_kv_cache_to_gpu,
        eval_batch_size=c.eval_batch_size,
        num_experts=c.num_experts,
        rope_freq_base=c.rope_freq_base,
        rope_freq_scale=c.rope_freq_scale,
    )


def _convert_run(r) -> OptimizationRunResponse:
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


@router.get("/api/status")
async def get_status():
    """Get system status."""
    hardware = hardware_detector.detect()
    client = await get_lm_client()
    lm_connected = await client.health_check()
    await client.close()

    # Get currently loaded model
    loaded_model = None
    if lm_connected:
        client = await get_lm_client()
        models = await client.list_models()
        loaded = [m for m in models if m.id in client._loaded_models]
        if loaded:
            loaded_model = _convert_model(loaded[0])
        await client.close()

    return {
        "lm_studio": {
            "connected": lm_connected,
            "url": get_lm_studio_url(),
            "loaded_model": loaded_model,
        },
        "hardware": _convert_hardware(hardware),
    }


@router.post("/api/connect")
async def connect_lm_studio(url: str):
    """Test connection to LM Studio."""
    try:
        client = await create_client(url)
        await client.close()
        # Update setting
        settings_repo.set("lm_studio_url", url)
        return {"success": True, "message": "Connected successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# Models
# ============================================================


@router.get("/api/models")
async def list_models():
    """List all available models from LM Studio."""
    client = await get_lm_client()
    try:
        models = await client.list_models()
        return {"models": [_convert_model(m) for m in models]}
    finally:
        await client.close()


@router.get("/api/models/{model_id}")
async def get_model(model_id: str):
    """Get model details."""
    client = await get_lm_client()
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


@router.post("/api/optimize", response_model=OptimizationRunResponse)
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

    # Get hardware
    hardware = hardware_detector.detect()

    # Create services
    benchmark_service = BenchmarkService(client)
    quality_evaluator = QualityEvaluator(QualityConfig(minimum_score=request.quality_threshold))
    search_generator = SearchSpaceGenerator(client)
    optimizer = AdaptiveOptimizer(client, benchmark_service, quality_evaluator, search_generator)

    _current_optimizer = optimizer
    _current_run_id = optimizer.state.run.id if optimizer.state else None

    # Run in background
    background_tasks.add_task(
        run_optimization_task,
        optimizer,
        model,
        hardware,
        request.profile,
        request.custom_weights,
        request.quality_threshold,
        request.advanced_settings,
    )

    return _convert_run(optimizer.state.run)


async def run_optimization_task(
    optimizer: AdaptiveOptimizer,
    model: ModelIdentity,
    hardware: HardwareInfo,
    profile: OptimizationProfileSchema,
    custom_weights: ProfileWeightsSchema | None,
    quality_threshold: float,
    advanced_settings: AdvancedSettingsSchema | None,
):
    """Background task for optimization."""
    try:
        weights = ProfileWeights(**custom_weights.dict()) if custom_weights else None
        advanced = advanced_settings.dict() if advanced_settings else None
        await optimizer.optimize(
            model,
            hardware,
            OptimizationProfile(profile.value),
            weights,
            quality_threshold,
            advanced,
        )
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error("Optimization task failed", error=str(e))
    finally:
        global _current_optimizer, _current_run_id
        _current_optimizer = None
        _current_run_id = None


@router.post("/api/optimize/{run_id}/pause")
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


@router.post("/api/optimize/{run_id}/resume")
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


@router.post("/api/optimize/{run_id}/cancel")
async def cancel_optimization(run_id: UUID):
    """Cancel current optimization."""
    global _current_optimizer
    if (
        _current_optimizer
        and _current_optimizer.state
        and _current_optimizer.state.run.id == run_id
    ):
        _current_optimizer.cancel()
        return {"success": True}
    raise HTTPException(status_code=404, detail="Run not found or not running")


@router.get("/api/optimize/{run_id}/progress", response_model=RunProgressResponse)
async def get_optimization_progress(run_id: UUID):
    """Get live optimization progress."""
    global _current_optimizer
    if (
        _current_optimizer
        and _current_optimizer.state
        and _current_optimizer.state.run.id == run_id
    ):
        state = _current_optimizer.state
        total = state.search_space.estimate_size()
        tested = len(state.tested_configs)
        passed = sum(1 for c in state.tested_configs if c.status == ConfigurationStatus.PASSED)
        failed = sum(1 for c in state.tested_configs if c.status == ConfigurationStatus.FAILED)
        oom = sum(1 for c in state.tested_configs if c.status == ConfigurationStatus.OOM)

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
                            "ttft_ms": m.ttft_ms,
                            "vram_gb": c.peak_vram_gb,
                            "quality": c.quality_score.overall if c.quality_score else None,
                        }
                    break

        best_score = 0
        best_config = None
        if state.best_config:
            best_score = state.best_config.score
            best_config = _convert_config(state.best_config.config)

        return RunProgressResponse(
            run_id=run_id,
            stage=state.run.stage.value,
            progress=min(100, int(tested / max(total, 1) * 100)),
            configs_tested=tested,
            configs_total=total,
            configs_passed=passed,
            configs_failed=failed,
            configs_oom=oom,
            current_config=current_config,
            current_metrics=current_metrics,
            best_score=best_score,
            best_config=best_config,
        )

    # Return saved progress
    run = run_repo.get(str(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return RunProgressResponse(
        run_id=run_id,
        stage=run.stage.value,
        progress=100 if run.status == RunStatus.COMPLETED else 0,
        configs_tested=len(run.configurations),
        configs_total=0,
        configs_passed=sum(1 for c in run.configurations if c.status == ConfigurationStatus.PASSED),
        configs_failed=sum(1 for c in run.configurations if c.status == ConfigurationStatus.FAILED),
        configs_oom=sum(1 for c in run.configurations if c.status == ConfigurationStatus.OOM),
    )


# ============================================================
# Results
# ============================================================


@router.get("/api/runs")
async def list_runs(limit: int = 50, offset: int = 0, model_id: str | None = None):
    """List optimization runs."""
    if model_id:
        runs = run_repo.get_by_model(model_id, limit)
    else:
        runs = run_repo.list_all(limit, offset)
    return {"runs": [_convert_run(r) for r in runs]}


@router.get("/api/runs/{run_id}", response_model=OptimizationRunResponse)
async def get_run(run_id: UUID):
    """Get optimization run details."""
    run = run_repo.get(str(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return _convert_run(run)


@router.get("/api/runs/{run_id}/configurations")
async def list_configurations(run_id: UUID):
    """List all configurations for a run."""
    run = run_repo.get(str(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"configurations": [_convert_config_result(c) for c in run.configurations]}


@router.get(
    "/api/runs/{run_id}/configurations/{config_id}", response_model=ConfigurationResultResponse
)
async def get_configuration(run_id: UUID, config_id: UUID):
    """Get configuration result details."""
    config = config_repo.get(str(config_id))
    if not config or str(config.run_id) != str(run_id):
        raise HTTPException(status_code=404, detail="Configuration not found")
    return _convert_config_result(config)


@router.get("/api/runs/{run_id}/pareto")
async def get_pareto_frontier(run_id: UUID):
    """Get Pareto frontier configurations."""
    run = run_repo.get(str(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    pareto = run.get_pareto_configs()
    return {"configurations": [_convert_config_result(c) for c in pareto]}


# ============================================================
# Apply Configuration
# ============================================================


@router.post("/api/apply")
async def apply_configuration(request: ApplyConfigRequest):
    """Apply a configuration to LM Studio."""
    client = await get_lm_client()
    try:
        result = await client.load_model(
            request.model_id, LoadConfiguration(**request.config.dict())
        )
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error or "Failed to load model")
        return {"success": True, "identifier": result.identifier}
    finally:
        await client.close()


@router.post("/api/restore/{model_id}")
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


@router.get("/api/presets")
async def list_presets(model_id: str | None = None):
    """List saved presets."""
    if model_id:
        presets = preset_repo.list_by_model(model_id)
    else:
        presets = preset_repo.list_all()
    return {"presets": [_convert_preset(p) for p in presets]}


@router.post("/api/presets")
async def save_preset(preset: PresetSchema):
    """Save a preset."""
    preset_data = preset.dict()
    preset_data["id"] = str(preset.id)
    preset_id = preset_repo.save(preset_data)
    return {"id": preset_id, "success": True}


@router.delete("/api/presets/{preset_id}")
async def delete_preset(preset_id: UUID):
    """Delete a preset."""
    success = preset_repo.delete(str(preset_id))
    if not success:
        raise HTTPException(status_code=404, detail="Preset not found")
    return {"success": True}


@router.post("/api/presets/{preset_id}/apply")
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


@router.get("/api/settings", response_model=SettingsSchema)
async def get_settings():
    """Get all settings."""
    settings = settings_repo.get_all()
    return SettingsSchema(**{k: v["value"] for k, v in settings.items()})


@router.put("/api/settings")
async def update_settings(settings: SettingsSchema):
    """Update settings."""
    for key, value in settings.dict().items():
        settings_repo.set(key, str(value))
    return {"success": True}


# ============================================================
# Export/Import
# ============================================================


@router.get("/api/runs/{run_id}/export")
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
            lines.extend(
                [
                    f"- **Context:** {best.context_length}",
                    f"- **GPU Ratio:** {best.gpu_ratio}",
                    f"- **Flash Attention:** {best.flash_attention}",
                    f"- **KV Cache:** {'GPU' if best.offload_kv_cache_to_gpu else 'CPU'}",
                    f"- **Batch Size:** {best.eval_batch_size}",
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
        lines.append(
            f"- {c.context_length} ctx, GPU {c.gpu_ratio}, {c.get_avg_generation_tok_s():.1f} tok/s, Q={c.quality_score.overall:.3f} {'✓' if c.status == ConfigurationStatus.PASSED else '✗'}"
        )

    return "\n".join(lines)


# Import logger
import structlog

logger = structlog.get_logger(__name__)
