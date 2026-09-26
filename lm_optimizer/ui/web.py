"""DEPRECATED LEGACY WEB UI - will be removed in 2.0. Use lm_optimizer.api.main:app instead.

This file uses the legacy optimizer/engine.py and api/client.py. The authoritative Web UI is FastAPI app in api/main.py
(used by lm_optimizer.web_main). Kept for reference only.
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from lm_optimizer.api.client import LMStudioClient, LoadConfig
from lm_optimizer.benchmark.runner import BenchmarkRunner
from lm_optimizer.config import config
from lm_optimizer.hardware.detection import detect_hardware
from lm_optimizer.logging_config import get_logger
from lm_optimizer.optimizer.engine import OptimizationEngine
from lm_optimizer.profiles.registry import ProfileRegistry
from lm_optimizer.storage.checkpoint import PresetStorage, ResultsStorage

logger = get_logger(__name__)

# Global state for web UI
app_state = {
    "client": None,
    "optimization_task": None,
    "current_progress": {},
    "connected_websockets": set(),
}

templates = Jinja2Templates(directory="lm_optimizer/ui/templates")


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)


ws_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    client = LMStudioClient()
    try:
        await client.connect()
        app_state["client"] = client
        logger.info("Web UI started, connected to LM Studio")
    except Exception as e:
        logger.warning("Could not connect to LM Studio on startup", error=str(e))

    yield

    # Shutdown
    if app_state["client"]:
        await app_state["client"].close()
    logger.info("Web UI shutdown")


app = FastAPI(
    title="LM Studio Auto Optimizer",
    description="Web UI for optimizing LM Studio model configurations",
    lifespan=lifespan,
)


# Serve static files if they exist
static_dir = Path("lm_optimizer/ui/static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard."""
    hardware = detect_hardware()
    hardware_info = {
        "gpu": hardware.gpu.name if hardware.gpu else "Not detected",
        "vram_gb": hardware.gpu.vram_gb if hardware.gpu else 0,
        "ram_gb": hardware.memory.total_gb,
        "cpu": hardware.cpu.name,
        "os": hardware.os,
    }

    # Get models
    models = []
    if app_state["client"]:
        try:
            models_data = await app_state["client"].list_models()
            models = [
                {
                    "id": m.id,
                    "name": m.name,
                    "architecture": m.architecture,
                    "context_length": m.context_length,
                    "max_context_length": m.max_context_length,
                    "quantization": m.quantization,
                    "loaded": m.loaded,
                }
                for m in models_data
            ]
        except Exception as e:
            logger.error("Failed to list models", error=str(e))

    # Get presets
    preset_storage = PresetStorage()
    presets = preset_storage.list_presets()

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "hardware": hardware_info,
            "models": models,
            "presets": presets,
            "lm_studio_url": config.lm_studio.base_url,
            "profiles": [p.name for p in ProfileRegistry().list_profiles()],
        },
    )


@app.get("/api/status")
async def api_status():
    """Get system status."""
    hardware = detect_hardware()
    client = app_state["client"]

    status = {
        "lm_studio": {
            "connected": client is not None,
            "url": config.lm_studio.base_url,
        },
        "hardware": {
            "gpu": hardware.gpu.name if hardware.gpu else "Not detected",
            "vram_gb": hardware.gpu.vram_gb if hardware.gpu else 0,
            "ram_gb": hardware.memory.total_gb,
            "cpu": hardware.cpu.name,
        },
        "optimization": {
            "running": app_state["optimization_task"] is not None,
            "progress": app_state["current_progress"],
        },
    }
    return JSONResponse(status)


@app.get("/api/models")
async def api_models():
    """List models."""
    client = app_state["client"]
    if not client:
        return JSONResponse({"error": "Not connected to LM Studio"}, status_code=503)

    try:
        models = await client.list_models()
        return JSONResponse(
            {
                "models": [
                    {
                        "id": m.id,
                        "name": m.name,
                        "architecture": m.architecture,
                        "context_length": m.context_length,
                        "max_context_length": m.max_context_length,
                        "quantization": m.quantization,
                        "loaded": m.loaded,
                    }
                    for m in models
                ]
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/optimize")
async def api_optimize(request: Request):
    """Start optimization."""
    client = app_state["client"]
    if not client:
        return JSONResponse({"error": "Not connected to LM Studio"}, status_code=503)

    data = await request.json()
    model_id = data.get("model")
    profile = data.get("profile", "balanced")

    if not model_id:
        return JSONResponse({"error": "Model ID required"}, status_code=400)

    # Check if already running
    if app_state["optimization_task"] and not app_state["optimization_task"].done():
        return JSONResponse({"error": "Optimization already running"}, status_code=409)

    # Start optimization in background
    async def run_optimization():
        try:
            app_state["current_progress"] = {"stage": "starting", "progress": 0}
            await ws_manager.broadcast({"type": "progress", "data": app_state["current_progress"]})

            engine = OptimizationEngine(client)

            # Progress callback would need to be implemented in engine
            result = await engine.optimize(model_id, profile)

            app_state["current_progress"] = {"stage": "complete", "progress": 100}
            await ws_manager.broadcast({"type": "complete", "data": _serialize_result(result)})

        except Exception as e:
            logger.exception("Optimization failed")
            app_state["current_progress"] = {"stage": "error", "error": str(e)}
            await ws_manager.broadcast({"type": "error", "data": {"error": str(e)}})
        finally:
            app_state["optimization_task"] = None

    app_state["optimization_task"] = asyncio.create_task(run_optimization())
    return JSONResponse({"status": "started", "model": model_id, "profile": profile})


@app.post("/api/benchmark")
async def api_benchmark(request: Request):
    """Run a single benchmark."""
    client = app_state["client"]
    if not client:
        return JSONResponse({"error": "Not connected to LM Studio"}, status_code=503)

    data = await request.json()
    model_id = data.get("model")
    load_config = LoadConfig(**data.get("config", {}))
    context = data.get("context", 4096)

    if not model_id:
        return JSONResponse({"error": "Model ID required"}, status_code=400)

    try:
        runner = BenchmarkRunner(client)
        result = await runner.run_benchmark(model_id, load_config, context)
        return JSONResponse(_serialize_result(result))
    except Exception as e:
        logger.exception("Benchmark failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/apply")
async def api_apply(request: Request):
    """Apply a preset."""
    client = app_state["client"]
    if not client:
        return JSONResponse({"error": "Not connected to LM Studio"}, status_code=503)

    data = await request.json()
    model_id = data.get("model")
    profile = data.get("profile", "balanced")

    if not model_id:
        return JSONResponse({"error": "Model ID required"}, status_code=400)

    preset_storage = PresetStorage()
    preset = preset_storage.load_preset(model_id, profile)

    if not preset:
        return JSONResponse({"error": "Preset not found"}, status_code=404)

    try:
        load_config = LoadConfig(**preset["recommended_config"])
        result = await client.load_model(model_id, load_config)

        if result.success:
            return JSONResponse({"status": "success", "identifier": result.identifier})
        return JSONResponse({"error": result.error}, status_code=500)
    except Exception as e:
        logger.exception("Apply failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/results")
async def api_results(model: str | None = None):
    """Get benchmark results."""
    results_storage = ResultsStorage()
    results = results_storage.list_results(model)
    return JSONResponse({"results": results})


@app.get("/api/presets")
async def api_presets():
    """List presets."""
    preset_storage = PresetStorage()
    presets = preset_storage.list_presets()
    return JSONResponse({"presets": presets})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
            await websocket.send_json({"type": "ack", "data": data})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


def _serialize_result(result) -> dict:
    """Serialize benchmark result for JSON."""
    return {
        "config_id": result.config_id,
        "model_id": result.model_id,
        "context_length": result.context_length,
        "gpu_ratio": result.gpu_ratio,
        "flash_attention": result.flash_attention,
        "kv_cache_gpu": result.kv_cache_gpu,
        "eval_batch_size": result.eval_batch_size,
        "num_experts": result.num_experts,
        "timestamp": result.timestamp.isoformat(),
        "generation_tok_s": result.get_avg_generation_tok_s(),
        "prompt_tok_s": result.get_avg_prompt_tok_s(),
        "ttft_ms": result.get_avg_ttft_ms(),
        "quality_score": result.quality_score,
        "stability_score": result.stability_score,
        "passed": result.passed,
        "failure_reason": result.failure_reason,
        "metrics": [
            {
                "test_name": m.test_name,
                "category": m.category,
                "success": m.success,
                "generation_tok_s": m.generation_tok_s,
                "prompt_tok_s": m.prompt_tok_s,
                "ttft_ms": m.ttft_ms,
                "quality": m.output_text[:100] if m.output_text else "",
            }
            for m in result.metrics
        ],
    }


def run_web_ui(host: str = "127.0.0.1", port: int = 8080):
    """Run the web UI server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
