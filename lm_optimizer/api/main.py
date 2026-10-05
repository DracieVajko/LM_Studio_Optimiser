"""FastAPI application for LM Studio Auto Optimizer."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from lm_optimizer.api.routes import router as api_router
from lm_optimizer.api.websocket import ws_router
from lm_optimizer.config import config
from lm_optimizer.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# Setup logging
setup_logging()

templates = Jinja2Templates(directory="lm_optimizer/ui/templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler (shutdown writes checkpoint first)."""
    logger.info("Starting LM Studio Auto Optimizer")
    try:
        from lm_optimizer.api import routes as _routes

        await _routes.warm_capability_cache()
    except Exception as e:
        logger.info("Startup warmup failed", error=str(e))
    yield
    try:
        from lm_optimizer.api import routes as _routes

        opt = _routes._current_optimizer
        if opt is not None and opt.state is not None:
            opt.checkpoint_now(reason="shutdown")
            logger.info("Shutdown checkpoint saved", run_id=str(opt.state.run.id))
    except Exception as e:
        logger.info("Shutdown checkpoint failed", error=str(e))
    logger.info("Shutting down LM Studio Auto Optimizer")


app = FastAPI(
    title="LM Studio Auto Optimizer",
    description="Finds empirically validated configurations optimized for your hardware, model and selected goal",
    version="1.5.0b1",
    lifespan=lifespan,
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(api_router, prefix="/api")
app.include_router(ws_router)

# Static files
app.mount("/static", StaticFiles(directory="lm_optimizer/ui/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard."""
    from lm_optimizer.services.hardware import hardware_detector

    hardware = hardware_detector.detect()
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "hardware": hardware,
            "lm_studio_url": config.lm_studio.base_url,
        },
    )


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    """History page."""
    return templates.TemplateResponse(request, "history.html")


@app.get("/results/{run_id}", response_class=HTMLResponse)
async def results_page(request: Request, run_id: str):
    """Results page for a specific run."""
    return templates.TemplateResponse(
        request,
        "results.html",
        {
            "run_id": run_id,
        },
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page."""
    return templates.TemplateResponse(request, "settings.html")


@app.get("/results/{run_id}/configs/{config_id}", response_class=HTMLResponse)
async def config_json_page(request: Request, run_id: str, config_id: str):
    """Clean pretty-printed JSON of one tested configuration."""
    import json as _json

    from fastapi import HTTPException as _HE

    from lm_optimizer.api.routes import (
        _convert_config_result,
        _enrich_config_detail,
        config_repo,
        run_repo,
    )

    config = config_repo.get(config_id)
    if not config or str(config.run_id) != str(run_id):
        raise _HE(status_code=404, detail="Configuration not found")
    model_id = ""
    try:
        run = run_repo.get(str(run_id))
        if run is not None and getattr(run, "model", None) is not None:
            model_id = run.model.id or ""
    except Exception:
        pass
    conv = _convert_config_result(config)
    enriched = _enrich_config_detail(conv.model_dump(mode="json"), config, model_id)
    payload = _json.dumps(enriched, indent=1, ensure_ascii=False, default=str)
    return templates.TemplateResponse(
        request,
        "json.html",
        {"payload": payload, "run_id": str(run_id), "config_id": str(config_id)},
    )


@app.get("/sandbox", response_class=HTMLResponse)
async def sandbox_page(request: Request):
    """Model-vs-model sandbox page (text / html / scene)."""
    return templates.TemplateResponse(request, "sandbox.html")


@app.get("/deep", response_class=HTMLResponse)
async def deep_page(request: Request):
    """Deep-research benchmark leaderboard + output preview."""
    return templates.TemplateResponse(request, "deep.html")


@app.get("/sandbox/files/{job_id}/{side}/index.html")
async def sandbox_file(job_id: str, side: str):
    """Serve a duel-generated index.html. Fixed name + A/B only: no traversal."""
    import re

    from fastapi.responses import Response

    if side not in ("A", "B"):
        from fastapi import HTTPException as _HE

        raise _HE(status_code=400, detail="side must be A or B")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", job_id):
        from fastapi import HTTPException as _HE

        raise _HE(status_code=400, detail="invalid job id")
    from lm_optimizer.services import sandbox as _sb

    root = _sb.DEFAULT_SANDBOX_ROOT
    target = root / job_id / side / "index.html"
    try:
        resolved = target.resolve()
        if resolved.parent != (root / job_id / side).resolve():
            raise ValueError("traversal")
        html = resolved.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError, ValueError):
        from fastapi import HTTPException as _HE

        raise _HE(status_code=404, detail="file not found")
    return Response(content=html, media_type="text/html")


# Import at end
