"""FastAPI application for LM Studio Auto Optimizer."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
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
    """Application lifespan handler."""
    logger.info("Starting LM Studio Auto Optimizer")
    yield
    logger.info("Shutting down LM Studio Auto Optimizer")


app = FastAPI(
    title="LM Studio Auto Optimizer",
    description="Finds empirically validated configurations optimized for your hardware, model and selected goal",
    version="1.0.0-beta",
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
        "dashboard.html",
        {
            "request": request,
            "hardware": hardware,
            "lm_studio_url": config.lm_studio.base_url,
        },
    )


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    """History page."""
    return templates.TemplateResponse("history.html", {"request": request})


@app.get("/results/{run_id}", response_class=HTMLResponse)
async def results_page(request: Request, run_id: str):
    """Results page for a specific run."""
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "run_id": run_id,
        },
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page."""
    return templates.TemplateResponse("settings.html", {"request": request})


# Import at end
