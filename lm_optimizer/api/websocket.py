"""WebSocket handler for live optimization updates."""

import json
from uuid import UUID

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from lm_optimizer.api.routes import _convert_config, _current_optimizer

logger = structlog.get_logger(__name__)

ws_router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[UUID, set[WebSocket]] = {}

    async def connect(self, run_id: UUID, websocket: WebSocket):
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = set()
        self.active_connections[run_id].add(websocket)

    def disconnect(self, run_id: UUID, websocket: WebSocket):
        if run_id in self.active_connections:
            self.active_connections[run_id].discard(websocket)
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]

    async def broadcast(self, run_id: UUID, message: dict):
        if run_id in self.active_connections:
            dead = set()
            for ws in self.active_connections[run_id]:
                try:
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_json(message)
                except Exception:
                    dead.add(ws)
            for ws in dead:
                self.disconnect(run_id, ws)


manager = ConnectionManager()


@ws_router.websocket("/ws/optimize/{run_id}")
async def optimization_websocket(websocket: WebSocket, run_id: UUID):
    """WebSocket for live optimization updates."""
    await manager.connect(run_id, websocket)

    try:
        # Send initial state
        await send_current_state(run_id, websocket)

        # Listen for messages (ping/pong, etc.)
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        manager.disconnect(run_id, websocket)
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        manager.disconnect(run_id, websocket)


async def send_current_state(run_id: UUID, websocket: WebSocket):
    """Send current optimization state to WebSocket."""
    from lm_optimizer.api.routes import run_repo

    # Check if it's the current running optimization
    if (
        _current_optimizer
        and _current_optimizer.state
        and _current_optimizer.state.run.id == run_id
    ):
        state = _current_optimizer.state
    else:
        # Get from database
        run = run_repo.get(str(run_id))
        if not run:
            await websocket.send_json({"type": "error", "message": "Run not found"})
            return
        state = run

    # Calculate progress
    if hasattr(state, "search_space") and hasattr(state.search_space, "estimate_size"):
        total = state.search_space.estimate_size()
    else:
        total = len(state.configurations) if hasattr(state, "configurations") else 0

    tested = len(state.configurations) if hasattr(state, "configurations") else 0
    passed = sum(1 for c in state.configurations if c.status == ConfigurationStatus.PASSED)
    failed = sum(1 for c in state.configurations if c.status == ConfigurationStatus.FAILED)
    oom = sum(1 for c in state.configurations if c.status == ConfigurationStatus.OOM)

    # Current config
    current_config = None
    current_metrics = None
    if hasattr(state, "current_config_id") and state.current_config_id:
        for c in state.configurations:
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

    # Best config
    best_score = 0
    best_config = None
    if hasattr(state, "best_config") and state.best_config:
        best_score = state.best_config.score
        best_config = _convert_config(state.best_config.config)

    await websocket.send_json(
        {
            "type": "progress",
            "run_id": str(run_id),
            "stage": state.stage.value if hasattr(state, "stage") else "unknown",
            "progress": min(100, int(tested / max(total, 1) * 100)) if total else 0,
            "configs_tested": tested,
            "configs_total": total,
            "configs_passed": passed,
            "configs_failed": failed,
            "configs_oom": oom,
            "current_config": current_config.dict() if current_config else None,
            "current_metrics": current_metrics,
            "best_score": best_score,
            "best_config": best_config.dict() if best_config else None,
            "timestamp": datetime.now().isoformat(),
        }
    )


# Import
from datetime import datetime


async def broadcast_progress(run_id: UUID, progress_data: dict):
    """Broadcast progress update to all connected clients."""
    message = {
        "type": "progress",
        "run_id": str(run_id),
        **progress_data,
        "timestamp": datetime.now().isoformat(),
    }
    await manager.broadcast(run_id, message)


async def broadcast_complete(run_id: UUID, run_data: dict):
    """Broadcast completion."""
    message = {
        "type": "complete",
        "run_id": str(run_id),
        "run": run_data,
        "timestamp": datetime.now().isoformat(),
    }
    await manager.broadcast(run_id, message)


async def broadcast_error(run_id: UUID, error: str):
    """Broadcast error."""
    message = {
        "type": "error",
        "run_id": str(run_id),
        "error": error,
        "timestamp": datetime.now().isoformat(),
    }
    await manager.broadcast(run_id, message)
