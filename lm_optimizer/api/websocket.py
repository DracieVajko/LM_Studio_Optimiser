"""WebSocket handler for live optimization updates."""

import json
from datetime import datetime
from uuid import UUID

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from lm_optimizer.domain.models import ConfigurationStatus

logger = structlog.get_logger(__name__)

ws_router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self) -> None:
        self.active_connections: dict[UUID, set[WebSocket]] = {}

    async def connect(self, run_id: UUID, websocket: WebSocket) -> None:
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = set()
        self.active_connections[run_id].add(websocket)

    def disconnect(self, run_id: UUID, websocket: WebSocket) -> None:
        if run_id in self.active_connections:
            self.active_connections[run_id].discard(websocket)
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]

    async def broadcast(self, run_id: UUID, message: dict) -> None:
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
async def optimization_websocket(websocket: WebSocket, run_id: UUID) -> None:
    """WebSocket for live optimization updates."""
    await manager.connect(run_id, websocket)

    try:
        # Send initial state
        await send_current_state(run_id, websocket)

        # Listen for messages (ping/pong, cancel, etc.)
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif msg.get("type") == "cancel":
                    # Browser-triggered cancellation: checkpoint before ack.
                    from lm_optimizer.api import routes as _routes

                    opt = _routes._current_optimizer
                    if opt and opt.state and opt.state.run.id == run_id:
                        opt.checkpoint_now(reason="browser-cancel")
                    await websocket.send_json({"type": "cancel_ack", "run_id": str(run_id)})
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        manager.disconnect(run_id, websocket)
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        manager.disconnect(run_id, websocket)


def _progress_payload(run_id: UUID, configs: list, extra: dict) -> dict:
    from lm_optimizer.services.failure_states import is_terminal_failure

    tested = len(configs)
    passed = sum(1 for c in configs if c.status == ConfigurationStatus.PASSED)
    failed = sum(
        1 for c in configs
        if is_terminal_failure(c.status)
        and c.status not in (ConfigurationStatus.OOM, ConfigurationStatus.TIMEOUT)
    )
    oom = sum(1 for c in configs if c.status == ConfigurationStatus.OOM)
    timeouts = sum(1 for c in configs if c.status == ConfigurationStatus.TIMEOUT)
    best = max(
        (c for c in configs
         if c.status == ConfigurationStatus.PASSED and c.score is not None),
        key=lambda c: c.score or 0.0,
        default=None,
    )
    elapsed = extra.pop("elapsed_s", 0.0)
    payload = {
        "type": "progress",
        "run_id": str(run_id),
        "stage": extra.pop("stage", "unknown"),
        "model": extra.pop("model", None),
        "progress": 0,
        "configs_tested": tested,
        "configs_total": extra.pop("configs_total", tested),
        "configs_passed": passed,
        "configs_failed": failed,
        "configs_oom": oom,
        "configs_timeouts": timeouts,
        "current_config": extra.pop("current_config", None),
        "current_metrics": extra.pop("current_metrics", None),
        "current_speed": extra.pop("current_speed", 0.0),
        "vram_gb": extra.pop("vram_gb", None),
        "ram_gb": extra.pop("ram_gb", None),
        "best_score": best.score if best else 0.0,
        "best_config": None,
        "context_progress": extra.pop("context_progress", []),
        "elapsed_s": elapsed,
        "remaining": extra.pop("remaining", "adaptive / unknown"),
        "events": extra.pop("events", []),
    }
    if best is not None:
        from lm_optimizer.api.routes import _convert_config

        payload["best_config"] = _convert_config(best.config).model_dump()
        payload["current_speed"] = round(best.get_avg_generation_tok_s(), 1)
        payload["vram_gb"] = best.peak_vram_gb
        payload["ram_gb"] = best.peak_ram_gb
    total = payload["configs_total"] or 0
    payload["progress"] = min(100, int(tested / max(total, 1) * 100)) if total else 0
    # Honest ETA: only when a stable rate exists; else adaptive/unknown.
    if elapsed and tested and total and total > tested:
        rate = tested / max(elapsed, 1e-6)
        payload["eta_s"] = round((total - tested) / rate, 1) if rate > 0 else None
    else:
        payload["eta_s"] = None
    payload.update(extra)
    payload["timestamp"] = datetime.now().isoformat()
    return payload


async def send_current_state(run_id: UUID, websocket: WebSocket) -> None:
    """Send current optimization state to WebSocket."""
    # Lazy import to avoid circular dependency at module load.
    from lm_optimizer.api import routes as _routes

    opt = _routes._current_optimizer
    if opt and opt.state and opt.state.run.id == run_id:
        state = opt.state
        configs = list(state.tested_configs)
        total = state.search_space.estimate_size() if state.search_space else len(configs)
        elapsed = (
            (datetime.now() - state.started_at).total_seconds()
            if getattr(state, "started_at", None)
            else 0.0
        )
        current_config = None
        current_metrics = None
        if state.current_config_id:
            for c in configs:
                if c.id == state.current_config_id:
                    from lm_optimizer.api.routes import _convert_config as _cc

                    current_config = _cc(c.config).model_dump()
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
        payload = _progress_payload(
            run_id,
            configs,
            {
                "stage": state.run.stage.value,
                "model": state.run.model.id,
                "configs_total": total,
                "current_config": current_config,
                "current_metrics": current_metrics,
                "current_speed": (
                    state.best_config.get_avg_generation_tok_s() if state.best_config else 0.0
                ),
                "vram_gb": state.best_config.peak_vram_gb if state.best_config else None,
                "ram_gb": state.best_config.peak_ram_gb if state.best_config else None,
                "context_progress": sorted({c.context_length for c in configs}),
                "elapsed_s": round(elapsed, 1),
                "remaining": "adaptive / unknown",
                "events": list(getattr(state, "event_log", []))[-10:],
            },
        )
        await websocket.send_json(payload)
        # Follow with the compact live log snapshot.
        await websocket.send_json(
            {
                "type": "log",
                "run_id": str(run_id),
                "events": list(getattr(state, "event_log", []))[-20:],
                "timestamp": datetime.now().isoformat(),
            }
        )
        return

    run = _routes.run_repo.get(str(run_id))
    if not run:
        await websocket.send_json({"type": "error", "message": "Run not found"})
        return
    payload = _progress_payload(
        run_id,
        list(run.configurations),
        {"stage": run.stage.value, "model": run.model.id},
    )
    await websocket.send_json(payload)


async def broadcast_progress(run_id: UUID, progress_data: dict) -> None:
    """Broadcast progress update to all connected clients."""
    message = {
        "type": "progress",
        "run_id": str(run_id),
        **progress_data,
        "timestamp": datetime.now().isoformat(),
    }
    await manager.broadcast(run_id, message)


async def broadcast_log(run_id: UUID, event: dict) -> None:
    """Broadcast a single live-log event."""
    await manager.broadcast(
        run_id,
        {
            "type": "log",
            "run_id": str(run_id),
            "events": [event],
            "timestamp": datetime.now().isoformat(),
        },
    )


async def broadcast_complete(run_id: UUID, run_data: dict) -> None:
    """Broadcast completion."""
    message = {
        "type": "complete",
        "run_id": str(run_id),
        "run": run_data,
        "timestamp": datetime.now().isoformat(),
    }
    await manager.broadcast(run_id, message)


async def broadcast_error(run_id: UUID, error: str) -> None:
    """Broadcast error."""
    message = {
        "type": "error",
        "run_id": str(run_id),
        "error": error,
        "timestamp": datetime.now().isoformat(),
    }
    await manager.broadcast(run_id, message)
