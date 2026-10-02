"""Ollama backend client (mocked-httpx tested; no live calls in tests).

Endpoint mapping (spec `docs/specs/2026-10-02-multi-backend-design.md`):
- Discovery: ``GET /api/tags``; details: ``POST /api/show``.
- Sweep: ``POST /api/generate`` (``stream=false``) with per-request ``options``.
- Metrics: ``eval_count/eval_duration`` (gen tok/s), ``prompt_eval_*``,
  ``load_duration`` (all durations are nanoseconds server-side).
- Unload: ``keep_alive: 0`` + verify empty via ``GET /api/ps`` (fail-closed).

httpx usage (timeouts, tenacity retry style, error extraction) mirrors
``lm_optimizer/services/lm_studio.py`` for consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lm_optimizer.domain.models import ModelIdentity
from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)

OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434"

# Slow-model allowances: big loads/generations legitimately take minutes.
GENERATE_TIMEOUT_S = 600.0
SHOW_TIMEOUT_S = 60.0

OLLAMA_MAX_RETRIES = 3
OLLAMA_RETRY_DELAY = 1.0


@dataclass
class OllamaCapabilities:
    """Detected Ollama server capabilities (static table lands in Task 3)."""

    version: str = "unknown"
    options: list[str] = field(default_factory=list)


@dataclass
class OllamaMetrics:
    """Normalized generation metrics.

    Rates are computed as ``count/duration_ns*1e9``; ``load_ms`` is
    ``load_duration_ns/1e6``. Zero durations yield 0.0 (never NaN/inf).
    """

    text: str
    gen_tok_s: float = 0.0
    prompt_tok_s: float = 0.0
    load_ms: float = 0.0
    eval_count: int = 0
    prompt_eval_count: int = 0
    total_duration_ns: int = 0
    done_reason: str | None = None
    raw: dict = field(default_factory=dict)

    @property
    def response(self) -> str:
        """Alias for ``text`` (Ollama names the field ``response``)."""
        return self.text


def _error_message(exc: httpx.HTTPStatusError) -> str:
    """Extract server error message from an HTTP error response."""
    try:
        data = exc.response.json()
        err = data.get("error", data)
        if isinstance(err, dict):
            return str(err.get("message", err))
        return str(err)
    except Exception:
        return f"HTTP {exc.response.status_code}"


def _rate_per_s(count: int, duration_ns: int) -> float:
    """Tokens/sec from Ollama nanosecond durations (0.0 when unmeasurable)."""
    if count <= 0 or duration_ns <= 0:
        return 0.0
    return count / duration_ns * 1e9


class OllamaClient:
    """Ollama API client implementing the ``BackendClient`` seam."""

    backend_name: str = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 120.0,
        transport: httpx.AsyncHTTPTransport | None = None,
    ):
        self.base_url = (base_url or OLLAMA_DEFAULT_URL).rstrip("/")
        self.timeout = timeout
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._capabilities: OllamaCapabilities | None = None
        self._connected = False
        self._models_cache: list[ModelIdentity] = []

    async def __aenter__(self) -> "OllamaClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def _ensure_client(self) -> httpx.AsyncClient:
        # Unlike LMStudioClient (which raises when unconnected), the HTTP
        # client is created lazily so seam calls such as list_models() work
        # without an explicit connect() (tests construct bare clients).
        if self._client is None:
            kwargs: dict = {}
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                **kwargs,
            )
        return self._client

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client (created lazily)."""
        return self._ensure_client()

    @property
    def capabilities(self) -> OllamaCapabilities:
        """Get detected server capabilities."""
        if self._capabilities is None:
            raise RuntimeError("Capabilities not detected. Call connect() first.")
        return self._capabilities

    async def connect(self) -> None:
        """Establish connection and detect capabilities (idempotent)."""
        if self._connected and self._client is not None:
            return
        self._ensure_client()
        try:
            await self._request_with_retry("GET", "/api/tags")
        except Exception as e:
            raise ConnectionError(f"Ollama server unreachable at {self.base_url}: {e}")
        version = "unknown"
        try:
            response = await self._request_with_retry("GET", "/api/version")
            version = str(response.json().get("version", "unknown"))
        except Exception:
            logger.debug("Ollama version probe failed; continuing", base_url=self.base_url)
        self._capabilities = OllamaCapabilities(version=version)
        self._connected = True
        await self.list_models(force_refresh=True)
        logger.info("Connected to Ollama", base_url=self.base_url, version=version)

    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        json_data: dict | None = None,
        params: dict | None = None,
        timeout_s: float | None = None,
    ) -> httpx.Response:
        """Make HTTP request with retry logic (transient errors only).

        timeout_s overrides the default client timeout for slow operations
        (big-model generations legitimately take minutes).
        """
        retry_config = AsyncRetrying(
            stop=stop_after_attempt(OLLAMA_MAX_RETRIES),
            wait=wait_exponential(multiplier=OLLAMA_RETRY_DELAY, min=1, max=10),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)
            ),
            reraise=True,
        )

        request_timeout = httpx.Timeout(timeout_s) if timeout_s else None
        async for attempt in retry_config:
            with attempt:
                kwargs: dict = {}
                if request_timeout is not None:
                    kwargs["timeout"] = request_timeout
                response = await self.client.request(
                    method, path, json=json_data, params=params, **kwargs
                )
                response.raise_for_status()
                return response

        raise RuntimeError("Retry logic failed unexpectedly")

    async def list_models(self, force_refresh: bool = False) -> list[ModelIdentity]:
        """List all available models via ``GET /api/tags``."""
        if self._models_cache and not force_refresh:
            return self._models_cache

        response = await self._request_with_retry("GET", "/api/tags")
        data = response.json()

        models = []
        items = data.get("models", []) if isinstance(data, dict) else []
        for item in items:
            name = item.get("name") or item.get("model", "")
            details = item.get("details") or {}
            models.append(
                ModelIdentity(
                    id=name,
                    name=name,
                    quantization=details.get("quantization_level"),
                    size_bytes=item.get("size"),
                )
            )

        self._models_cache = models
        return models

    async def get_model(self, model_id: str) -> ModelIdentity | None:
        """Get specific model information."""
        for model in await self.list_models():
            if model.id == model_id:
                return model
        return None

    async def show(self, model_id: str) -> dict:
        """Return raw ``POST /api/show`` details (modelfile, model_info)."""
        try:
            response = await self._request_with_retry(
                "POST", "/api/show", json_data={"model": model_id}, timeout_s=SHOW_TIMEOUT_S
            )
        except httpx.HTTPStatusError as e:
            raise ValueError(f"Ollama model {model_id!r} not found: {_error_message(e)}")
        return response.json()

    async def load_model(self, model_id: str, config: object = None) -> dict:
        """Verify a model exists (Ollama loads implicitly on generate).

        Per-request ``options`` are applied by ``generate``; the load cost
        surfaces in ``load_duration`` metrics rather than here.
        """
        try:
            await self._request_with_retry(
                "POST", "/api/show", json_data={"model": model_id}, timeout_s=SHOW_TIMEOUT_S
            )
        except httpx.HTTPStatusError as e:
            return {"success": False, "model": model_id, "error": _error_message(e)}
        return {"success": True, "model": model_id}

    async def generate(
        self, model_id: str, prompt: str, options: dict | None = None
    ) -> OllamaMetrics:
        """Generate via ``POST /api/generate`` (stream=false), metrics normalized."""
        body: dict = {
            "model": model_id,
            "prompt": prompt,
            "stream": False,
            "options": dict(options or {}),
        }
        response = await self._request_with_retry(
            "POST", "/api/generate", json_data=body, timeout_s=GENERATE_TIMEOUT_S
        )
        data = response.json()

        eval_count = int(data.get("eval_count") or 0)
        eval_duration = int(data.get("eval_duration") or 0)
        prompt_count = int(data.get("prompt_eval_count") or 0)
        prompt_duration = int(data.get("prompt_eval_duration") or 0)
        load_duration = int(data.get("load_duration") or 0)
        return OllamaMetrics(
            text=str(data.get("response", "")),
            gen_tok_s=_rate_per_s(eval_count, eval_duration),
            prompt_tok_s=_rate_per_s(prompt_count, prompt_duration),
            load_ms=load_duration / 1e6,
            eval_count=eval_count,
            prompt_eval_count=prompt_count,
            total_duration_ns=int(data.get("total_duration") or 0),
            done_reason=data.get("done_reason"),
            raw=data,
        )

    async def create(
        self, name: str, *, from_: str, parameters: dict | None = None
    ) -> dict:
        """Persist a new tag via ``POST /api/create`` (``{model, from}`` + params).

        Only new tags — callers must list first and refuse taken names
        (see ``ensure_tag_free`` in ``modelfile.py``); the server has no
        dry-run, so validation happens client-side before this call.
        """
        target = (name or "").strip()
        base = (from_ or "").strip()
        if not target:
            raise ValueError("create: target model name must be non-empty")
        if not base:
            raise ValueError("create: base model (from_) must be non-empty")
        body: dict = {"model": target, "from": base, "stream": False}
        if parameters:
            body["parameters"] = dict(parameters)
        response = await self._request_with_retry(
            "POST", "/api/create", json_data=body, timeout_s=GENERATE_TIMEOUT_S
        )
        try:
            data = response.json()
        except Exception:
            data = {}
        return data if isinstance(data, dict) else {"status": data}

    async def unload(self, model_id: str) -> bool:
        """Unload a model via ``keep_alive: 0``, verified empty (fail-closed).

        Returns True only when ``GET /api/ps`` no longer lists the model.
        """
        try:
            response = await self._request_with_retry(
                "POST",
                "/api/generate",
                json_data={"model": model_id, "prompt": "", "stream": False, "keep_alive": 0},
            )
            unloaded = response.json().get("done_reason") == "unload"
        except httpx.HTTPStatusError as e:
            logger.warning("Ollama unload failed", model=model_id, error=_error_message(e))
            return False
        except Exception as e:
            logger.warning("Ollama unload failed", model=model_id, error=str(e))
            return False

        try:
            ps = await self._request_with_retry("GET", "/api/ps")
            running = [
                m.get("name", m.get("model"))
                for m in (ps.json().get("models", []) if isinstance(ps.json(), dict) else [])
            ]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return unloaded  # old server without /api/ps: trust done_reason
            logger.warning("Ollama unload verify failed", model=model_id, error=str(e))
            return False
        except Exception as e:
            logger.warning("Ollama unload verify failed", model=model_id, error=str(e))
            return False

        if model_id in running:
            logger.warning("Model still loaded after unload", model=model_id)
            return False
        return True

    async def health_check(self) -> bool:
        """Check if Ollama is responsive."""
        try:
            await self._request_with_retry("GET", "/api/tags")
            return True
        except Exception:
            return False
