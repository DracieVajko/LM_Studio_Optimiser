"""LM Studio API client with async support.

Live-verified contract (2026-09-19): flat load params, unload by instance_id,
native /api/v1/chat with `input` + `max_output_tokens` (see services/lm_studio.py).
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, Field
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lm_optimizer.config import config
from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)

# Fake model for safe capability probing (key validation precedes model
# resolution: unknown keys 400 without loading, known keys 404).
_PROBE_MODEL = "lms-probe-nonexistent-model"

# Slow-model allowances: big loads/generations legitimately take minutes.
LOAD_TIMEOUT_S = 600.0
CHAT_TIMEOUT_FLOOR_S = 300.0

_PROBE_VALUES: dict[str, Any] = {
    "context_length": 512,
    "flash_attention": True,
    "offload_kv_cache_to_gpu": True,
    "eval_batch_size": 256,
    "physical_batch_size": 512,
    "parallel": 2,
    "context_checkpoints": 16,
    "reasoning_budget_message": "",
    "num_experts": 8,
    "speculative_draft_max_tokens": 8,
    "gpu_ratio": 0.5,
    "rope_freq_base": 10000.0,
    "rope_freq_scale": 1.0,
}


class ModelStatus(str, Enum):
    """Model load status."""

    LOADED = "loaded"
    UNLOADED = "unloaded"
    LOADING = "loading"
    ERROR = "error"


class LoadConfig(BaseModel):
    """Model load configuration parameters.

    `gpu_ratio` / `rope_freq_*` are kept for compatibility but are NOT sent
    (server rejects them via REST). `to_api_params()` returns only supported keys.
    """

    API_KEYS: ClassVar[frozenset] = frozenset(
        {
            "context_length",
            "flash_attention",
            "offload_kv_cache_to_gpu",
            "eval_batch_size",
            "physical_batch_size",
            "parallel",
            "context_checkpoints",
            "reasoning_budget_message",
            "num_experts",
            "speculative_draft_max_tokens",
            "speculative_draft_min_tokens",
            "speculative_draft_min_continue_probability",
            "speculative_draft_mtp",
            "speculative_draft_simple",
            "speculative_draft_model",
        }
    )

    context_length: int | None = Field(default=None, description="Context window size")
    gpu_ratio: float | None = Field(
        default=None, ge=0.0, le=1.0, description="GPU offload ratio (0-1, REST-unsupported)"
    )
    flash_attention: bool | None = Field(default=None, description="Enable Flash Attention")
    offload_kv_cache_to_gpu: bool | None = Field(
        default=None, description="Offload KV cache to GPU"
    )
    eval_batch_size: int | None = Field(default=None, ge=1, description="Evaluation batch size")
    physical_batch_size: int | None = Field(default=None, ge=1, description="Physical batch size")
    parallel: int | None = Field(default=None, ge=1, description="Max concurrency")
    context_checkpoints: int | None = Field(default=None, ge=0, description="Context checkpoints")
    reasoning_budget_message: str | None = Field(
        default=None, description="Reasoning budget message"
    )
    num_experts: int | None = Field(
        default=None, ge=1, description="Number of experts for MoE models"
    )
    speculative_draft_max_tokens: int | None = Field(
        default=None, ge=1, description="Speculative draft max tokens"
    )
    speculative_draft_min_tokens: int | None = Field(
        default=None, ge=0, description="Speculative draft min tokens"
    )
    speculative_draft_min_continue_probability: float | None = Field(
        default=None, description="Speculative min continue probability"
    )
    speculative_draft_mtp: bool | None = Field(
        default=None, description="Draft MTP speculative decoding"
    )
    speculative_draft_simple: bool | None = Field(
        default=None, description="Draft simple speculative decoding"
    )
    speculative_draft_model: str | None = Field(
        default=None, description="Draft model for speculative decoding"
    )
    rope_freq_base: float | None = Field(default=None, description="RoPE frequency base")
    rope_freq_scale: float | None = Field(default=None, description="RoPE frequency scale")

    def to_api_params(self) -> dict[str, Any]:
        """Convert to API parameters: only server-supported keys, excluding None."""
        return {k: v for k, v in self.model_dump().items() if k in self.API_KEYS and v is not None}


class ModelInfo(BaseModel):
    """Model information from LM Studio."""

    id: str
    name: str
    description: str | None = None
    architecture: str | None = None
    context_length: int | None = None
    max_context_length: int | None = None
    model_type: str | None = None
    quantization: str | None = None
    size_bytes: int | None = None
    parameter_count: int | None = None
    loaded: bool = False
    load_config: LoadConfig | None = None
    supported_parameters: list[str] = Field(default_factory=list)


class LoadModelResponse(BaseModel):
    """Response from loading a model."""

    success: bool
    identifier: str | None = None
    error: str | None = None
    load_config: LoadConfig | None = None
    echo_mismatches: dict | None = None


class UnloadModelResponse(BaseModel):
    """Response from unloading a model."""

    success: bool
    error: str | None = None


class ChatMessage(BaseModel):
    """Chat message."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Native chat request (/api/v1/chat)."""

    model: str
    input: str
    temperature: float = 0.7
    max_output_tokens: int = 512
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    reasoning: str | None = None


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""

    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Chat completion response (converted from native output[]/stats)."""

    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
    system_fingerprint: str | None = None
    stats: dict | None = None  # raw native stats (real TTFT, tok/s, load time)


class EmbeddingRequest(BaseModel):
    """Embedding request."""

    model: str
    input: str | list[str]


class EmbeddingResponse(BaseModel):
    """Embedding response."""

    object: str
    data: list[dict]
    model: str
    usage: dict


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


def _is_unrecognized_key_error(exc: httpx.HTTPStatusError, key: str) -> bool:
    """True when the server 400s on an unrecognized key naming `key`."""
    if exc.response.status_code != 400:
        return False
    try:
        err = exc.response.json().get("error", {})
        return err.get("code") == "unrecognized_keys" and key in str(err.get("message", ""))
    except Exception:
        return False


def _is_no_reasoning_error(exc: httpx.HTTPStatusError) -> bool:
    """True when the model exposes no reasoning config at all.

    Only this shape poisons the remember-set; bad reasoning VALUES on
    supporting models (allowed sets vary: off/on vs full range) retry
    without remembering.
    """
    if exc.response.status_code != 400:
        return False
    try:
        err = exc.response.json().get("error", {})
        return err.get("code") == "invalid_value" and "does not expose reasoning" in str(
            err.get("message", "")
        )
    except Exception:
        return False


def _is_reasoning_error(exc: httpx.HTTPStatusError) -> bool:
    """True for any reasoning-key 400 (no-config AND bad-value shapes)."""
    if exc.response.status_code != 400:
        return False
    try:
        err = exc.response.json().get("error", {})
        return err.get("code") == "invalid_value" and "reasoning" in str(
            err.get("message", "")
        ).lower()
    except Exception:
        return False


@dataclass
class APICapabilities:
    """Detected API capabilities."""

    version: str = "unknown"
    supports_context_length: bool = True
    supports_gpu_ratio: bool = True
    supports_flash_attention: bool = False
    supports_kv_cache_offload: bool = False
    supports_eval_batch_size: bool = True
    supports_physical_batch_size: bool = True
    supports_parallel: bool = True
    supports_context_checkpoints: bool = True
    supports_num_experts: bool = False
    supports_rope_scaling: bool = False
    load_parameters: list[str] = field(default_factory=list)

    def get_supported_load_params(self) -> list[str]:
        """Get list of supported load parameters."""
        params = []
        if self.supports_context_length:
            params.append("context_length")
        if self.supports_gpu_ratio:
            params.append("gpu_ratio")
        if self.supports_flash_attention:
            params.append("flash_attention")
        if self.supports_kv_cache_offload:
            params.append("offload_kv_cache_to_gpu")
        if self.supports_eval_batch_size:
            params.append("eval_batch_size")
        if self.supports_physical_batch_size:
            params.append("physical_batch_size")
        if self.supports_parallel:
            params.append("parallel")
        if self.supports_context_checkpoints:
            params.append("context_checkpoints")
        if self.supports_num_experts:
            params.append("num_experts")
        if self.supports_rope_scaling:
            params.extend(["rope_freq_base", "rope_freq_scale"])
        for extra in ("reasoning_budget_message", "speculative_draft_max_tokens"):
            if extra in (self.load_parameters or []) and extra not in params:
                params.append(extra)
        return params


class LMStudioClient:
    """Async client for LM Studio API."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or config.lm_studio.base_url).rstrip("/")
        self.timeout = timeout or config.lm_studio.timeout
        self._client: httpx.AsyncClient | None = None
        self._capabilities: APICapabilities | None = None
        self._models_cache: list[ModelInfo] | None = None
        self._loaded_models: dict[str, str] = {}  # model_id -> instance_id
        self._no_reasoning_models: set[str] = set()

    async def __aenter__(self) -> "LMStudioClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def connect(self) -> None:
        """Establish connection and detect capabilities."""
        if self._client is not None:
            return

        from urllib.parse import urlparse

        host = (urlparse(self.base_url).hostname or "").lower()
        if host not in ("127.0.0.1", "localhost", "::1", ""):
            logger.warning(
                "Remote LM Studio: local hardware must NOT score the remote model.",
                base_url=self.base_url,
            )

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        # Test connection and detect API version
        await self._detect_capabilities()
        await self.list_models(force_refresh=True)
        logger.info(
            "Connected to LM Studio", base_url=self.base_url, version=self._capabilities.version
        )

    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client, raising if not connected."""
        if self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._client

    @property
    def capabilities(self) -> APICapabilities:
        """Get detected API capabilities."""
        if self._capabilities is None:
            raise RuntimeError("Capabilities not detected. Call connect() first.")
        return self._capabilities

    async def _detect_capabilities(self) -> None:
        """Detect LM Studio API version and capabilities."""
        try:
            # Verify connection against the v1 models endpoint
            response = await self._request_with_retry("GET", "/api/v1/models")
            response.json()
            self._capabilities = APICapabilities(version="v1")

            # Probe supported parameters (safe: no model loads)
            await self._probe_load_parameters()

        except Exception as e:
            logger.warning("Failed to detect capabilities, using defaults", error=str(e))
            self._capabilities = APICapabilities(version="v1")

    async def _probe_key(self, key: str, value: Any) -> bool:
        """Safe-probe one load key against a fake model. True = accepted."""
        try:
            await self._request_with_retry(
                "POST", "/api/v1/models/load", json_data={"model": _PROBE_MODEL, key: value}
            )
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                try:
                    code = e.response.json().get("error", {}).get("code", "")
                except Exception:
                    code = ""
                return code != "unrecognized_keys"
            return True
        except Exception:
            return False

    def _apply_accepted(self, accepted: set[str]) -> None:
        """Set capability flags from the probed accepted-key set."""
        caps = self._capabilities
        caps.load_parameters = sorted(accepted)
        caps.supports_context_length = "context_length" in accepted
        caps.supports_gpu_ratio = "gpu_ratio" in accepted
        caps.supports_flash_attention = "flash_attention" in accepted
        caps.supports_kv_cache_offload = "offload_kv_cache_to_gpu" in accepted
        caps.supports_eval_batch_size = "eval_batch_size" in accepted
        caps.supports_physical_batch_size = "physical_batch_size" in accepted
        caps.supports_parallel = "parallel" in accepted
        caps.supports_context_checkpoints = "context_checkpoints" in accepted
        caps.supports_num_experts = "num_experts" in accepted
        caps.supports_rope_scaling = "rope_freq_base" in accepted

    async def _probe_load_parameters(self) -> None:
        """Probe which load parameters are supported.

        Safe phase (no loads): per-key 400-probing against a fake model.
        Echo phase: one real load of the smallest LLM with unload in `finally`.
        """
        if not self._capabilities:
            return

        accepted = {
            key for key, value in _PROBE_VALUES.items() if await self._probe_key(key, value)
        }
        self._apply_accepted(accepted)

        smallest = None
        identifier = None
        try:
            models = await self.list_models()
            candidates = [
                m
                for m in models
                if (m.size_bytes or float("inf")) < float("inf")
                and "embed" not in (m.id or "").lower()
            ]
            if not candidates:
                return
            smallest = min(candidates, key=lambda m: m.size_bytes or 0)
            probe_ctx = min(512, smallest.max_context_length or smallest.context_length or 512)
            result = await self.load_model(smallest.id, LoadConfig(context_length=probe_ctx))
            if result.success:
                identifier = result.identifier
                echo = (
                    result.load_config.model_dump(exclude_none=True) if result.load_config else {}
                )
                logger.info("Probe echo", model=smallest.id, echo=echo)
        except Exception as e:
            logger.warning("Probe echo load failed", error=str(e))
        finally:
            # Unload by captured identifier (cache-independent); fall back to model id.
            if identifier:
                try:
                    await self.unload_model(identifier=identifier)
                except Exception:
                    pass
            elif smallest is not None:
                try:
                    await self.unload_model(model_id=smallest.id)
                except Exception:
                    pass

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        json_data: dict | None = None,
        params: dict | None = None,
        timeout_s: float | None = None,
    ) -> httpx.Response:
        """Make HTTP request with retry logic.

        timeout_s overrides the default client timeout for slow operations
        (big-model loads/generations legitimately take minutes).
        """
        retry_config = AsyncRetrying(
            stop=stop_after_attempt(config.lm_studio.max_retries),
            wait=wait_exponential(multiplier=config.lm_studio.retry_delay, min=1, max=10),
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

    async def list_models(self, force_refresh: bool = False) -> list[ModelInfo]:
        """List all available models."""
        if self._models_cache is not None and not force_refresh:
            return self._models_cache

        response = await self._request_with_retry("GET", "/api/v1/models")
        data = response.json()

        models = []
        items = data.get("models", []) if isinstance(data, dict) else []

        for item in items:
            quant = item.get("quantization") or {}
            quant_name = quant.get("name") if isinstance(quant, dict) else quant
            loaded_instances = item.get("loaded_instances") or []
            echo = (loaded_instances[0].get("config") or {}) if loaded_instances else {}
            model = ModelInfo(
                id=item.get("key", ""),
                name=item.get("display_name") or item.get("key", ""),
                description=item.get("description"),
                architecture=item.get("architecture"),
                context_length=item.get("max_context_length"),
                max_context_length=item.get("max_context_length"),
                model_type=item.get("type"),
                quantization=quant_name,
                size_bytes=item.get("size_bytes"),
                parameter_count=item.get("parameter_count"),
                loaded=bool(loaded_instances),
                load_config=LoadConfig(
                    **{k: v for k, v in echo.items() if k in LoadConfig.model_fields}
                )
                if echo
                else None,
            )
            models.append(model)

        self._models_cache = models
        return models

    async def get_loaded_instances(self) -> list[dict]:
        """List currently loaded instances across all models (server state)."""
        response = await self._request_with_retry("GET", "/api/v1/models")
        data = response.json()
        items = data.get("models", []) if isinstance(data, dict) else []
        instances = []
        for item in items:
            for inst in item.get("loaded_instances") or []:
                instances.append(
                    {
                        "instance_id": inst.get("id"),
                        "model": item.get("key", ""),
                        "config": inst.get("config") or {},
                    }
                )
        self._loaded_models = {
            inst["model"]: inst["instance_id"]
            for inst in instances
            if inst["model"] and inst["instance_id"]
        }
        return instances

    async def get_model(self, model_id: str) -> ModelInfo | None:
        """Get specific model information."""
        models = await self.list_models()
        for model in models:
            if model.id == model_id:
                return model
        return None

    async def _read_echo_config(self, instance_id: str) -> LoadConfig | None:
        """Read the server-echoed applied config for an instance (best effort)."""
        try:
            instances = await self.get_loaded_instances()
            for inst in instances:
                if inst["instance_id"] == instance_id:
                    echo = inst.get("config") or {}
                    return LoadConfig(
                        **{k: v for k, v in echo.items() if k in LoadConfig.model_fields}
                    )
        except Exception as e:
            logger.debug("Echo read failed", error=str(e))
        return None

    async def load_model(
        self, model_id: str, load_config: LoadConfig | None = None
    ) -> LoadModelResponse:
        """Load a model with specified configuration (flat parameters + echo verify)."""
        api_params = load_config.to_api_params() if load_config is not None else {}
        body: dict[str, Any] = {"model": model_id, **api_params, "echo_load_config": True}

        try:
            response = await self._request_with_retry(
                "POST", "/api/v1/models/load", json_data=body, timeout_s=LOAD_TIMEOUT_S
            )
        except httpx.HTTPStatusError as e:
            if _is_unrecognized_key_error(e, "echo_load_config"):
                body = {"model": model_id, **api_params}
                try:
                    response = await self._request_with_retry(
                        "POST", "/api/v1/models/load", json_data=body,
                        timeout_s=LOAD_TIMEOUT_S,
                    )
                except httpx.HTTPStatusError as e2:
                    return LoadModelResponse(success=False, error=_error_message(e2))
            else:
                return LoadModelResponse(success=False, error=_error_message(e))

        result_data = response.json()
        identifier = result_data.get("instance_id")
        if result_data.get("status") == "loaded" and identifier:
            self._loaded_models[model_id] = identifier
            loaded_config = self._config_from_echo(result_data.get("load_config"))
            if loaded_config is None:
                loaded_config = await self._read_echo_config(identifier)
            mismatches = self._compare_echo(api_params, loaded_config)
            if mismatches:
                logger.warning(
                    "Requested vs applied mismatch", model=model_id, mismatches=mismatches
                )
            return LoadModelResponse(
                success=True,
                identifier=identifier,
                loaded_config=loaded_config,
                echo_mismatches=mismatches or None,
            )
        return LoadModelResponse(
            success=False,
            error=result_data.get("error") or f"Unexpected response: {result_data}",
        )

    @staticmethod
    def _config_from_echo(echo: dict | None) -> LoadConfig | None:
        """Build LoadConfig from an echo dict."""
        if not isinstance(echo, dict) or not echo:
            return None
        return LoadConfig(**{k: v for k, v in echo.items() if k in LoadConfig.model_fields})

    @staticmethod
    def _compare_echo(requested: dict, applied: LoadConfig | None) -> dict:
        """REQUESTED vs APPLIED diff on keys we sent (empty = exact match)."""
        if applied is None:
            return {}
        actual = applied.model_dump(exclude_none=True)
        return {
            k: {"requested": v, "applied": actual.get(k)}
            for k, v in requested.items()
            if actual.get(k) != v
        }

    async def unload_model(
        self, model_id: str | None = None, identifier: str | None = None
    ) -> UnloadModelResponse:
        """Unload a model by instance id."""
        if identifier is None and model_id is not None:
            identifier = self._loaded_models.get(model_id)

        if identifier is None:
            return UnloadModelResponse(
                success=False, error="No identifier provided or model not loaded"
            )

        try:
            await self._request_with_retry(
                "POST", "/api/v1/models/unload", json_data={"instance_id": identifier}
            )
        except httpx.HTTPStatusError as e:
            return UnloadModelResponse(success=False, error=_error_message(e))
        except Exception as e:
            return UnloadModelResponse(success=False, error=str(e))

        if model_id and model_id in self._loaded_models:
            del self._loaded_models[model_id]
        else:
            for mid, iid in list(self._loaded_models.items()):
                if iid == identifier:
                    del self._loaded_models[mid]
        return UnloadModelResponse(success=True)

    async def unload_all(self) -> dict[str, UnloadModelResponse]:
        """Unload all currently loaded models (server state wins)."""
        results: dict[str, UnloadModelResponse] = {}
        try:
            instances = await self.get_loaded_instances()
        except Exception:
            instances = []
        for inst in instances:
            if inst["instance_id"]:
                results[inst["model"]] = await self.unload_model(identifier=inst["instance_id"])
        for model_id, identifier in list(self._loaded_models.items()):
            if model_id not in results:
                results[model_id] = await self.unload_model(identifier=identifier)
        return results

    async def ensure_unloaded(self, model_id: str) -> bool:
        """Unload a model and verify it is really gone (server state).

        Retries once by instance id: covers stale client cache and transient
        unload failures. Returns True only when no instance of the model
        remains loaded.
        """
        for _ in range(2):
            try:
                await self.unload_model(model_id=model_id)
            except Exception as e:
                logger.warning("Unload attempt failed", model=model_id, error=str(e))
            try:
                instances = await self.get_loaded_instances()
            except Exception as e:
                logger.warning("State check failed", model=model_id, error=str(e))
                continue
            leftovers = [i for i in instances if i["model"] == model_id and i["instance_id"]]
            if not leftovers:
                self._loaded_models.pop(model_id, None)
                return True
            for inst in leftovers:
                try:
                    await self.unload_model(identifier=inst["instance_id"])
                except Exception as e:
                    logger.warning(
                        "Forced unload failed",
                        instance=inst["instance_id"],
                        error=str(e),
                    )
        try:
            instances = await self.get_loaded_instances()
        except Exception:
            return False
        left = [i for i in instances if i["model"] == model_id]
        if left:
            logger.warning("Model still loaded after ensure_unloaded", model=model_id)
            return False
        self._loaded_models.pop(model_id, None)
        return True

    async def chat_completion(
        self,
        model: str,
        input_text: str,
        temperature: float = 0.7,
        max_output_tokens: int = 512,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        reasoning: str | None = None,
    ) -> ChatCompletionResponse:
        """Generate completion via the native /api/v1/chat endpoint.

        Non-reasoning models reject the reasoning key: retry once without it
        and remember the verdict per model.
        """
        use_reasoning = reasoning if model not in self._no_reasoning_models else None
        body: dict[str, Any] = {
            "model": model,
            "input": input_text,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        for key, value in (
            ("top_p", top_p),
            ("top_k", top_k),
            ("min_p", min_p),
            ("presence_penalty", presence_penalty),
            ("reasoning", use_reasoning),
        ):
            if value is not None:
                body[key] = value

        try:
            response = await self._request_with_retry(
                "POST",
                "/api/v1/chat",
                json_data=body,
                timeout_s=max(CHAT_TIMEOUT_FLOOR_S, float(max_output_tokens)),
            )
        except httpx.HTTPStatusError as e:
            if use_reasoning is not None and _is_reasoning_error(e):
                if _is_no_reasoning_error(e):
                    logger.info("Model has no reasoning config; retrying without", model=model)
                    self._no_reasoning_models.add(model)
                else:
                    logger.info("Reasoning value rejected; retrying without", model=model)
                body.pop("reasoning", None)
                response = await self._request_with_retry(
                    "POST",
                    "/api/v1/chat",
                    json_data=body,
                    timeout_s=max(CHAT_TIMEOUT_FLOOR_S, float(max_output_tokens)),
                )
            else:
                raise
        data = response.json()

        parts = data.get("output") or []
        texts = [p.get("content", "") for p in parts if p.get("type") == "message"]
        if not texts:
            texts = [p.get("content", "") for p in parts if p.get("content")]
        text = "\n".join(texts)

        stats = data.get("stats") or {}
        prompt_tokens = int(stats.get("input_tokens", 0))
        completion_tokens = int(stats.get("total_output_tokens", 0))
        return ChatCompletionResponse(
            id=data.get("response_id", f"resp_{uuid.uuid4().hex}"),
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            stats=stats,
        )

    async def get_embeddings(self, model: str, input_text: str | list[str]) -> EmbeddingResponse:
        """Get embeddings for input text."""
        request = EmbeddingRequest(model=model, input=input_text)
        response = await self._request_with_retry(
            "POST", "/api/v1/embeddings", json_data=request.model_dump()
        )
        return EmbeddingResponse(**response.json())

    async def health_check(self) -> bool:
        """Check if LM Studio is responsive."""
        try:
            await self._request_with_retry("GET", "/api/v1/models")
            return True
        except Exception:
            return False

    def model_known_no_reasoning(self, model_id: str) -> bool:
        """True when this model already rejected the reasoning key."""
        return model_id in self._no_reasoning_models


async def create_client(base_url: str | None = None) -> LMStudioClient:
    """Factory function to create and connect a client."""
    client = LMStudioClient(base_url=base_url)
    await client.connect()
    return client
