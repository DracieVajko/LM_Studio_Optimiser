"""LM Studio API client with async support."""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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


class ModelStatus(str, Enum):
    """Model load status."""

    LOADED = "loaded"
    UNLOADED = "unloaded"
    LOADING = "loading"
    ERROR = "error"


class LoadConfig(BaseModel):
    """Model load configuration parameters."""

    context_length: int | None = Field(default=None, description="Context window size")
    gpu_ratio: float | None = Field(
        default=None, ge=0.0, le=1.0, description="GPU offload ratio (0-1)"
    )
    flash_attention: bool | None = Field(default=None, description="Enable Flash Attention")
    offload_kv_cache_to_gpu: bool | None = Field(
        default=None, description="Offload KV cache to GPU"
    )
    eval_batch_size: int | None = Field(default=None, ge=1, description="Evaluation batch size")
    num_experts: int | None = Field(
        default=None, ge=1, description="Number of experts for MoE models"
    )
    rope_freq_base: float | None = Field(default=None, description="RoPE frequency base")
    rope_freq_scale: float | None = Field(default=None, description="RoPE frequency scale")

    def to_api_params(self) -> dict[str, Any]:
        """Convert to API parameters, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


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


class LoadModelRequest(BaseModel):
    """Request to load a model."""

    model: str
    config: LoadConfig | None = None
    identifier: str | None = None


class LoadModelResponse(BaseModel):
    """Response from loading a model."""

    success: bool
    identifier: str | None = None
    error: str | None = None
    load_config: LoadConfig | None = None


class UnloadModelRequest(BaseModel):
    """Request to unload a model."""

    identifier: str


class UnloadModelResponse(BaseModel):
    """Response from unloading a model."""

    success: bool
    error: str | None = None


class ChatMessage(BaseModel):
    """Chat message."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""

    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False
    seed: int | None = None
    stop: list[str] | None = None


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
    """Chat completion response."""

    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
    system_fingerprint: str | None = None


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


@dataclass
class APICapabilities:
    """Detected API capabilities."""

    version: str = "unknown"
    supports_context_length: bool = True
    supports_gpu_ratio: bool = True
    supports_flash_attention: bool = False
    supports_kv_cache_offload: bool = False
    supports_eval_batch_size: bool = True
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
        if self.supports_num_experts:
            params.append("num_experts")
        if self.supports_rope_scaling:
            params.extend(["rope_freq_base", "rope_freq_scale"])
        return params


class LMStudioClient:
    """Async client for LM Studio API."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or config.lm_studio.base_url).rstrip("/")
        self.timeout = timeout or config.lm_studio.timeout
        self._client: httpx.AsyncClient | None = None
        self._capabilities: APICapabilities | None = None
        self._models_cache: list[ModelInfo] | None = None
        self._loaded_models: dict[str, str] = {}  # model_id -> identifier

    async def __aenter__(self) -> "LMStudioClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def connect(self) -> None:
        """Establish connection and detect capabilities."""
        if self._client is not None:
            return

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        # Test connection and detect API version
        await self._detect_capabilities()
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
            # Try to get models to verify connection
            response = await self._request_with_retry("GET", "/api/v1/models")
            models_data = response.json()

            # Detect version from response structure
            self._capabilities = APICapabilities()

            # Check for v1 API indicators
            if isinstance(models_data, dict) and "data" in models_data:
                self._capabilities.version = "v1"
            elif isinstance(models_data, list):
                self._capabilities.version = "v0"

            # Try to detect supported parameters by checking model schema or docs endpoint
            await self._probe_load_parameters()

        except Exception as e:
            logger.warning("Failed to detect capabilities, using defaults", error=str(e))
            self._capabilities = APICapabilities(version="v1")

    async def _probe_load_parameters(self) -> None:
        """Probe which load parameters are supported."""
        if not self._capabilities:
            return

        # Try to load a model with various parameters to see what's accepted
        # This is a heuristic - we test with a minimal model if available
        try:
            models = await self.list_models()
            if models:
                # Use first model for probing
                test_model = models[0].id
                # Try loading with minimal config to see response
                test_config = LoadConfig(context_length=512)
                result = await self.load_model(test_model, test_config)
                if result.success and result.load_config:
                    # Check which parameters were accepted
                    accepted = set(result.load_config.model_dump(exclude_none=True).keys())
                    self._capabilities.load_parameters = list(accepted)
        except Exception:
            # Probing failed, use defaults
            pass

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        json_data: dict | None = None,
        params: dict | None = None,
    ) -> httpx.Response:
        """Make HTTP request with retry logic."""
        retry_config = AsyncRetrying(
            stop=stop_after_attempt(config.lm_studio.max_retries),
            wait=wait_exponential(multiplier=config.lm_studio.retry_delay, min=1, max=10),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)
            ),
            reraise=True,
        )

        async for attempt in retry_config:
            with attempt:
                response = await self.client.request(method, path, json=json_data, params=params)
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
        if isinstance(data, dict) and "data" in data:
            items = data["data"]
        elif isinstance(data, list):
            items = data
        else:
            items = []

        for item in items:
            model = ModelInfo(
                id=item.get("id", ""),
                name=item.get("name", item.get("id", "")),
                description=item.get("description"),
                architecture=item.get("architecture"),
                context_length=item.get("context_length"),
                max_context_length=item.get("max_context_length"),
                model_type=item.get("type") or item.get("model_type"),
                quantization=item.get("quantization"),
                size_bytes=item.get("size_bytes"),
                parameter_count=item.get("parameter_count"),
                loaded=item.get("loaded", False),
            )
            models.append(model)

        self._models_cache = models
        return models

    async def get_model(self, model_id: str) -> ModelInfo | None:
        """Get specific model information."""
        models = await self.list_models()
        for model in models:
            if model.id == model_id:
                return model
        return None

    async def load_model(
        self, model_id: str, load_config: LoadConfig | None = None
    ) -> LoadModelResponse:
        """Load a model with specified configuration."""
        identifier = str(uuid.uuid4())[:8]
        request = LoadModelRequest(model=model_id, config=load_config, identifier=identifier)

        # Filter config to only supported parameters
        if load_config and self._capabilities:
            supported = set(self._capabilities.get_supported_load_params())
            filtered_config = LoadConfig(
                **{
                    k: v
                    for k, v in load_config.model_dump().items()
                    if k in supported and v is not None
                }
            )
            request.config = filtered_config

        response = await self._request_with_retry(
            "POST", "/api/v1/models/load", json_data=request.model_dump()
        )
        result = LoadModelResponse(**response.json())

        if result.success and result.identifier:
            self._loaded_models[model_id] = result.identifier

        return result

    async def unload_model(
        self, model_id: str | None = None, identifier: str | None = None
    ) -> UnloadModelResponse:
        """Unload a model."""
        if identifier is None and model_id is not None:
            identifier = self._loaded_models.get(model_id)

        if identifier is None:
            return UnloadModelResponse(
                success=False, error="No identifier provided or model not loaded"
            )

        request = UnloadModelRequest(identifier=identifier)
        response = await self._request_with_retry(
            "POST", "/api/v1/models/unload", json_data=request.model_dump()
        )
        result = UnloadModelResponse(**response.json())

        if result.success and model_id and model_id in self._loaded_models:
            del self._loaded_models[model_id]

        return result

    async def unload_all(self) -> dict[str, UnloadModelResponse]:
        """Unload all currently loaded models."""
        results = {}
        for model_id, identifier in list(self._loaded_models.items()):
            results[model_id] = await self.unload_model(identifier=identifier)
        return results

    async def chat_completion(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
        seed: int | None = None,
        stop: list[str] | None = None,
    ) -> ChatCompletionResponse:
        """Generate chat completion."""
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            seed=seed,
            stop=stop,
        )

        response = await self._request_with_retry(
            "POST", "/api/v1/chat/completions", json_data=request.model_dump()
        )
        return ChatCompletionResponse(**response.json())

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


async def create_client(base_url: str | None = None) -> LMStudioClient:
    """Factory function to create and connect a client."""
    client = LMStudioClient(base_url=base_url)
    await client.connect()
    return client
