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


class GenerationParameters(BaseModel):
    """Generation/inference parameters."""

    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling top-p")
    top_k: int | None = Field(default=None, ge=1, description="Top-k sampling")
    repetition_penalty: float | None = Field(default=None, ge=0.0, le=2.0, description="Repetition penalty")
    min_p: float | None = Field(default=None, ge=0.0, le=1.0, description="Min-p sampling")
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0, description="Frequency penalty")
    typical_p: float | None = Field(default=None, ge=0.0, le=1.0, description="Typical-p sampling")
    mirostat_mode: int | None = Field(default=None, ge=0, le=2, description="Mirostat mode")
    mirostat_tau: float | None = Field(default=None, ge=0.0, le=10.0, description="Mirostat tau")
    mirostat_eta: float | None = Field(default=None, ge=0.0, le=1.0, description="Mirostat eta")
    seed: int | None = Field(default=None, description="Random seed")
    stop_sequences: list[str] | None = Field(default=None, description="Stop sequences")

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None and v != []}

    def to_api_params(self) -> dict[str, Any]:
        return self.to_dict()


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
    # Generation parameters
    generation: GenerationParameters | None = Field(default=None, description="Generation parameters")

    def to_api_params(self) -> dict[str, Any]:
        """Convert to API parameters, excluding None values."""
        params = {k: v for k, v in self.model_dump(exclude={"generation"}).items() if v is not None}
        if self.generation:
            gen_params = self.generation.to_api_params()
            if gen_params:
                params["generation"] = gen_params
        return params


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
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    repetition_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    typical_p: float | None = Field(default=None, ge=0.0, le=1.0)
    mirostat_mode: int | None = Field(default=None, ge=0, le=2)
    mirostat_tau: float | None = Field(default=None, ge=0.0, le=10.0)
    mirostat_eta: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1)
    stream: bool = False
    seed: int | None = None
    stop: list[str] | None = None

    def to_api_params(self) -> dict[str, Any]:
        """Convert to API parameters, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None and v != []}


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
    """Async client for LM Studio API.

    Supports both native LM Studio API (/api/v1/...) and
    OpenAI-compatible API (/v1/...).
    """

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        raw_url = (base_url or config.lm_studio.base_url).rstrip("/")
        # If URL ends with /v1, strip it and treat as OpenAI-compatible hint
        if raw_url.endswith("/v1"):
            self.base_url = raw_url[:-3]  # Remove /v1
            self._prefer_openai_compatible = True
        else:
            self.base_url = raw_url
            self._prefer_openai_compatible = False
        self.timeout = timeout or config.lm_studio.timeout
        self._client: httpx.AsyncClient | None = None
        self._capabilities: APICapabilities | None = None
        self._models_cache: list[ModelInfo] | None = None
        self._loaded_models: dict[str, str] = {}  # model_id -> identifier
        self._api_base_path: str = "/api/v1"  # Default to native API
        self._is_openai_compatible: bool = False

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

        # Test connection and detect API version/type
        await self._detect_api_type()
        await self._detect_capabilities()
        logger.info(
            "Connected to LM Studio",
            base_url=self.base_url,
            api_type="openai_compatible" if self._is_openai_compatible else "native",
            version=self._capabilities.version,
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

    @property
    def api_base_path(self) -> str:
        """Get the detected API base path."""
        return self._api_base_path

    @property
    def is_openai_compatible(self) -> bool:
        """Check if using OpenAI-compatible API."""
        return self._is_openai_compatible

    async def _detect_api_type(self) -> None:
        """Detect whether the endpoint is native LM Studio API or OpenAI-compatible."""
        # If user provided URL ending with /v1, prefer OpenAI-compatible API
        if self._prefer_openai_compatible:
            try:
                response = await self._request_with_retry("GET", "/v1/models")
                data = response.json()
                if isinstance(data, dict) and "data" in data:
                    self._api_base_path = "/v1"
                    self._is_openai_compatible = True
                    return
            except Exception:
                pass

        # Try native LM Studio API first
        try:
            response = await self._request_with_retry("GET", "/api/v1/models")
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                self._api_base_path = "/api/v1"
                self._is_openai_compatible = False
                return
            elif isinstance(data, list):
                self._api_base_path = "/api/v1"
                self._is_openai_compatible = False
                return
        except Exception:
            pass

        # Try OpenAI-compatible API
        try:
            response = await self._request_with_retry("GET", "/v1/models")
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                self._api_base_path = "/v1"
                self._is_openai_compatible = True
                return
        except Exception:
            pass

        # Default to native API
        self._api_base_path = "/api/v1"
        self._is_openai_compatible = False

    async def _detect_capabilities(self) -> None:
        """Detect LM Studio API version and capabilities."""
        try:
            # Use detected API base path
            models_path = f"{self._api_base_path}/models"
            response = await self._request_with_retry("GET", models_path)
            models_data = response.json()

            # Detect version from response structure
            self._capabilities = APICapabilities()

            # Check for v1 API indicators
            if isinstance(models_data, dict) and "data" in models_data:
                self._capabilities.version = "v1"
            elif isinstance(models_data, list):
                self._capabilities.version = "v0"

            # For OpenAI-compatible API, we can't probe load parameters the same way
            if not self._is_openai_compatible:
                await self._probe_load_parameters()
            else:
                # OpenAI-compatible API doesn't support model load/unload via API
                # Set defaults for native API parameters
                self._capabilities.supports_context_length = False
                self._capabilities.supports_gpu_ratio = False
                self._capabilities.supports_flash_attention = False
                self._capabilities.supports_kv_cache_offload = False
                self._capabilities.supports_eval_batch_size = False
                self._capabilities.supports_num_experts = False
                self._capabilities.supports_rope_scaling = False
                self._capabilities.load_parameters = []

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

    def _api_path(self, endpoint: str) -> str:
        """Build full API path using detected base path."""
        return f"{self._api_base_path}{endpoint}"

    async def list_models(self, force_refresh: bool = False) -> list[ModelInfo]:
        """List all available models."""
        if self._models_cache is not None and not force_refresh:
            return self._models_cache

        response = await self._request_with_retry("GET", self._api_path("/models"))
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
        """Load a model with specified configuration (native API only)."""
        if self._is_openai_compatible:
            return LoadModelResponse(
                success=False,
                error="Model load/unload not supported via OpenAI-compatible API. Use native LM Studio API."
            )

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
            "POST", self._api_path("/models/load"), json_data=request.model_dump()
        )
        result = LoadModelResponse(**response.json())

        if result.success and result.identifier:
            self._loaded_models[model_id] = result.identifier

        return result

    async def unload_model(
        self, model_id: str | None = None, identifier: str | None = None
    ) -> UnloadModelResponse:
        """Unload a model (native API only)."""
        if self._is_openai_compatible:
            return UnloadModelResponse(
                success=False,
                error="Model load/unload not supported via OpenAI-compatible API. Use native LM Studio API."
            )

        if identifier is None and model_id is not None:
            identifier = self._loaded_models.get(model_id)

        if identifier is None:
            return UnloadModelResponse(
                success=False, error="No identifier provided or model not loaded"
            )

        request = UnloadModelRequest(identifier=identifier)
        response = await self._request_with_retry(
            "POST", self._api_path("/models/unload"), json_data=request.model_dump()
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
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        typical_p: float | None = None,
        mirostat_mode: int | None = None,
        mirostat_tau: float | None = None,
        mirostat_eta: float | None = None,
    ) -> ChatCompletionResponse:
        """Generate chat completion with full generation parameters."""
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            typical_p=typical_p,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            max_tokens=max_tokens,
            stream=stream,
            seed=seed,
            stop=stop,
        )

        response = await self._request_with_retry(
            "POST", self._api_path("/chat/completions"), json_data=request.to_api_params()
        )
        return ChatCompletionResponse(**response.json())

    async def get_embeddings(self, model: str, input_text: str | list[str]) -> EmbeddingResponse:
        """Get embeddings for input text."""
        request = EmbeddingRequest(model=model, input=input_text)
        response = await self._request_with_retry(
            "POST", self._api_path("/embeddings"), json_data=request.model_dump()
        )
        return EmbeddingResponse(**response.json())

    async def health_check(self) -> bool:
        """Check if LM Studio is responsive."""
        try:
            await self._request_with_retry("GET", self._api_path("/models"))
            return True
        except Exception:
            return False


async def create_client(base_url: str | None = None) -> LMStudioClient:
    """Factory function to create and connect a client."""
    client = LMStudioClient(base_url=base_url)
    await client.connect()
    return client
