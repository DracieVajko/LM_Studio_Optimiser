"""LM Studio service with capability discovery."""

import uuid

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lm_optimizer.config import config
from lm_optimizer.domain.models import (
    LMStudioCapabilities,
    LoadConfiguration,
    ModelIdentity,
)
from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)


class LMStudioClient:
    """LM Studio API client with capability discovery."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or config.lm_studio.base_url).rstrip("/")
        self.timeout = timeout or config.lm_studio.timeout
        self._client: httpx.AsyncClient | None = None
        self._capabilities: LMStudioCapabilities | None = None
        self._models_cache: list[ModelIdentity] = []
        self._loaded_models: dict[str, str] = {}

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
        """Get the HTTP client."""
        if self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._client

    @property
    def capabilities(self) -> LMStudioCapabilities:
        """Get detected API capabilities."""
        if self._capabilities is None:
            raise RuntimeError("Capabilities not detected. Call connect() first.")
        return self._capabilities

    async def _detect_capabilities(self) -> None:
        """Detect LM Studio API version and capabilities."""
        try:
            response = await self._request_with_retry("GET", "/api/v1/models")
            models_data = response.json()

            self._capabilities = LMStudioCapabilities()

            # Detect version from response structure
            if isinstance(models_data, dict) and "data" in models_data:
                self._capabilities.version = "v1"
            elif isinstance(models_data, list):
                self._capabilities.version = "v0"

            # Probe supported parameters
            await self._probe_load_parameters()

        except Exception as e:
            logger.warning("Failed to detect capabilities, using defaults", error=str(e))
            self._capabilities = LMStudioCapabilities(version="v1")

    async def _probe_load_parameters(self) -> None:
        """Probe which load parameters are supported."""
        if not self._capabilities:
            return

        try:
            models = await self.list_models()
            if models:
                test_model = models[0].id
                test_config = LoadConfiguration(context_length=512)
                result = await self.load_model(test_model, test_config)
                if result.success and result.loaded_config:
                    accepted = set(result.loaded_config.to_dict().keys())
                    self._capabilities.load_parameters = list(accepted)
                    # Update capability flags based on accepted params
                    self._capabilities.supports_context_length = "context_length" in accepted
                    self._capabilities.supports_gpu_ratio = "gpu_ratio" in accepted
                    self._capabilities.supports_flash_attention = "flash_attention" in accepted
                    self._capabilities.supports_kv_cache_placement = (
                        "offload_kv_cache_to_gpu" in accepted
                    )
                    self._capabilities.supports_eval_batch_size = "eval_batch_size" in accepted
                    self._capabilities.supports_num_experts = "num_experts" in accepted
                    self._capabilities.supports_rope_scaling = "rope_freq_base" in accepted
        except Exception:
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

    async def list_models(self, force_refresh: bool = False) -> list[ModelIdentity]:
        """List all available models."""
        if self._models_cache and not force_refresh:
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
            model = ModelIdentity(
                id=item.get("id", ""),
                name=item.get("name", item.get("id", "")),
                architecture=item.get("architecture"),
                parameter_count=item.get("parameter_count"),
                quantization=item.get("quantization"),
                context_limit=item.get("max_context_length") or item.get("context_length"),
                is_moe=self._detect_moe(item),
                num_experts=item.get("num_experts"),
                size_bytes=item.get("size_bytes"),
            )
            models.append(model)

        self._models_cache = models
        return models

    def _detect_moe(self, item: dict) -> bool:
        """Detect if model is Mixture of Experts."""
        arch = (item.get("architecture") or "").lower()
        model_type = (item.get("type") or "").lower()
        name = (item.get("name") or "").lower()

        moe_indicators = ["moe", "mixtral", "grok", "deepseek-moe", "qwen-moe", "phi-moe"]
        return any(
            indicator in arch or indicator in model_type or indicator in name
            for indicator in moe_indicators
        )

    async def get_model(self, model_id: str) -> ModelIdentity | None:
        """Get specific model information."""
        models = await self.list_models()
        for model in models:
            if model.id == model_id:
                return model
        return None

    async def load_model(
        self,
        model_id: str,
        load_config: LoadConfiguration,
    ) -> "LoadModelResult":
        """Load a model with specified configuration."""
        identifier = str(uuid.uuid4())[:8]

        # Filter config to only supported parameters
        supported = set(self._capabilities.get_supported_load_params())
        filtered_config = LoadConfiguration(
            **{k: v for k, v in load_config.to_dict().items() if k in supported and v is not None}
        )

        request = {
            "model": model_id,
            "config": filtered_config.to_api_params(),
            "identifier": identifier,
        }

        response = await self._request_with_retry("POST", "/api/v1/models/load", json_data=request)
        result_data = response.json()

        from dataclasses import dataclass

        @dataclass
        class LoadModelResult:
            success: bool
            identifier: str | None = None
            error: str | None = None
            loaded_config: LoadConfiguration | None = None

        result = LoadModelResult(
            success=result_data.get("success", False),
            identifier=result_data.get("identifier"),
            error=result_data.get("error"),
        )

        if result.success and result.identifier:
            self._loaded_models[model_id] = result.identifier
            # Parse loaded config if returned
            if "config" in result_data:
                result.loaded_config = LoadConfiguration(**result_data["config"])

        return result

    async def unload_model(
        self, model_id: str | None = None, identifier: str | None = None
    ) -> bool:
        """Unload a model."""
        if identifier is None and model_id is not None:
            identifier = self._loaded_models.get(model_id)

        if identifier is None:
            return False

        request = {"identifier": identifier}
        try:
            response = await self._request_with_retry(
                "POST", "/api/v1/models/unload", json_data=request
            )
            result = response.json()
            if result.get("success") and model_id and model_id in self._loaded_models:
                del self._loaded_models[model_id]
            return result.get("success", False)
        except Exception:
            return False

    async def unload_all(self) -> dict[str, bool]:
        """Unload all currently loaded models."""
        results = {}
        for model_id, identifier in list(self._loaded_models.items()):
            results[model_id] = await self.unload_model(identifier=identifier)
        return results

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
        seed: int | None = None,
        stop: list[str] | None = None,
    ) -> dict:
        """Generate chat completion."""
        request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "seed": seed,
            "stop": stop,
        }

        response = await self._request_with_retry(
            "POST", "/api/v1/chat/completions", json_data=request
        )
        return response.json()

    async def health_check(self) -> bool:
        """Check if LM Studio is responsive."""
        try:
            await self._request_with_retry("GET", "/api/v1/models")
            return True
        except Exception:
            return False

    def get_loaded_model(self, model_id: str) -> str | None:
        """Get loaded model identifier."""
        return self._loaded_models.get(model_id)


async def create_client(base_url: str | None = None) -> LMStudioClient:
    """Factory function to create and connect a client."""
    client = LMStudioClient(base_url=base_url)
    await client.connect()
    return client
