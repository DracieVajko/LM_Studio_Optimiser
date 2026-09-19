"""LM Studio service with capability discovery.

This is a wrapper around the authoritative API client in lm_optimizer.api.client
to maintain compatibility with existing service interfaces.
"""

from dataclasses import dataclass

from lm_optimizer.api.client import (
    LMStudioClient as _LMStudioClient,
    LoadModelResponse as _LoadModelResponse,
    ChatCompletionResponse as _ChatCompletionResponse,
    LoadConfig,
    GenerationParameters as APIGenParams,
    ChatMessage,
)
from lm_optimizer.config import config
from lm_optimizer.domain.models import (
    LMStudioCapabilities,
    LoadConfiguration,
    ModelIdentity,
    GenerationParameters,
)
from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class LoadModelResult:
    """Result of loading a model (wrapper for compatibility)."""
    success: bool
    identifier: str | None = None
    error: str | None = None
    loaded_config: LoadConfiguration | None = None


class LMStudioClient:
    """LM Studio API client with capability discovery (compatibility wrapper)."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self._inner = _LMStudioClient(base_url=base_url, timeout=timeout)

    async def __aenter__(self) -> "LMStudioClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def connect(self) -> None:
        """Establish connection and detect capabilities."""
        await self._inner.connect()

    async def close(self) -> None:
        """Close the client connection."""
        await self._inner.close()

    @property
    def client(self):
        """Get the inner HTTP client."""
        return self._inner.client

    @property
    def capabilities(self) -> LMStudioCapabilities:
        """Get detected API capabilities."""
        caps = self._inner.capabilities
        # Convert APICapabilities to LMStudioCapabilities
        return LMStudioCapabilities(
            version=caps.version,
            supports_context_length=caps.supports_context_length,
            supports_gpu_ratio=caps.supports_gpu_ratio,
            supports_flash_attention=caps.supports_flash_attention,
            supports_kv_cache_placement=caps.supports_kv_cache_offload,
            supports_eval_batch_size=caps.supports_eval_batch_size,
            supports_num_experts=caps.supports_num_experts,
            supports_rope_scaling=caps.supports_rope_scaling,
            load_parameters=caps.load_parameters,
        )

    async def list_models(self, force_refresh: bool = False) -> list[ModelIdentity]:
        """List all available models."""
        models = await self._inner.list_models(force_refresh=force_refresh)
        return [
            ModelIdentity(
                id=m.id,
                name=m.name,
                architecture=m.architecture,
                parameter_count=m.parameter_count,
                quantization=m.quantization,
                context_limit=m.context_length or m.max_context_length,
                is_moe=False,  # Not provided by API
                num_experts=None,
                size_bytes=m.size_bytes,
            )
            for m in models
        ]

    async def get_model(self, model_id: str) -> ModelIdentity | None:
        """Get specific model information."""
        model = await self._inner.get_model(model_id)
        if not model:
            return None
        return ModelIdentity(
            id=model.id,
            name=model.name,
            architecture=model.architecture,
            parameter_count=model.parameter_count,
            quantization=model.quantization,
            context_limit=model.context_length or model.max_context_length,
            is_moe=False,
            num_experts=None,
            size_bytes=model.size_bytes,
        )

    async def load_model(
        self,
        model_id: str,
        load_config: LoadConfiguration,
    ) -> LoadModelResult:
        """Load a model with specified configuration."""
        # Convert LoadConfiguration to LoadConfig
        api_load_config = LoadConfig(
            context_length=load_config.context_length,
            gpu_ratio=load_config.gpu_ratio,
            flash_attention=load_config.flash_attention,
            offload_kv_cache_to_gpu=load_config.offload_kv_cache_to_gpu,
            eval_batch_size=load_config.eval_batch_size,
            num_experts=load_config.num_experts,
            rope_freq_base=load_config.rope_freq_base,
            rope_freq_scale=load_config.rope_freq_scale,
        )

        result = await self._inner.load_model(model_id, api_load_config)
        
        loaded_config = None
        if result.loaded_config:
            loaded_config = LoadConfiguration(
                context_length=result.loaded_config.context_length,
                gpu_ratio=result.loaded_config.gpu_ratio,
                flash_attention=result.loaded_config.flash_attention,
                offload_kv_cache_to_gpu=result.loaded_config.offload_kv_cache_to_gpu,
                eval_batch_size=result.loaded_config.eval_batch_size,
                num_experts=result.loaded_config.num_experts,
                rope_freq_base=result.loaded_config.rope_freq_base,
                rope_freq_scale=result.loaded_config.rope_freq_scale,
            )
        return LoadModelResult(
            success=result.success,
            identifier=result.identifier,
            error=result.error,
            loaded_config=loaded_config,
        )

    async def unload_model(
        self, model_id: str | None = None, identifier: str | None = None
    ) -> bool:
        """Unload a model."""
        return await self._inner.unload_model(model_id=model_id, identifier=identifier)

    async def unload_all(self) -> dict[str, bool]:
        """Unload all currently loaded models."""
        return await self._inner.unload_all()

    def get_loaded_model(self, model_id: str) -> str | None:
        """Get loaded model identifier."""
        return self._inner.get_loaded_model(model_id)

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
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
        reasoning: str | None = None,
        max_output_tokens: int | None = None,
        generation: GenerationParameters | None = None,
    ) -> dict:
        """Generate chat completion."""
        gen_params = generation or GenerationParameters(
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
            seed=seed,
            stop_sequences=stop,
            reasoning=reasoning,
            max_output_tokens=max_output_tokens,
        )
        
        # Convert messages format
        from lm_optimizer.api.client import ChatMessage
        chat_messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]
        
        response = await self._inner.chat_completion(
            model=model,
            messages=chat_messages,
            generation=gen_params,
            stream=stream,
        )
        
        # Convert to dict format expected by callers
        return {
            "choices": [
                {
                    "message": {"content": c.message.content},
                    "finish_reason": c.finish_reason,
                    "index": c.index,
                }
                for c in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "id": response.id,
            "model": response.model,
        }

    async def health_check(self) -> bool:
        """Check if LM Studio is responsive."""
        return await self._inner.health_check()


async def create_client(base_url: str | None = None) -> LMStudioClient:
    """Factory function to create and connect a client."""
    client = LMStudioClient(base_url=base_url)
    await client.connect()
    return client