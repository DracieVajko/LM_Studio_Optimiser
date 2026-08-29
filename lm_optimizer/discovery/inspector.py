"""Model discovery and capability inspection."""

from dataclasses import dataclass

from lm_optimizer.api.client import LMStudioClient, LoadConfig, ModelInfo
from lm_optimizer.config import config
from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ModelCapabilities:
    """Detailed model capabilities."""

    model_id: str
    architecture: str | None = None
    context_length: int | None = None
    max_context_length: int | None = None
    parameter_count: int | None = None
    quantization: str | None = None
    is_moe: bool = False
    num_experts: int | None = None
    num_experts_per_token: int | None = None
    supported_parameters: list[str] = None
    estimated_vram_gb: float | None = None

    def __post_init__(self):
        if self.supported_parameters is None:
            self.supported_parameters = []


class ModelDiscovery:
    """Discover and inspect models in LM Studio."""

    def __init__(self, client: LMStudioClient):
        self.client = client

    async def discover_all(self) -> list[ModelInfo]:
        """Discover all available models."""
        return await self.client.list_models(force_refresh=True)

    async def inspect_model(self, model_id: str) -> ModelCapabilities:
        """Inspect a model's capabilities in detail."""
        model_info = await self.client.get_model(model_id)
        if not model_info:
            raise ValueError(f"Model not found: {model_id}")

        capabilities = ModelCapabilities(model_id=model_id)

        # Basic info from model listing
        capabilities.architecture = model_info.architecture
        capabilities.context_length = model_info.context_length
        capabilities.max_context_length = model_info.max_context_length or model_info.context_length
        capabilities.parameter_count = model_info.parameter_count
        capabilities.quantization = model_info.quantization

        # Detect MoE
        capabilities.is_moe = self._detect_moe(model_info)
        if capabilities.is_moe:
            capabilities.num_experts = self._estimate_experts(model_info)

        # Estimate VRAM requirements
        capabilities.estimated_vram_gb = self._estimate_vram(model_info)

        # Get supported parameters from API capabilities
        capabilities.supported_parameters = self.client.capabilities.get_supported_load_params()

        # Try to load model with minimal config to inspect actual capabilities
        try:
            await self._probe_model_capabilities(model_id, capabilities)
        except Exception as e:
            logger.warning("Failed to probe model capabilities", model=model_id, error=str(e))

        return capabilities

    def _detect_moe(self, model_info: ModelInfo) -> bool:
        """Detect if model is Mixture of Experts."""
        arch = (model_info.architecture or "").lower()
        model_type = (model_info.model_type or "").lower()
        name = (model_info.name or "").lower()

        moe_indicators = ["moe", "mixtral", "grok", "deepseek-moe", "qwen-moe", "phi-moe"]
        return any(
            indicator in arch or indicator in model_type or indicator in name
            for indicator in moe_indicators
        )

    def _estimate_experts(self, model_info: ModelInfo) -> int | None:
        """Estimate number of experts for MoE models."""
        name = (model_info.name or "").lower()

        # Known MoE configurations
        expert_map = {
            "mixtral-8x7b": 8,
            "mixtral-8x22b": 8,
            "deepseek-moe-16b": 64,
            "grok-1": 8,
            "qwen-moe": 64,
        }

        for key, experts in expert_map.items():
            if key in name:
                return experts

        # Try to infer from parameter count
        if model_info.parameter_count:
            # Rough heuristic: MoE models typically have 8 experts for 7B-scale
            if model_info.parameter_count < 10_000_000_000 or model_info.parameter_count < 50_000_000_000:
                return 8
            return 64

        return None

    def _estimate_vram(self, model_info: ModelInfo) -> float | None:
        """Estimate VRAM requirement for model."""
        if not model_info.parameter_count:
            return None

        # Rough estimation based on quantization
        quantization = (model_info.quantization or "").lower()
        bytes_per_param = 2.0  # Default FP16

        if "q4" in quantization or "4bit" in quantization:
            bytes_per_param = 0.5
        elif "q5" in quantization or "5bit" in quantization:
            bytes_per_param = 0.625
        elif "q6" in quantization or "6bit" in quantization:
            bytes_per_param = 0.75
        elif "q8" in quantization or "8bit" in quantization:
            bytes_per_param = 1.0
        elif "fp16" in quantization or "half" in quantization:
            bytes_per_param = 2.0
        elif "fp32" in quantization or "float" in quantization:
            bytes_per_param = 4.0

        # Model weights
        weights_gb = (model_info.parameter_count * bytes_per_param) / (1024**3)

        # Add overhead for KV cache, context, etc. (roughly 20-30%)
        overhead_factor = 1.3

        return round(weights_gb * overhead_factor, 2)

    async def _probe_model_capabilities(
        self, model_id: str, capabilities: ModelCapabilities
    ) -> None:
        """Probe model by loading with minimal config."""
        # Load with minimal context to inspect
        probe_config = LoadConfig(context_length=512)
        result = await self.client.load_model(model_id, probe_config)

        if result.success and result.load_config:
            # Update capabilities based on actual load config
            if result.load_config.context_length:
                capabilities.context_length = result.load_config.context_length

        # Always unload after probing
        await self.client.unload_model(model_id=model_id)

    def generate_context_candidates(self, capabilities: ModelCapabilities) -> list[int]:
        """Generate sensible context length candidates for testing."""
        max_ctx = capabilities.max_context_length or capabilities.context_length or 4096

        # Standard candidates - benchmark-specific test points (powers of two, not scoring constants)
        # See docs/OPTIMIZATION_METHOD.md: these are filtered to model max, not used for normalization
        candidates = [2048, 4096, 8192, 12288, 16384, 24576, 32768, 65536, 131072]

        # Filter to valid values <= max context
        valid = [c for c in candidates if c <= max_ctx]

        # Always include max context
        if max_ctx not in valid:
            valid.append(max_ctx)

        return sorted(set(valid))

    def generate_gpu_ratio_candidates(self, capabilities: ModelCapabilities) -> list[float]:
        """Generate GPU offload ratio candidates."""
        if not config.hardware.gpu_vram_gb:
            return [0.0]  # CPU only

        vram_gb = config.hardware.gpu_vram_gb
        estimated = capabilities.estimated_vram_gb or 0

        candidates = []

        if estimated <= vram_gb * 0.7:
            # Model fits comfortably, test high ratios
            candidates = [1.0, 0.9, 0.8, 0.7, 0.6]
        elif estimated <= vram_gb * 0.9:
            # Tight fit, test moderate ratios
            candidates = [0.8, 0.7, 0.6, 0.5, 0.4]
        else:
            # Model larger than VRAM, test lower ratios
            candidates = [0.6, 0.5, 0.4, 0.3, 0.25, 0.0]

        return candidates

    def generate_batch_size_candidates(self, capabilities: ModelCapabilities) -> list[int]:
        """Generate eval batch size candidates."""
        # Default candidates - benchmark-specific powers of two, not hardware-specific scoring
        return [64, 128, 256, 512, 1024]

    def generate_kv_cache_candidates(self, capabilities: ModelCapabilities) -> list[bool]:
        """Generate KV cache placement candidates."""
        # Test both GPU and CPU KV cache
        return [True, False]

    def generate_flash_attention_candidates(self, capabilities: ModelCapabilities) -> list[bool]:
        """Generate Flash Attention candidates."""
        if not self.client.capabilities.supports_flash_attention:
            return [False]
        return [True, False]
