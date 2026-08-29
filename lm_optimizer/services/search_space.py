"""Search space generation for optimization."""

from dataclasses import dataclass, field

from lm_optimizer.domain.models import (
    HardwareInfo,
    ModelIdentity,
    OptimizationProfile,
)
from lm_optimizer.logging_config import get_logger
from lm_optimizer.services.lm_studio import LMStudioCapabilities, LMStudioClient

logger = get_logger(__name__)


@dataclass
class SearchSpace:
    """Optimization search space."""

    context_lengths: list[int] = field(default_factory=list)
    gpu_ratios: list[float] = field(default_factory=list)
    flash_attention_options: list[bool] = field(default_factory=list)
    kv_cache_options: list[bool] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    expert_counts: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "context_lengths": self.context_lengths,
            "gpu_ratios": self.gpu_ratios,
            "flash_attention_options": self.flash_attention_options,
            "kv_cache_options": self.kv_cache_options,
            "batch_sizes": self.batch_sizes,
            "expert_counts": self.expert_counts,
        }

    def estimate_size(self) -> int:
        """Estimate total number of configurations."""
        return (
            len(self.context_lengths)
            * len(self.gpu_ratios)
            * len(self.flash_attention_options)
            * len(self.kv_cache_options)
            * len(self.batch_sizes)
            * max(1, len(self.expert_counts))
        )


class SearchSpaceGenerator:
    """Generates dynamic search space based on hardware, model, and LM Studio capabilities."""

    def __init__(self, client: LMStudioClient):
        self.client = client

    def generate(
        self,
        model: ModelIdentity,
        hardware: HardwareInfo,
        profile: OptimizationProfile,
        advanced_settings: dict | None = None,
    ) -> SearchSpace:
        """Generate search space for optimization."""
        caps = self.client.capabilities
        advanced = advanced_settings or {}

        space = SearchSpace()

        # Context lengths
        space.context_lengths = self._generate_context_candidates(model, caps, advanced)

        # GPU ratios
        space.gpu_ratios = self._generate_gpu_ratio_candidates(model, hardware, caps, advanced)

        # Flash Attention
        space.flash_attention_options = self._generate_flash_attention_options(caps, advanced)

        # KV Cache
        space.kv_cache_options = self._generate_kv_cache_options(caps, advanced)

        # Batch sizes
        space.batch_sizes = self._generate_batch_sizes(caps, advanced)

        # Expert counts (MoE only)
        if model.is_moe and caps.supports_num_experts:
            space.expert_counts = self._generate_expert_counts(model, caps, advanced)

        # RoPE parameters: EXPERIMENTAL - disabled by default
        # Only include if explicitly enabled via advanced_settings enable_rope=True
        # See docs/OPTIMIZATION_METHOD.md - RoPE is experimental and requires stronger validation
        if not advanced.get("enable_rope", False):
            # Ensure no RoPE configs are generated; strip any RoPE from search space if present
            pass  # RoPE not part of normal search space
        else:
            # If enabled, mark search space as experimental
            space.to_dict()["is_experimental_rope"] = True

        logger.info("Generated search space", **space.to_dict(), estimated=space.estimate_size())
        return space

    def _generate_context_candidates(
        self, model: ModelIdentity, caps: LMStudioCapabilities, advanced: dict
    ) -> list[int]:
        """Generate context length candidates."""
        max_ctx = model.context_limit or 4096

        # Use advanced settings if provided
        min_ctx = advanced.get("min_context", 2048)
        max_ctx_setting = advanced.get("max_context", max_ctx)
        max_ctx = min(max_ctx, max_ctx_setting)

        # Standard candidates - benchmark-specific (powers of 2 and common context sizes)
        # These are test points, not scoring constants. Documented in docs/OPTIMIZATION_METHOD.md
        candidates = [2048, 4096, 8192, 12288, 16384, 24576, 32768, 65536, 131072]

        # Filter to valid range
        valid = [c for c in candidates if min_ctx <= c <= max_ctx]

        # Custom values from advanced settings
        custom = advanced.get("custom_contexts", [])
        for c in custom:
            if min_ctx <= c <= max_ctx and c not in valid:
                valid.append(c)

        # Always include max context
        if max_ctx not in valid:
            valid.append(max_ctx)

        return sorted(set(valid))

    def _generate_gpu_ratio_candidates(
        self,
        model: ModelIdentity,
        hardware: HardwareInfo,
        caps: LMStudioCapabilities,
        advanced: dict,
    ) -> list[float]:
        """Generate GPU offload ratio candidates."""
        if not caps.supports_gpu_ratio:
            return [1.0]

        # Get VRAM
        primary_gpu = hardware.gpus[0] if hardware.gpus else None
        vram_gb = primary_gpu.vram_gb if primary_gpu else 0

        # Check if we have GPU
        if vram_gb <= 0:
            return [0.0]

        # Use advanced settings
        min_ratio = advanced.get("min_gpu_ratio", 0.0)
        max_ratio = advanced.get("max_gpu_ratio", 1.0)
        step = advanced.get("gpu_ratio_step", 0.1)

        # Estimate model VRAM
        estimated_vram = self._estimate_model_vram(model)

        # Generate candidates based on VRAM fit
        if estimated_vram <= vram_gb * 0.7:
            # Model fits comfortably
            base = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        elif estimated_vram <= vram_gb * 0.9:
            # Tight fit
            base = [0.8, 0.7, 0.6, 0.5, 0.4]
        else:
            # Model larger than VRAM
            base = [0.6, 0.5, 0.4, 0.3, 0.25, 0.0]

        # Filter by bounds
        valid = [r for r in base if min_ratio <= r <= max_ratio]

        # Add stepped values
        stepped = []
        r = min_ratio
        while r <= max_ratio + 0.001:
            if r not in valid:
                stepped.append(round(r, 2))
            r += step

        return sorted(set(valid + stepped))

    def _generate_flash_attention_options(
        self, caps: LMStudioCapabilities, advanced: dict
    ) -> list[bool]:
        """Generate Flash Attention options."""
        if not caps.supports_flash_attention:
            return [False]

        options = []
        if advanced.get("test_flash_on", True):
            options.append(True)
        if advanced.get("test_flash_off", True):
            options.append(False)
        return options or [False]

    def _generate_kv_cache_options(self, caps: LMStudioCapabilities, advanced: dict) -> list[bool]:
        """Generate KV cache placement options."""
        if not caps.supports_kv_cache_placement:
            return [True]  # Default to GPU if not controllable

        options = []
        if advanced.get("test_kv_gpu", True):
            options.append(True)
        if advanced.get("test_kv_cpu", True):
            options.append(False)
        return options or [True]

    def _generate_batch_sizes(self, caps: LMStudioCapabilities, advanced: dict) -> list[int]:
        """Generate eval batch size candidates."""
        if not caps.supports_eval_batch_size:
            return [256]

        if advanced.get("auto_batch", True):
            return [64, 128, 256, 512, 1024]

        min_batch = advanced.get("min_batch", 64)
        max_batch = advanced.get("max_batch", 1024)

        # Generate powers of 2
        sizes = []
        size = 64
        while size <= max_batch:
            if size >= min_batch:
                sizes.append(size)
            size *= 2

        return sizes or [256]

    def _generate_expert_counts(
        self, model: ModelIdentity, caps: LMStudioCapabilities, advanced: dict
    ) -> list[int]:
        """Generate expert count candidates for MoE models."""
        if not caps.supports_num_experts or not model.is_moe:
            return []

        # Default expert count from model
        default = model.num_experts or 8

        # Common configurations
        candidates = [default]

        if default > 1:
            # Try reduced expert counts
            for div in [2, 4, 8]:
                reduced = default // div
                if reduced >= 1 and reduced not in candidates:
                    candidates.append(reduced)

        return sorted(candidates)

    def _estimate_model_vram(self, model: ModelIdentity) -> float:
        """Estimate VRAM requirement for model."""
        if not model.parameter_count:
            return 0.0

        quantization = (model.quantization or "").lower()
        bytes_per_param = 2.0

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

        weights_gb = (model.parameter_count * bytes_per_param) / (1024**3)
        return weights_gb * 1.3  # 30% overhead
