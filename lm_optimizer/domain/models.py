"""Domain models for LM Studio Auto Optimizer."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4


class OptimizationProfile(str, Enum):
    """Optimization profile types."""

    SPEED = "speed"
    BALANCED = "balanced"
    CONTEXT = "context"
    QUALITY = "quality"
    CUSTOM = "custom"


class OptimizationStage(str, Enum):
    """Optimization stages."""

    DISCOVERY = "discovery"
    COARSE_SEARCH = "coarse_search"
    REFINEMENT = "refinement"
    BATCH_OPTIMIZATION = "batch_optimization"
    VALIDATION = "validation"
    COMPLETE = "complete"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    ERROR = "error"


class RunStatus(str, Enum):
    """Optimization run status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ConfigurationStatus(str, Enum):
    """Configuration test status."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    OOM = "oom"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class HardwareInfo:
    """Hardware information snapshot."""

    os: str
    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    total_ram_gb: float
    gpu_count: int
    gpus: list["GPUInfo"] = field(default_factory=list)
    cuda_version: str | None = None
    metal_available: bool = False
    vulkan_available: bool = False
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class GPUInfo:
    """GPU information."""

    index: int
    name: str
    vram_gb: float
    vendor: str
    driver_version: str | None = None
    compute_capability: str | None = None
    shared_memory_gb: float | None = None


@dataclass
class ModelIdentity:
    """Unique model identifier."""

    id: str
    name: str
    architecture: str | None = None
    parameter_count: int | None = None
    quantization: str | None = None
    context_limit: int | None = None
    is_moe: bool = False
    num_experts: int | None = None
    size_bytes: int | None = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, ModelIdentity):
            return self.id == other.id
        return False


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""

    name: str
    category: str
    prompt: str
    max_tokens: int
    temperature: float = 0.7
    stop_sequences: list[str] | None = None


@dataclass
class LoadConfiguration:
    """Model load configuration parameters."""

    context_length: int | None = None
    gpu_ratio: float | None = None
    flash_attention: bool | None = None
    offload_kv_cache_to_gpu: bool | None = None
    eval_batch_size: int | None = None
    num_experts: int | None = None
    rope_freq_base: float | None = None
    rope_freq_scale: float | None = None

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def to_api_params(self) -> dict:
        """Convert to API parameters."""
        return self.to_dict()


@dataclass
class LMStudioCapabilities:
    """Detected LM Studio API capabilities."""

    version: str = "unknown"
    supports_context_length: bool = True
    supports_gpu_ratio: bool = True
    supports_flash_attention: bool = False
    supports_kv_cache_placement: bool = False
    supports_kv_cache_quantization: bool = False
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
        if self.supports_kv_cache_placement:
            params.append("offload_kv_cache_to_gpu")
        if self.supports_kv_cache_quantization:
            params.append("kv_cache_quantization")
        if self.supports_eval_batch_size:
            params.append("eval_batch_size")
        if self.supports_num_experts:
            params.append("num_experts")
        if self.supports_rope_scaling:
            params.extend(["rope_freq_base", "rope_freq_scale"])
        return params


@dataclass
class BenchmarkMetrics:
    """Metrics from a single benchmark test.

    Note: estimated_ttft_ms is an *estimate* derived from total generation
    time when streaming is not available (ttft ~ 10% of total time).
    It is NOT a directly measured time-to-first-token. See docs/OPTIMIZATION_METHOD.md.
    """

    test_name: str
    category: str
    success: bool
    load_time_ms: float = 0.0
    estimated_ttft_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_processing_ms: float = 0.0
    generation_ms: float = 0.0
    prompt_tok_s: float = 0.0
    generation_tok_s: float = 0.0
    error: str | None = None
    output_text: str = ""

    @property
    def ttft_ms(self) -> float:
        """Backward compatibility alias for estimated_ttft_ms."""
        return self.estimated_ttft_ms

    @ttft_ms.setter
    def ttft_ms(self, value: float) -> None:
        self.estimated_ttft_ms = value


@dataclass
class QualityScore:
    """Correctness / quality assessment result.

    This is NOT an objective model quality percentage. It represents
    deterministic heuristic checks (JSON validation, keyword presence,
    code structure, truncation, repetition). Display as
    'Correctness Score' or 'X / Y checks passed', not as precise
    '99.1% quality'. See docs/OPTIMIZATION_METHOD.md.
    """

    overall: float
    task_completion: float
    factual_consistency: float
    format_compliance: float
    coding_correctness: float
    no_truncation: float
    no_malformed: float
    confident: bool = True
    details: dict = field(default_factory=dict)
    checks_passed: int | None = None
    checks_total: int | None = None

    def as_checks_str(self) -> str:
        """Human-readable checks summary."""
        total = self.checks_total
        passed = self.checks_passed
        if total and passed is not None:
            return f"{passed} / {total} checks passed"
        # Fallback: derive from overall (6 dimensions)
        # Count how many dimensions scored >= 0.9 as passed
        dims = [
            self.task_completion,
            self.factual_consistency,
            self.format_compliance,
            self.coding_correctness,
            self.no_truncation,
            self.no_malformed,
        ]
        passed = sum(1 for d in dims if d >= 0.9)
        return f"{passed} / {len(dims)} checks passed"


@dataclass
class ConfigurationResult:
    """Result of testing a single configuration."""

    id: UUID = field(default_factory=uuid4)
    run_id: UUID | None = None
    config: LoadConfiguration = field(default_factory=LoadConfiguration)
    context_length: int = 4096
    status: ConfigurationStatus = ConfigurationStatus.PENDING
    metrics: list[BenchmarkMetrics] = field(default_factory=list)
    quality_score: QualityScore | None = None
    stability_score: float = 1.0
    peak_vram_gb: float | None = None
    peak_ram_gb: float | None = None
    error: str | None = None
    score: float = 0.0
    score_breakdown: dict | None = None
    tested_at: datetime | None = None
    duration_ms: float = 0.0

    def get_avg_generation_tok_s(self) -> float:
        speeds = [m.generation_tok_s for m in self.metrics if m.success and m.generation_tok_s > 0]
        if not speeds:
            return 0.0
        return sum(speeds) / len(speeds)

    def get_avg_prompt_tok_s(self) -> float:
        speeds = [m.prompt_tok_s for m in self.metrics if m.success and m.prompt_tok_s > 0]
        if not speeds:
            return 0.0
        return sum(speeds) / len(speeds)

    def get_avg_ttft_ms(self) -> float:
        """Backward compat: returns estimated TTFT."""
        return self.get_avg_estimated_ttft_ms()

    def get_avg_estimated_ttft_ms(self) -> float:
        ttfts = [m.estimated_ttft_ms for m in self.metrics if m.success and m.estimated_ttft_ms > 0]
        if not ttfts:
            return 0.0
        return sum(ttfts) / len(ttfts)


@dataclass
class OptimizationRun:
    """Complete optimization run."""

    id: UUID = field(default_factory=uuid4)
    model: ModelIdentity = field(default_factory=lambda: ModelIdentity(id="", name=""))
    hardware: HardwareInfo = field(
        default_factory=lambda: HardwareInfo(
            os="",
            cpu_name="",
            cpu_cores_physical=0,
            cpu_cores_logical=0,
            total_ram_gb=0,
            gpu_count=0,
        )
    )
    hardware_id: str | None = None
    profile: OptimizationProfile = OptimizationProfile.BALANCED
    profile_weights: dict[str, float] = field(default_factory=dict)
    quality_threshold: float = 0.97
    status: RunStatus = RunStatus.PENDING
    stage: OptimizationStage = OptimizationStage.DISCOVERY
    configurations: list[ConfigurationResult] = field(default_factory=list)
    best_config_id: UUID | None = None
    pareto_config_ids: list[UUID] = field(default_factory=list)
    baseline_config_id: UUID | None = None
    baseline_metrics: dict | None = None
    is_experimental: bool = False
    experimental_reason: str | None = None
    search_space: dict = field(default_factory=dict)
    benchmark_params: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    error: str | None = None
    optimizer_version: str = "1.0.0-beta"
    benchmark_repetitions: int = 3
    validation_repetitions: int = 5

    def get_best_config(self) -> ConfigurationResult | None:
        if self.best_config_id:
            for c in self.configurations:
                if c.id == self.best_config_id:
                    return c
        return None

    def get_pareto_configs(self) -> list[ConfigurationResult]:
        return [c for c in self.configurations if c.id in self.pareto_config_ids]

    def get_baseline_config(self) -> ConfigurationResult | None:
        if self.baseline_config_id:
            for c in self.configurations:
                if c.id == self.baseline_config_id:
                    return c
        return None


@dataclass
class ProfileWeights:
    """Optimization profile weights."""

    generation_speed: float = 0.0
    prompt_speed: float = 0.0
    ttft: float = 0.0
    quality: float = 0.0
    stability: float = 0.0
    context: float = 0.0
    memory_efficiency: float = 0.0

    def validate(self) -> bool:
        total = (
            self.generation_speed
            + self.prompt_speed
            + self.ttft
            + self.quality
            + self.stability
            + self.context
            + self.memory_efficiency
        )
        return abs(total - 1.0) < 0.01

    def to_dict(self) -> dict:
        return {
            "generation_speed": self.generation_speed,
            "prompt_speed": self.prompt_speed,
            "ttft": self.ttft,
            "quality": self.quality,
            "stability": self.stability,
            "context": self.context,
            "memory_efficiency": self.memory_efficiency,
        }


DEFAULT_PROFILE_WEIGHTS = {
    OptimizationProfile.SPEED: ProfileWeights(
        generation_speed=0.40,
        prompt_speed=0.25,
        ttft=0.15,
        quality=0.10,
        stability=0.05,
        context=0.03,
        memory_efficiency=0.02,
    ),
    OptimizationProfile.BALANCED: ProfileWeights(
        generation_speed=0.27,
        prompt_speed=0.15,
        ttft=0.12,
        quality=0.12,
        stability=0.12,
        context=0.11,
        memory_efficiency=0.11,
    ),
    OptimizationProfile.CONTEXT: ProfileWeights(
        generation_speed=0.10,
        prompt_speed=0.10,
        ttft=0.10,
        quality=0.15,
        stability=0.15,
        context=0.25,
        memory_efficiency=0.15,
    ),
    OptimizationProfile.QUALITY: ProfileWeights(
        generation_speed=0.08,
        prompt_speed=0.08,
        ttft=0.08,
        quality=0.35,
        stability=0.13,
        context=0.21,
        memory_efficiency=0.07,
    ),
}
