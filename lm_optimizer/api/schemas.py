"""API schemas for request/response validation."""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, validator


class ProfileWeightsSchema(BaseModel):
    generation_speed: float = 0.0
    prompt_speed: float = 0.0
    ttft: float = 0.0
    quality: float = 0.0
    stability: float = 0.0
    context: float = 0.0
    memory_efficiency: float = 0.0

    @validator("*")
    def validate_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Weight must be between 0 and 1")
        return v


class OptimizationProfileSchema(str, Enum):
    SPEED = "speed"
    BALANCED = "balanced"
    CONTEXT = "context"
    QUALITY = "quality"
    CUSTOM = "custom"


class HardwareInfoSchema(BaseModel):
    os: str
    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    total_ram_gb: float
    gpu_count: int
    gpus: list[GPUInfoSchema] = []
    cuda_version: str | None = None
    metal_available: bool = False
    vulkan_available: bool = False


class GPUInfoSchema(BaseModel):
    index: int
    name: str
    vram_gb: float
    vendor: str
    driver_version: str | None = None
    compute_capability: str | None = None
    shared_memory_gb: float | None = None


class ModelIdentitySchema(BaseModel):
    id: str
    name: str
    architecture: str | None = None
    parameter_count: int | None = None
    quantization: str | None = None
    context_limit: int | None = None
    is_moe: bool = False
    num_experts: int | None = None
    size_bytes: int | None = None


class LoadConfigSchema(BaseModel):
    context_length: int | None = None
    gpu_ratio: float | None = None
    flash_attention: bool | None = None
    offload_kv_cache_to_gpu: bool | None = None
    eval_batch_size: int | None = None
    num_experts: int | None = None
    rope_freq_base: float | None = None
    rope_freq_scale: float | None = None


class AdvancedSettingsSchema(BaseModel):
    min_context: int | None = 2048
    max_context: int | None = 32768
    custom_contexts: list[int] = []
    min_gpu_ratio: float | None = 0.0
    max_gpu_ratio: float | None = 1.0
    gpu_ratio_step: float | None = 0.1
    test_flash_on: bool = True
    test_flash_off: bool = True
    test_kv_gpu: bool = True
    test_kv_cpu: bool = True
    auto_batch: bool = True
    min_batch: int | None = 64
    max_batch: int | None = 1024
    enable_rope: bool = False  # Experimental, disabled by default. Must be explicitly enabled.
    enable_rope_scaling: bool | None = None  # alias


class OptimizationRequest(BaseModel):
    model_id: str
    profile: OptimizationProfileSchema = OptimizationProfileSchema.BALANCED
    custom_weights: ProfileWeightsSchema | None = None
    quality_threshold: float = Field(default=0.97, ge=0.9, le=1.0)
    advanced_settings: AdvancedSettingsSchema | None = None


class OptimizationRunResponse(BaseModel):
    id: UUID
    model: ModelIdentitySchema
    hardware: HardwareInfoSchema
    profile: OptimizationProfileSchema
    profile_weights: ProfileWeightsSchema
    quality_threshold: float
    status: str
    stage: str
    search_space: dict
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0
    error: str | None = None
    baseline_config_id: UUID | None = None
    baseline_metrics: dict | None = None
    best_config_id: UUID | None = None
    pareto_config_ids: list[UUID] = []
    is_experimental: bool = False
    experimental_reason: str | None = None
    benchmark_params: dict | None = None


class ConfigurationResultResponse(BaseModel):
    id: UUID
    run_id: UUID
    config: LoadConfigSchema
    context_length: int
    status: str
    metrics: list[dict] = []
    quality: dict | None = None
    stability_score: float = 1.0
    peak_vram_gb: float | None = None
    peak_ram_gb: float | None = None
    error: str | None = None
    score: float = 0.0
    score_breakdown: dict | None = None
    tested_at: datetime | None = None
    duration_ms: float = 0.0
    avg_generation_tok_s: float = 0.0
    avg_prompt_tok_s: float = 0.0
    avg_estimated_ttft_ms: float = 0.0
    avg_ttft_ms: float = 0.0  # backward compat alias


class RunProgressResponse(BaseModel):
    run_id: UUID
    stage: str
    progress: int
    configs_tested: int
    configs_total: int
    configs_passed: int
    configs_failed: int
    configs_oom: int
    current_config: LoadConfigSchema | None = None
    current_metrics: dict | None = None
    best_score: float = 0.0
    best_config: LoadConfigSchema | None = None


class PresetSchema(BaseModel):
    id: UUID
    model_id: str
    profile: OptimizationProfileSchema
    name: str | None = None
    config: LoadConfigSchema
    metrics: dict | None = None
    quality: dict | None = None
    run_id: UUID | None = None
    created_at: datetime
    updated_at: datetime
    optimizer_version: str


class SettingsSchema(BaseModel):
    lm_studio_url: str = "http://127.0.0.1:1234"
    web_host: str = "127.0.0.1"
    web_port: int = 8080
    default_profile: str = "balanced"
    default_quality_threshold: float = 0.97
    default_benchmark_repetitions: int = 3
    default_validation_repetitions: int = 5
    auto_unload_after_tests: bool = True
    timeout_seconds: int = 300


class ApplyConfigRequest(BaseModel):
    model_id: str
    config: LoadConfigSchema
