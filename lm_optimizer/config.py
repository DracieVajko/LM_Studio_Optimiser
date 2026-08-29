"""Configuration management for LM Studio Auto Optimizer."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class HardwareConfig(BaseSettings):
    """Hardware configuration with auto-detection support."""

    model_config = SettingsConfigDict(env_prefix="HW_", extra="ignore")

    gpu_vram_gb: float | None = Field(
        default=None, description="GPU VRAM in GB (auto-detected if not set)"
    )
    system_ram_gb: float | None = Field(
        default=None, description="System RAM in GB (auto-detected if not set)"
    )
    gpu_name: str | None = Field(default=None, description="GPU name (auto-detected if not set)")
    gpu_count: int = Field(default=1, description="Number of GPUs")


class LMStudioConfig(BaseSettings):
    """LM Studio connection configuration."""

    model_config = SettingsConfigDict(env_prefix="LM_STUDIO_", extra="ignore")

    base_url: str = Field(default="http://127.0.0.1:1234", description="LM Studio API base URL")
    api_version: str | None = Field(default=None, description="API version (auto-detected)")
    timeout: float = Field(default=120.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum connection retries")
    retry_delay: float = Field(default=2.0, description="Delay between retries in seconds")

    @field_validator("base_url", mode="before")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate LM Studio URL is http(s) and not empty."""
        if not v or not isinstance(v, str):
            raise ValueError("LM Studio URL must be a non-empty string")
        v = v.strip().rstrip("/")
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("LM Studio URL must start with http:// or https://")
        return v


class OptimizationConfig(BaseSettings):
    """Optimization behavior configuration."""

    model_config = SettingsConfigDict(env_prefix="OPT_", extra="ignore")

    default_profile: Literal["speed", "balanced", "context", "quality", "custom"] = Field(
        default="balanced", description="Default optimization profile"
    )
    minimum_quality_score: float = Field(
        default=0.97, ge=0.0, le=1.0, description="Minimum quality threshold (0-1)"
    )
    benchmark_runs: int = Field(
        default=3, ge=1, le=10, description="Number of measured benchmark runs"
    )
    warmup_runs: int = Field(
        default=1, ge=0, le=5, description="Number of warmup runs before measurement"
    )

    benchmark_timeout: float = Field(
        default=300.0, description="Total benchmark timeout in seconds"
    )
    load_model_timeout: float = Field(default=60.0, description="Model load timeout in seconds")
    generation_timeout: float = Field(
        default=120.0, description="Single generation timeout in seconds"
    )


class StorageConfig(BaseSettings):
    """Storage paths configuration."""

    model_config = SettingsConfigDict(env_prefix="STORAGE_", extra="ignore")

    results_dir: Path = Field(
        default=Path("./data/reports"), description="Benchmark results directory"
    )
    profiles_dir: Path = Field(
        default=Path("./data/presets"), description="Optimized profiles directory"
    )
    logs_dir: Path = Field(default=Path("./data/logs"), description="Log files directory")
    config_dir: Path = Field(
        default=Path("./data"), description="Configuration and database directory"
    )
    database_path: Path = Field(
        default=Path("./data/optimizer.db"), description="SQLite database path"
    )

    @field_validator(
        "results_dir", "profiles_dir", "logs_dir", "config_dir", "database_path", mode="before"
    )
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        return Path(v).resolve()


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_", extra="ignore")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    format: Literal["json", "text"] = Field(default="json")


class WebUIConfig(BaseSettings):
    """Web UI configuration."""

    model_config = SettingsConfigDict(env_prefix="WEB_UI_", extra="ignore")

    enabled: bool = Field(default=True, description="Enable web UI")
    host: str = Field(default="127.0.0.1", description="Web UI host")
    port: int = Field(default=8080, description="Web UI port")


class SafetyConfig(BaseSettings):
    """Safety and recovery configuration."""

    model_config = SettingsConfigDict(env_prefix="SAFETY_", extra="ignore")

    auto_unload_between_tests: bool = Field(
        default=True, description="Unload model between configuration tests"
    )
    force_unload_on_error: bool = Field(default=True, description="Force unload on any error")
    checkpoint_interval: int = Field(
        default=5, description="Save checkpoint every N configurations"
    )


class Config(BaseSettings):
    """Main application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    lm_studio: LMStudioConfig = Field(default_factory=LMStudioConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    web_ui: WebUIConfig = Field(default_factory=WebUIConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)

    def ensure_directories(self) -> None:
        """Create all configured directories."""
        for dir_path in [
            self.storage.results_dir,
            self.storage.profiles_dir,
            self.storage.logs_dir,
            self.storage.config_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        # Ensure database parent dir exists
        self.storage.database_path.parent.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
config.ensure_directories()
