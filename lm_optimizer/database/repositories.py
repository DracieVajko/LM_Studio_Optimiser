"""Database repositories for domain entities."""

import json
import sqlite3
from datetime import datetime
from uuid import UUID

from lm_optimizer.database.manager import db_manager
from lm_optimizer.domain.models import (
    DEFAULT_PROFILE_WEIGHTS,
    BenchmarkMetrics,
    ConfigurationResult,
    ConfigurationStatus,
    GPUInfo,
    HardwareInfo,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
    OptimizationStage,
    ProfileWeights,
    QualityScore,
    RunStatus,
)


class HardwareRepository:
    """Repository for hardware snapshots."""

    def save(self, hardware: HardwareInfo) -> str:
        """Save hardware snapshot, return ID."""
        import uuid

        hw_id = str(uuid.uuid4())
        gpus_json = json.dumps(
            [
                {
                    "index": g.index,
                    "name": g.name,
                    "vram_gb": g.vram_gb,
                    "vendor": g.vendor,
                    "driver_version": g.driver_version,
                    "compute_capability": g.compute_capability,
                    "shared_memory_gb": g.shared_memory_gb,
                }
                for g in hardware.gpus
            ]
        )

        with db_manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO hardware_snapshots
                (id, os, cpu_name, cpu_cores_physical, cpu_cores_logical, total_ram_gb,
                 gpu_count, gpus_json, cuda_version, metal_available, vulkan_available, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    hw_id,
                    hardware.os,
                    hardware.cpu_name,
                    hardware.cpu_cores_physical,
                    hardware.cpu_cores_logical,
                    hardware.total_ram_gb,
                    hardware.gpu_count,
                    gpus_json,
                    hardware.cuda_version,
                    int(hardware.metal_available),
                    int(hardware.vulkan_available),
                    hardware.detected_at.isoformat(),
                ),
            )
        return hw_id

    def get(self, hw_id: str) -> HardwareInfo | None:
        """Get hardware snapshot by ID."""
        with db_manager.get_connection() as conn:
            row = conn.execute("SELECT * FROM hardware_snapshots WHERE id = ?", (hw_id,)).fetchone()
            if not row:
                return None
            return self._row_to_hardware(row)

    def _row_to_hardware(self, row: sqlite3.Row) -> HardwareInfo:
        gpus = json.loads(row["gpus_json"])
        return HardwareInfo(
            os=row["os"],
            cpu_name=row["cpu_name"],
            cpu_cores_physical=row["cpu_cores_physical"],
            cpu_cores_logical=row["cpu_cores_logical"],
            total_ram_gb=row["total_ram_gb"],
            gpu_count=row["gpu_count"],
            gpus=[GPUInfo(**g) for g in gpus],
            cuda_version=row["cuda_version"],
            metal_available=bool(row["metal_available"]),
            vulkan_available=bool(row["vulkan_available"]),
        )


class ModelRepository:
    """Repository for model identities."""

    def save(self, model: ModelIdentity) -> str:
        """Save or update model, return ID."""

        now = datetime.now().isoformat()
        metadata = {
            k: v
            for k, v in {
                "architecture": model.architecture,
                "parameter_count": model.parameter_count,
                "quantization": model.quantization,
                "context_limit": model.context_limit,
                "is_moe": model.is_moe,
                "num_experts": model.num_experts,
                "size_bytes": model.size_bytes,
            }.items()
            if v is not None
        }

        with db_manager.get_connection() as conn:
            existing = conn.execute("SELECT id FROM models WHERE id = ?", (model.id,)).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE models SET name=?, architecture=?, parameter_count=?, quantization=?,
                    context_limit=?, is_moe=?, num_experts=?, size_bytes=?, metadata_json=?,
                    last_seen=? WHERE id=?
                """,
                    (
                        model.name,
                        model.architecture,
                        model.parameter_count,
                        model.quantization,
                        model.context_limit,
                        int(model.is_moe),
                        model.num_experts,
                        model.size_bytes,
                        json.dumps(metadata),
                        now,
                        model.id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO models
                    (id, name, architecture, parameter_count, quantization, context_limit,
                     is_moe, num_experts, size_bytes, metadata_json, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        model.id,
                        model.name,
                        model.architecture,
                        model.parameter_count,
                        model.quantization,
                        model.context_limit,
                        int(model.is_moe),
                        model.num_experts,
                        model.size_bytes,
                        json.dumps(metadata),
                        now,
                        now,
                    ),
                )
        return model.id

    def get(self, model_id: str) -> ModelIdentity | None:
        """Get model by ID."""
        with db_manager.get_connection() as conn:
            row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
            if not row:
                return None
            metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            return ModelIdentity(
                id=row["id"],
                name=row["name"],
                architecture=row["architecture"],
                parameter_count=row["parameter_count"],
                quantization=row["quantization"],
                context_limit=row["context_limit"],
                is_moe=bool(row["is_moe"]),
                num_experts=row["num_experts"],
                size_bytes=row["size_bytes"],
            )

    def list_all(self) -> list[ModelIdentity]:
        """List all models."""
        with db_manager.get_connection() as conn:
            rows = conn.execute("SELECT * FROM models ORDER BY last_seen DESC").fetchall()
            return [self._row_to_model(row) for row in rows]

    def _row_to_model(self, row: sqlite3.Row) -> ModelIdentity:
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        return ModelIdentity(
            id=row["id"],
            name=row["name"],
            architecture=row["architecture"],
            parameter_count=row["parameter_count"],
            quantization=row["quantization"],
            context_limit=row["context_limit"],
            is_moe=bool(row["is_moe"]),
            num_experts=row["num_experts"],
            size_bytes=row["size_bytes"],
        )


class RunRepository:
    """Repository for optimization runs."""

    def save(self, run: OptimizationRun) -> str:
        """Save optimization run, return ID."""
        import uuid

        run_id = str(run.id) if run.id else str(uuid.uuid4())
        profile_weights = (
            run.profile_weights
            or DEFAULT_PROFILE_WEIGHTS.get(run.profile, ProfileWeights()).to_dict()
        )
        # Resolve hardware_id
        hw_id = run.hardware_id
        if not hw_id and hasattr(run, "hardware") and run.hardware:
            # If hardware object exists but no id, try to save
            try:
                hw_id = hardware_repo.save(run.hardware)  # type: ignore
                run.hardware_id = hw_id
            except Exception:
                hw_id = "unknown"

        with db_manager.get_connection() as conn:
            existing = conn.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE runs SET
                        model_id=?, hardware_id=?, profile=?, profile_weights_json=?,
                        quality_threshold=?, status=?, stage=?, search_space_json=?,
                        optimizer_version=?, benchmark_repetitions=?, validation_repetitions=?,
                        created_at=?, started_at=?, completed_at=?, duration_seconds=?,
                        error=?, baseline_config_id=?, baseline_metrics_json=?,
                        is_experimental=?, experimental_reason=?, benchmark_params_json=?
                    WHERE id=?
                """,
                    (
                        run.model.id,
                        hw_id,
                        run.profile.value,
                        json.dumps(profile_weights),
                        run.quality_threshold,
                        run.status.value,
                        run.stage.value,
                        json.dumps(run.search_space),
                        run.optimizer_version,
                        run.benchmark_repetitions,
                        run.validation_repetitions,
                        run.created_at.isoformat() if run.created_at else None,
                        run.started_at.isoformat() if run.started_at else None,
                        run.completed_at.isoformat() if run.completed_at else None,
                        run.duration_seconds,
                        run.error,
                        str(run.baseline_config_id) if run.baseline_config_id else None,
                        json.dumps(run.baseline_metrics) if run.baseline_metrics else None,
                        int(run.is_experimental),
                        run.experimental_reason,
                        json.dumps(run.benchmark_params) if run.benchmark_params else None,
                        run_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO runs
                    (id, model_id, hardware_id, profile, profile_weights_json, quality_threshold,
                     status, stage, search_space_json, optimizer_version,
                     benchmark_repetitions, validation_repetitions,
                     created_at, started_at, completed_at, duration_seconds, error,
                     baseline_config_id, baseline_metrics_json, is_experimental, experimental_reason, benchmark_params_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        run_id,
                        run.model.id,
                        hw_id,
                        run.profile.value,
                        json.dumps(profile_weights),
                        run.quality_threshold,
                        run.status.value,
                        run.stage.value,
                        json.dumps(run.search_space),
                        run.optimizer_version,
                        run.benchmark_repetitions,
                        run.validation_repetitions,
                        run.created_at.isoformat(),
                        run.started_at.isoformat() if run.started_at else None,
                        run.completed_at.isoformat() if run.completed_at else None,
                        run.duration_seconds,
                        run.error,
                        str(run.baseline_config_id) if run.baseline_config_id else None,
                        json.dumps(run.baseline_metrics) if run.baseline_metrics else None,
                        int(run.is_experimental),
                        run.experimental_reason,
                        json.dumps(run.benchmark_params) if run.benchmark_params else None,
                    ),
                )
        return run_id

    def get(self, run_id: str) -> OptimizationRun | None:
        """Get run by ID with configurations."""
        with db_manager.get_connection() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                return None
            run = self._row_to_run(row)
            run.configurations = self._get_configurations(run_id)
            return run

    def list_all(self, limit: int = 50, offset: int = 0) -> list[OptimizationRun]:
        """List all runs."""
        with db_manager.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
            return [self._row_to_run(row) for row in rows]

    def get_by_model(self, model_id: str, limit: int = 20) -> list[OptimizationRun]:
        """Get runs for a specific model."""
        with db_manager.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE model_id = ? ORDER BY created_at DESC LIMIT ?",
                (model_id, limit),
            ).fetchall()
            return [self._row_to_run(row) for row in rows]

    def _row_to_run(self, row: sqlite3.Row) -> OptimizationRun:
        profile_weights = (
            json.loads(row["profile_weights_json"]) if row["profile_weights_json"] else {}
        )
        search_space = json.loads(row["search_space_json"]) if row["search_space_json"] else {}
        # Handle optional new columns safely
        try:
            baseline_config_id = row["baseline_config_id"]
        except Exception:
            baseline_config_id = None
        try:
            baseline_metrics = (
                json.loads(row["baseline_metrics_json"]) if row["baseline_metrics_json"] else None
            )
        except Exception:
            baseline_metrics = None
        try:
            is_experimental = bool(row["is_experimental"])
        except Exception:
            is_experimental = False
        try:
            experimental_reason = row["experimental_reason"]
        except Exception:
            experimental_reason = None
        try:
            benchmark_params = (
                json.loads(row["benchmark_params_json"]) if row["benchmark_params_json"] else {}
            )
        except Exception:
            benchmark_params = {}

        return OptimizationRun(
            id=UUID(row["id"]),
            model=ModelIdentity(id=row["model_id"], name=""),
            hardware_id=row["hardware_id"],
            profile=OptimizationProfile(row["profile"]),
            profile_weights=profile_weights,
            quality_threshold=row["quality_threshold"],
            status=RunStatus(row["status"]),
            stage=OptimizationStage(row["stage"]),
            search_space=search_space,
            optimizer_version=row["optimizer_version"],
            benchmark_repetitions=row["benchmark_repetitions"],
            validation_repetitions=row["validation_repetitions"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None,
            duration_seconds=row["duration_seconds"],
            error=row["error"],
            baseline_config_id=UUID(baseline_config_id) if baseline_config_id else None,
            baseline_metrics=baseline_metrics,
            is_experimental=is_experimental,
            experimental_reason=experimental_reason,
            benchmark_params=benchmark_params,
        )

    def _get_configurations(self, run_id: str) -> list[ConfigurationResult]:
        """Get all configurations for a run."""
        with db_manager.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM configurations WHERE run_id = ? ORDER BY tested_at", (run_id,)
            ).fetchall()
            return [self._row_to_config(row) for row in rows]

    def _row_to_config(self, row: sqlite3.Row) -> ConfigurationResult:
        config = LoadConfiguration()
        if row["config_json"]:
            config_data = json.loads(row["config_json"])
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        metrics = []
        if row["metrics_json"]:
            metrics_data = json.loads(row["metrics_json"])
            for m in metrics_data:
                if "estimated_ttft_ms" not in m and "ttft_ms" in m:
                    m["estimated_ttft_ms"] = m.pop("ttft_ms")
                m.pop("ttft_ms", None)
                known = {
                    "test_name",
                    "category",
                    "success",
                    "load_time_ms",
                    "estimated_ttft_ms",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "prompt_processing_ms",
                    "generation_ms",
                    "prompt_tok_s",
                    "generation_tok_s",
                    "error",
                    "output_text",
                }
                filtered = {k: v for k, v in m.items() if k in known}
                metrics.append(BenchmarkMetrics(**filtered))

        quality = None
        if row["quality_json"]:
            quality = QualityScore(**json.loads(row["quality_json"]))

        try:
            breakdown = (
                json.loads(row["score_breakdown_json"]) if row["score_breakdown_json"] else None
            )
        except Exception:
            breakdown = None

        return ConfigurationResult(
            id=UUID(row["id"]),
            run_id=UUID(row["run_id"]),
            config=config,
            context_length=row["context_length"],
            status=ConfigurationStatus(row["status"]),
            metrics=metrics,
            quality_score=quality,
            stability_score=row["stability_score"],
            peak_vram_gb=row["peak_vram_gb"],
            peak_ram_gb=row["peak_ram_gb"],
            error=row["error"],
            score=row["score"],
            score_breakdown=breakdown,
            tested_at=datetime.fromisoformat(row["tested_at"]) if row["tested_at"] else None,
            duration_ms=row["duration_ms"],
        )


class ConfigurationRepository:
    """Repository for configuration results."""

    def save(self, config: ConfigurationResult) -> str:
        """Save configuration result."""
        metrics_json = json.dumps(
            [
                {
                    "test_name": m.test_name,
                    "category": m.category,
                    "success": m.success,
                    "load_time_ms": m.load_time_ms,
                    "estimated_ttft_ms": m.estimated_ttft_ms,
                    "prompt_tokens": m.prompt_tokens,
                    "completion_tokens": m.completion_tokens,
                    "total_tokens": m.total_tokens,
                    "prompt_processing_ms": m.prompt_processing_ms,
                    "generation_ms": m.generation_ms,
                    "prompt_tok_s": m.prompt_tok_s,
                    "generation_tok_s": m.generation_tok_s,
                    "error": m.error,
                    "output_text": m.output_text,
                }
                for m in config.metrics
            ]
        )

        quality_json = None
        if config.quality_score:
            quality_json = json.dumps(
                {
                    "overall": config.quality_score.overall,
                    "task_completion": config.quality_score.task_completion,
                    "factual_consistency": config.quality_score.factual_consistency,
                    "format_compliance": config.quality_score.format_compliance,
                    "coding_correctness": config.quality_score.coding_correctness,
                    "no_truncation": config.quality_score.no_truncation,
                    "no_malformed": config.quality_score.no_malformed,
                    "confident": config.quality_score.confident,
                    "details": config.quality_score.details,
                    "checks_passed": config.quality_score.checks_passed,
                    "checks_total": config.quality_score.checks_total,
                }
            )

        config_json = config.config.to_dict()

        breakdown_json = json.dumps(config.score_breakdown) if config.score_breakdown else None

        with db_manager.get_connection() as conn:
            existing = conn.execute(
                "SELECT id FROM configurations WHERE id = ?", (str(config.id),)
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE configurations SET
                        run_id=?, config_json=?, context_length=?, status=?,
                        metrics_json=?, quality_json=?, stability_score=?,
                        peak_vram_gb=?, peak_ram_gb=?, error=?, score=?,
                        score_breakdown_json=?, tested_at=?, duration_ms=?
                    WHERE id=?
                """,
                    (
                        str(config.run_id),
                        json.dumps(config_json),
                        config.context_length,
                        config.status.value,
                        metrics_json,
                        quality_json,
                        config.stability_score,
                        config.peak_vram_gb,
                        config.peak_ram_gb,
                        config.error,
                        config.score,
                        breakdown_json,
                        config.tested_at.isoformat() if config.tested_at else None,
                        config.duration_ms,
                        str(config.id),
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO configurations
                    (id, run_id, config_json, context_length, status, metrics_json,
                     quality_json, stability_score, peak_vram_gb, peak_ram_gb,
                     error, score, score_breakdown_json, tested_at, duration_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        str(config.id),
                        str(config.run_id),
                        json.dumps(config_json),
                        config.context_length,
                        config.status.value,
                        metrics_json,
                        quality_json,
                        config.stability_score,
                        config.peak_vram_gb,
                        config.peak_ram_gb,
                        config.error,
                        config.score,
                        breakdown_json,
                        config.tested_at.isoformat() if config.tested_at else None,
                        config.duration_ms,
                    ),
                )
        return str(config.id)

    def get(self, config_id: str) -> ConfigurationResult | None:
        """Get configuration by ID."""
        with db_manager.get_connection() as conn:
            row = conn.execute("SELECT * FROM configurations WHERE id = ?", (config_id,)).fetchone()
            if not row:
                return None
            return self._row_to_config(row)

    def _row_to_config(self, row: sqlite3.Row) -> ConfigurationResult:
        config = LoadConfiguration()
        if row["config_json"]:
            config_data = json.loads(row["config_json"])
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        metrics = []
        if row["metrics_json"]:
            metrics_data = json.loads(row["metrics_json"])
            for m in metrics_data:
                if "estimated_ttft_ms" not in m and "ttft_ms" in m:
                    m["estimated_ttft_ms"] = m.pop("ttft_ms")
                m.pop("ttft_ms", None)
                known = {
                    "test_name",
                    "category",
                    "success",
                    "load_time_ms",
                    "estimated_ttft_ms",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "prompt_processing_ms",
                    "generation_ms",
                    "prompt_tok_s",
                    "generation_tok_s",
                    "error",
                    "output_text",
                }
                filtered = {k: v for k, v in m.items() if k in known}
                metrics.append(BenchmarkMetrics(**filtered))

        quality = None
        if row["quality_json"]:
            quality = QualityScore(**json.loads(row["quality_json"]))

        try:
            breakdown = (
                json.loads(row["score_breakdown_json"]) if row["score_breakdown_json"] else None
            )
        except Exception:
            breakdown = None

        return ConfigurationResult(
            id=UUID(row["id"]),
            run_id=UUID(row["run_id"]),
            config=config,
            context_length=row["context_length"],
            status=ConfigurationStatus(row["status"]),
            metrics=metrics,
            quality_score=quality,
            stability_score=row["stability_score"],
            peak_vram_gb=row["peak_vram_gb"],
            peak_ram_gb=row["peak_ram_gb"],
            error=row["error"],
            score=row["score"],
            score_breakdown=breakdown,
            tested_at=datetime.fromisoformat(row["tested_at"]) if row["tested_at"] else None,
            duration_ms=row["duration_ms"],
        )


class PresetRepository:
    """Repository for saved presets."""

    def save(self, preset: dict) -> str:
        """Save preset, return ID."""
        import uuid

        preset_id = preset.get("id", str(uuid.uuid4()))
        now = datetime.now().isoformat()

        with db_manager.get_connection() as conn:
            existing = conn.execute("SELECT id FROM presets WHERE id = ?", (preset_id,)).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE presets SET
                        model_id=?, profile=?, name=?, config_json=?, metrics_json=?,
                        quality_json=?, run_id=?, updated_at=?, optimizer_version=?
                    WHERE id=?
                """,
                    (
                        preset["model_id"],
                        preset["profile"],
                        preset.get("name"),
                        json.dumps(preset["config"]),
                        json.dumps(preset.get("metrics")),
                        json.dumps(preset.get("quality")),
                        preset.get("run_id"),
                        now,
                        preset.get("optimizer_version", "1.0.0-beta"),
                        preset_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO presets
                    (id, model_id, profile, name, config_json, metrics_json,
                     quality_json, run_id, created_at, updated_at, optimizer_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        preset_id,
                        preset["model_id"],
                        preset["profile"],
                        preset.get("name"),
                        json.dumps(preset["config"]),
                        json.dumps(preset.get("metrics")),
                        json.dumps(preset.get("quality")),
                        preset.get("run_id"),
                        now,
                        now,
                        preset.get("optimizer_version", "1.0.0-beta"),
                    ),
                )
        return preset_id

    def get(self, preset_id: str) -> dict | None:
        """Get preset by ID."""
        with db_manager.get_connection() as conn:
            row = conn.execute("SELECT * FROM presets WHERE id = ?", (preset_id,)).fetchone()
            if not row:
                return None
            return self._row_to_preset(row)

    def list_by_model(self, model_id: str) -> list[dict]:
        """List presets for a model."""
        with db_manager.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM presets WHERE model_id = ? ORDER BY created_at DESC", (model_id,)
            ).fetchall()
            return [self._row_to_preset(row) for row in rows]

    def get_by_model_and_profile(self, model_id: str, profile: str) -> dict | None:
        """Get preset by model ID and profile."""
        with db_manager.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM presets WHERE model_id = ? AND profile = ? ORDER BY created_at DESC LIMIT 1",
                (model_id, profile),
            ).fetchone()
            if not row:
                return None
            return self._row_to_preset(row)

    def list_all(self) -> list[dict]:
        """List all presets."""
        with db_manager.get_connection() as conn:
            rows = conn.execute("SELECT * FROM presets ORDER BY created_at DESC").fetchall()
            return [self._row_to_preset(row) for row in rows]

    def delete(self, preset_id: str) -> bool:
        """Delete preset."""
        with db_manager.get_connection() as conn:
            cursor = conn.execute("DELETE FROM presets WHERE id = ?", (preset_id,))
            return cursor.rowcount > 0

    def _row_to_preset(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "model_id": row["model_id"],
            "profile": row["profile"],
            "name": row["name"],
            "config": json.loads(row["config_json"]),
            "metrics": json.loads(row["metrics_json"]) if row["metrics_json"] else None,
            "quality": json.loads(row["quality_json"]) if row["quality_json"] else None,
            "run_id": row["run_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "optimizer_version": row["optimizer_version"],
        }


class SettingsRepository:
    """Repository for application settings."""

    def get(self, key: str, default: str | None = None) -> str | None:
        """Get setting value."""
        with db_manager.get_connection() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else default

    def set(self, key: str, value: str, description: str = "") -> None:
        """Set setting value."""
        with db_manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO settings (key, value, description)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = CURRENT_TIMESTAMP
            """,
                (key, value, description, value),
            )

    def get_all(self) -> dict:
        """Get all settings."""
        with db_manager.get_connection() as conn:
            rows = conn.execute("SELECT key, value, description FROM settings").fetchall()
            return {
                row["key"]: {"value": row["value"], "description": row["description"]}
                for row in rows
            }


# Global repositories
hardware_repo = HardwareRepository()
model_repo = ModelRepository()
run_repo = RunRepository()
config_repo = ConfigurationRepository()
preset_repo = PresetRepository()
settings_repo = SettingsRepository()
