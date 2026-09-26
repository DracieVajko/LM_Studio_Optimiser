"""Checkpoint and result storage."""

from datetime import datetime
from pathlib import Path

import orjson

from lm_optimizer.config import config
from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)

# Type hint for forward reference
if False:  # TYPE_CHECKING
    from lm_optimizer.optimizer.engine import OptimizationState


class CheckpointManager:
    """Manages optimization checkpoints."""

    def __init__(self):
        self.checkpoint_dir = config.storage.results_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, state: "OptimizationState") -> Path:
        """Save optimization state checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{state.model_id.replace('/', '_')}_{state.profile.name}_{timestamp}.json"
        filepath = self.checkpoint_dir / filename

        # Convert state to serializable dict
        data = {
            "model_id": state.model_id,
            "profile": state.profile.name,
            "stage": state.stage.value,
            "start_time": state.start_time.isoformat(),
            "timestamp": datetime.now().isoformat(),
            "capabilities": self._serialize_capabilities(state.capabilities),
            "tested_configs": [self._serialize_result(r) for r in state.tested_configs],
            "best_config": self._serialize_result(state.best_config) if state.best_config else None,
            "pareto_frontier": [self._serialize_result(r) for r in state.pareto_frontier],
            "errors": state.errors,
        }

        try:
            filepath.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
            logger.debug("Checkpoint saved", path=str(filepath))
        except Exception as e:
            logger.error("Failed to save checkpoint", error=str(e))

        return filepath

    def _serialize_capabilities(self, caps) -> dict | None:
        if not caps:
            return None
        return {
            "model_id": caps.model_id,
            "architecture": caps.architecture,
            "context_length": caps.context_length,
            "max_context_length": caps.max_context_length,
            "parameter_count": caps.parameter_count,
            "quantization": caps.quantization,
            "is_moe": caps.is_moe,
            "num_experts": caps.num_experts,
            "supported_parameters": caps.supported_parameters,
            "estimated_vram_gb": caps.estimated_vram_gb,
        }

    def _serialize_result(self, result) -> dict | None:
        if not result:
            return None
        # Use estimated_ttft consistently, with fallback
        try:
            ettft = (
                result.get_avg_estimated_ttft_ms()
                if hasattr(result, "get_avg_estimated_ttft_ms")
                else result.get_avg_ttft_ms()
            )
        except Exception:
            ettft = result.get_avg_ttft_ms() if hasattr(result, "get_avg_ttft_ms") else 0
        return {
            "config_id": result.config_id,
            "model_id": result.model_id,
            "context_length": result.context_length,
            "gpu_ratio": result.gpu_ratio,
            "flash_attention": result.flash_attention,
            "kv_cache_gpu": result.kv_cache_gpu,
            "eval_batch_size": result.eval_batch_size,
            "num_experts": result.num_experts,
            "timestamp": result.timestamp.isoformat(),
            "generation_tok_s": result.get_avg_generation_tok_s(),
            "prompt_tok_s": result.get_avg_prompt_tok_s(),
            "estimated_ttft_ms": ettft,
            "ttft_ms": ettft,  # compat
            "quality_score": result.quality_score,
            "stability_score": result.stability_score,
            "passed": result.passed,
            "failure_reason": result.failure_reason,
            "load_time_ms": result.metrics[0].load_time_ms if result.metrics else 0,
        }


class PresetStorage:
    """Manages optimized presets for models."""

    def __init__(self):
        self.presets_dir = config.storage.profiles_dir
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def save_preset(
        self,
        model_id: str,
        hardware_info: dict,
        profile_name: str,
        recommended_config: dict,
        metrics: dict,
    ) -> Path:
        """Save an optimized preset."""
        safe_model_id = model_id.replace("/", "_").replace(":", "_")
        filename = f"{safe_model_id}_{profile_name}.json"
        filepath = self.presets_dir / filename

        preset = {
            "model": model_id,
            "hardware": hardware_info,
            "profile": profile_name,
            "recommended_config": recommended_config,
            "metrics": metrics,
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
        }

        try:
            filepath.write_bytes(orjson.dumps(preset, option=orjson.OPT_INDENT_2))
            logger.info("Preset saved", model=model_id, profile=profile_name, path=str(filepath))
        except Exception as e:
            logger.error("Failed to save preset", error=str(e))
            raise

        return filepath

    def load_preset(self, model_id: str, profile_name: str) -> dict | None:
        """Load a preset."""
        safe_model_id = model_id.replace("/", "_").replace(":", "_")
        filename = f"{safe_model_id}_{profile_name}.json"
        filepath = self.presets_dir / filename

        if not filepath.exists():
            return None

        try:
            return orjson.loads(filepath.read_bytes())
        except Exception as e:
            logger.error("Failed to load preset", error=str(e))
            return None

    def list_presets(self) -> list[dict]:
        """List all saved presets."""
        presets = []
        for filepath in self.presets_dir.glob("*.json"):
            try:
                data = orjson.loads(filepath.read_bytes())
                presets.append(
                    {
                        "model": data.get("model"),
                        "profile": data.get("profile"),
                        "created_at": data.get("created_at"),
                        "metrics": data.get("metrics", {}),
                    }
                )
            except Exception:
                continue
        return presets

    def delete_preset(self, model_id: str, profile_name: str) -> bool:
        """Delete a preset."""
        safe_model_id = model_id.replace("/", "_").replace(":", "_")
        filename = f"{safe_model_id}_{profile_name}.json"
        filepath = self.presets_dir / filename

        if filepath.exists():
            filepath.unlink()
            return True
        return False


class ResultsStorage:
    """Manages detailed benchmark results."""

    def __init__(self):
        self.results_dir = config.storage.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_result(self, result, run_id: str | None = None) -> Path:
        """Save detailed benchmark result."""
        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = result.model_id.replace("/", "_").replace(":", "_")
        filename = f"{safe_model}_{result.config_id}_{run_id}.json"
        filepath = self.results_dir / filename

        data = {
            "config_id": result.config_id,
            "model_id": result.model_id,
            "load_config": {
                "context_length": result.context_length,
                "gpu_ratio": result.gpu_ratio,
                "flash_attention": result.flash_attention,
                "offload_kv_cache_to_gpu": result.kv_cache_gpu,
                "eval_batch_size": result.eval_batch_size,
                "num_experts": result.num_experts,
            },
            "timestamp": result.timestamp.isoformat(),
            "metrics": [
                {
                    "test_name": m.test_name,
                    "category": m.category,
                    "success": m.success,
                    "load_time_ms": m.load_time_ms,
                    "estimated_ttft_ms": getattr(m, "estimated_ttft_ms", getattr(m, "ttft_ms", 0)),
                    "ttft_ms": getattr(m, "estimated_ttft_ms", getattr(m, "ttft_ms", 0)),  # compat
                    "prompt_tokens": m.prompt_tokens,
                    "completion_tokens": m.completion_tokens,
                    "total_tokens": m.total_tokens,
                    "prompt_processing_ms": m.prompt_processing_ms,
                    "generation_ms": m.generation_ms,
                    "prompt_tok_s": m.prompt_tok_s,
                    "generation_tok_s": m.generation_tok_s,
                    "output_text": m.output_text[:500] if m.output_text else "",
                    "error": m.error,
                }
                for m in result.metrics
            ],
            "peak_vram_gb": result.peak_vram_gb,
            "peak_ram_gb": result.peak_ram_gb,
            "stability_score": result.stability_score,
            "quality_score": result.quality_score,
            "passed": result.passed,
            "failure_reason": result.failure_reason,
        }

        try:
            filepath.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
            logger.debug("Result saved", path=str(filepath))
        except Exception as e:
            logger.error("Failed to save result", error=str(e))

        return filepath

    def load_result(self, filepath: Path) -> dict | None:
        """Load a result file."""
        try:
            return orjson.loads(filepath.read_bytes())
        except Exception as e:
            logger.error("Failed to load result", error=str(e))
            return None

    def list_results(self, model_id: str | None = None) -> list[dict]:
        """List all results, optionally filtered by model."""
        results = []
        pattern = f"{model_id.replace('/', '_').replace(':', '_')}_*" if model_id else "*.json"

        for filepath in self.results_dir.glob(pattern):
            try:
                data = orjson.loads(filepath.read_bytes())
                results.append(
                    {
                        "file": filepath.name,
                        "model_id": data.get("model_id"),
                        "config_id": data.get("config_id"),
                        "timestamp": data.get("timestamp"),
                        "passed": data.get("passed"),
                        "generation_tok_s": data.get("metrics", [{}])[0].get("generation_tok_s", 0)
                        if data.get("metrics")
                        else 0,
                        "quality_score": data.get("quality_score"),
                    }
                )
            except Exception:
                continue

        return sorted(results, key=lambda x: x["timestamp"], reverse=True)
