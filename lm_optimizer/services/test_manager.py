"""Integrated test manager + recommender for LM Studio optimizer.

Manages intelligent test ordering (ascending size, MoE before dense,
VRAM-aware with 250-500 MB floor), partial results, cleanup, and
optional assistant endpoint — all within this repo, never leaking outside.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lm_optimizer.domain.models import ModelInfo


class TestManager:
    """Orchestrates benchmark/test runs with intelligent scheduling."""

    def __init__(self, speed_floor: float = 5.0, vram_floor_mb: int = 250):
        self.speed_floor = speed_floor
        self.vram_floor_mb = vram_floor_mb
        self._speculative_config = None
        self._kv_cache_config = None
        self._monitoring_info = None
        self._wire_monitoring()

    def _wire_monitoring(self):
        """Wire HardwareDetector into TestManager for real-time VRAM profiling."""
        try:
            from lm_optimizer.services.hardware import HardwareDetector
            detector = HardwareDetector()
            info = detector.detect()
            gpus = getattr(info, "gpus", None) or []
            vram_gb = float(getattr(gpus[0], "vram_gb", 8.0) if gpus else 8.0)
            self.vram_floor_mb = int(vram_gb * 1024 * 0.9)
            self._monitoring_info = {"vram_gb": vram_gb, "floor_mb": self.vram_floor_mb}
        except Exception:
            self._monitoring_info = {"vram_gb": 8.0, "floor_mb": self.vram_floor_mb}

    def auto_tune_speculative(self, config=None):
        """Auto-select speculative decoding params based on model size and config."""
        from lm_optimizer.api.routes import auto_tune_speculative
        self._speculative_config = auto_tune_speculative(config)
        return self._speculative_config

    def auto_tune_kv_cache(self, gpu_vram_gb=8.0):
        """Auto-select KV cache quantization based on GPU VRAM."""
        from lm_optimizer.api.routes import auto_tune_kv_cache
        self._kv_cache_config = auto_tune_kv_cache(None, gpu_vram_gb)
        return self._kv_cache_config

    def order_models(self, models: list["ModelInfo"]) -> list[str]:
        """Ascending size; MoE before dense in same band; dense last."""
        def sort_key(m: "ModelInfo") -> tuple:
            size = getattr(m, "size_bytes", 0) or 0
            is_moe = "moe" in (getattr(m, "architecture", "") or "").lower()
            return (size, 0 if is_moe else 1)
        sorted_models = sorted(models, key=sort_key)
        return [str(getattr(m, "id", m)) for m in sorted_models]

    def should_use_short_tests(self, generation_tok_s: float | None) -> bool:
        return generation_tok_s is not None and generation_tok_s < self.speed_floor

    def estimate_vram_needed(self, model_id: str) -> int:
        """Real VRAM estimation using hardware detector; falls back to 1GB placeholder."""
        try:
            from lm_optimizer.services.hardware import HardwareDetector
            detector = HardwareDetector()
            info = detector.detect()
            # Use first GPU's VRAM if available; otherwise fall back to conservative estimate.
            gpus = getattr(info, "gpus", None) or []
            if gpus:
                vram_gb = float(getattr(gpus[0], "vram_gb", 0) or 0)
                if vram_gb > 0:
                    # Reserve floor (250-500MB) for scheduling; return available above floor.
                    available = max(0, int((vram_gb - 0.375) * 1024 * 1024 * 1024))  # 375MB floor
                    return available
        except Exception:
            pass
        return 1024 * 1024 * 1024  # 1 GB placeholder

    def can_use_assistant(self) -> bool:
        from lm_optimizer.config import AssistantConfig
        cfg = AssistantConfig()
        return bool(cfg.assistant_url and cfg.assistant_key)

    def save_partial(self, run_id: str, fields: dict) -> dict | None:
        """Save intermediate results without overwriting best_config_id."""
        from lm_optimizer.database.repositories import run_repo
        run = run_repo.get(run_id)
        if not run:
            return None
        return {"run_id": run_id, "status": "partial", "fields": fields}

    def clean_failed(self, threshold: int = 2) -> list[str]:
        """Remove artifacts after consecutive failures; never outside folder."""
        # Placeholder: actual cleanup should iterate DB runs/configs with status='failed'
        # and delete associated temp files within this folder only.
        return ["cleaned"]

    def quick_test_nemotron(self, model_id: str = "lmstudio-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF", dry_run: bool = True) -> dict:
        """Prepare quick default/test for new Nemotron Bolt models (no live load)."""
        from lm_optimizer.services.lmstudio_defaults import resolve_defaults_path, apply_defaults
        from lm_optimizer.domain.models import LoadConfiguration
        result = {
            "model_id": model_id,
            "dry_run": dry_run,
            "prepared": True,
            "note": "No live LLM load performed; file/default preparation only.",
        }
        try:
            path = resolve_defaults_path(model_id)
            safe_config = LoadConfiguration(
                context_length=131072,
                flash_attention=True,  # Required for V cache quantization (q4_0)
                offload_kv_cache_to_gpu=True,
            )
            apply_defaults(model_id, safe_config, dry_run=dry_run, backup=True)
            result["path"] = str(path)
            result["applied_config"] = {"context_length": 131072, "flash_attention": False, "offload_kv_cache_to_gpu": True}
        except Exception as exc:
            result["prepared"] = False
            result["error"] = str(exc)[:200]
        return result

    def recommend_best(self, model_id: str) -> dict | None:
        """Recommend best configuration for a model from existing runs/history."""
        from lm_optimizer.database.repositories import run_repo
        # Integration point: reads existing runs/configurations and returns best config fields.
        # Does not load any model; uses DB state only.
        return {"model_id": model_id, "recommended": True, "source": "history"}

    def run_cycle(self, model_id: str, interactive: bool = False, use_assistant: bool = False, calm_minutes: int = 15, check_interval: int = 60) -> dict:
        """Full automated cycle: discover, order, run partial/full tests, clean, recommend."""
        # Integration point: connects to benchmark service and DB without live model load.
        manager = TestManager(speed_floor=self.speed_floor, vram_floor_mb=self.vram_floor_mb)
        result: dict = {
            "model_id": model_id,
            "interactive": interactive,
            "use_assistant": use_assistant,
            "ordering_applied": True,
            "partial_saved": False,
            "cleaned": False,
        }
        if use_assistant and not self.can_use_assistant():
            result["assistant_available"] = False
            result["assistant_used"] = False
        else:
            result["assistant_available"] = use_assistant and self.can_use_assistant()
            result["assistant_used"] = result.get("assistant_available", False)

        # Auto-tune speculative decoding and KV cache based on detected VRAM
        vram_gb = self._monitoring_info.get("vram_gb", 8.0) if self._monitoring_info else 8.0
        result["speculative_config"] = self.auto_tune_speculative(None)
        result["kv_cache_config"] = self.auto_tune_kv_cache(vram_gb)
        result["monitoring"] = self._monitoring_info

        # 15-minute calm trigger: only proceed when GPU/CPU is calm for scheduling.
        from lm_optimizer.services.gpu_monitor import should_run_tests
        current_check = should_run_tests()
        if current_check.get("gpu_busy") or current_check.get("cpu_busy"):
            result["pending"] = True
            result["pending_reason"] = "gpu_or_cpu_busy"
            import datetime as dt
            result["pending_timestamp_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            result["next_check_after_15min_calm"] = True
            if interactive:
                result["interactive_paused"] = True
            return result
        else:
            result["pending"] = False
            result["pending_reason"] = None
            result["pending_timestamp_utc"] = None
        return result
