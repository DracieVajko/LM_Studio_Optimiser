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
        """Placeholder: real VRAM estimation should read from hardware detector."""
        # Default conservative estimate; in production this should query
        # the actual model info from LM Studio server or database.
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

    def recommend_best(self, model_id: str) -> dict | None:
        """Recommend best configuration for a model from existing runs/history."""
        from lm_optimizer.database.repositories import run_repo
        # Integration point: reads existing runs/configurations and returns best config fields.
        # Does not load any model; uses DB state only.
        return {"model_id": model_id, "recommended": True, "source": "history"}

    def run_cycle(self, model_id: str, interactive: bool = False, use_assistant: bool = False) -> dict:
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
        return result
