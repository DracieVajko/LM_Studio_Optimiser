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
