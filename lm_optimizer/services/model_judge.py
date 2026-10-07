"""AI Model Judge - evaluates optimization results and provides recommendations.

This is an optional second-opinion tool. The USER always decides first.
The judge provides analysis to help the user make informed decisions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from lm_optimizer.domain.models import OptimizationRun, ConfigurationResult


class JudgeBackend(str, Enum):
    """Available judge backends."""
    LOCAL = "local"  # Use a local model via LM Studio/Ollama
    API = "api"      # Use external API (OpenRouter, etc.)
    HEURISTIC = "heuristic"  # Rule-based fallback (no AI)


@dataclass
class JudgeConfig:
    """Configuration for the model judge."""
    backend: JudgeBackend = JudgeBackend.HEURISTIC
    model: str | None = None  # Model to use for judging (local or API)
    api_key: str | None = None  # For API backend
    api_base: str | None = None  # For API backend
    temperature: float = 0.1
    max_tokens: int = 2048


@dataclass
class ModelComparison:
    """Comparison input for the judge."""
    run: OptimizationRun
    configs: list[ConfigurationResult]
    use_case: str  # e.g., "coding", "chat", "reasoning", "general"
    hardware_constraints: dict[str, Any]  # vram_gb, ram_gb, etc.
    priority: str  # "speed", "quality", "balanced", "context"


@dataclass
class JudgeVerdict:
    """Judge's recommendation."""
    recommended_config_id: str
    reasoning: str
    confidence: float  # 0.0 - 1.0
    pros: list[str]
    cons: list[str]
    alternative_config_ids: list[str]
    warnings: list[str]


class ModelJudge(ABC):
    """Abstract base class for model judges."""

    @abstractmethod
    async def evaluate(self, comparison: ModelComparison) -> JudgeVerdict:
        """Evaluate configurations and return a verdict."""
        pass

    @abstractmethod
    async def close(self):
        """Clean up resources."""
        pass


class HeuristicJudge(ModelJudge):
    """Rule-based judge - no AI required. Good fallback."""

    async def evaluate(self, comparison: ModelComparison) -> JudgeVerdict:
        configs = comparison.configs
        if not configs:
            return JudgeVerdict(
                recommended_config_id="",
                reasoning="No configurations to evaluate",
                confidence=0.0,
                pros=[],
                cons=[],
                alternative_config_ids=[],
                warnings=["No valid configurations found"],
            )

        # Filter passed configs
        passed = [c for c in configs if c.status == "passed"]
        if not passed:
            return JudgeVerdict(
                recommended_config_id="",
                reasoning="No configurations passed quality threshold",
                confidence=0.0,
                pros=[],
                cons=[],
                alternative_config_ids=[],
                warnings=["All configurations failed quality checks"],
            )

        # Score based on priority
        if comparison.priority == "speed":
            best = max(passed, key=lambda c: c.avg_generation_tok_s or 0)
            reasoning = "Highest generation speed among passing configs"
        elif comparison.priority == "quality":
            best = max(passed, key=lambda c: c.quality.overall if c.quality else 0)
            reasoning = "Highest quality score among passing configs"
        elif comparison.priority == "context":
            best = max(passed, key=lambda c: c.context_length)
            reasoning = "Longest context among passing configs"
        else:  # balanced
            # Weighted score: speed * quality
            def balanced_score(c):
                speed = c.avg_generation_tok_s or 0
                qual = c.quality.overall if c.quality else 0
                return speed * qual
            best = max(passed, key=balanced_score)
            reasoning = "Best speed×quality balance among passing configs"

        # Check VRAM constraint
        vram_limit = comparison.hardware_constraints.get("vram_gb")
        warnings = []
        if vram_limit and best.peak_vram_gb and best.peak_vram_gb > vram_limit:
            warnings.append(f"Selected config uses {best.peak_vram_gb:.1f}GB VRAM, limit is {vram_limit}GB")
            # Find best within limit
            within_limit = [c for c in passed if c.peak_vram_gb and c.peak_vram_gb <= vram_limit]
            if within_limit:
                if comparison.priority == "speed":
                    best = max(within_limit, key=lambda c: c.avg_generation_tok_s or 0)
                elif comparison.priority == "quality":
                    best = max(within_limit, key=lambda c: c.quality.overall if c.quality else 0)
                else:
                    best = max(within_limit, key=lambda c: (c.avg_generation_tok_s or 0) * (c.quality.overall if c.quality else 0))
                warnings.append(f"Auto-selected alternative within VRAM limit: {best.context_length} ctx")

        alternatives = [c.id for c in passed if c.id != best.id][:3]

        return JudgeVerdict(
            recommended_config_id=best.id,
            reasoning=reasoning,
            confidence=0.85,
            pros=[
                f"Speed: {best.avg_generation_tok_s:.1f} tok/s" if best.avg_generation_tok_s else "Speed: N/A",
                f"Quality: {best.quality.overall:.3f}" if best.quality else "Quality: N/A",
                f"Context: {best.context_length}",
                f"VRAM: {best.peak_vram_gb:.1f}GB" if best.peak_vram_gb else "VRAM: N/A",
            ],
            cons=warnings,
            alternative_config_ids=alternatives,
            warnings=warnings,
        )

    async def close(self):
        pass


def create_judge(config: JudgeConfig) -> ModelJudge:
    """Factory function to create a judge based on config."""
    if config.backend == JudgeBackend.HEURISTIC:
        return HeuristicJudge()
    elif config.backend == JudgeBackend.LOCAL:
        # TODO: Implement local model judge
        return HeuristicJudge()  # Fallback for now
    elif config.backend == JudgeBackend.API:
        # TODO: Implement API judge (OpenRouter, etc.)
        return HeuristicJudge()  # Fallback for now
    return HeuristicJudge()