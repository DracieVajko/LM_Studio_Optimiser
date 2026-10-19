"""AI Model Judge - evaluates optimization results and provides recommendations.

This is an optional second-opinion tool. The USER always decides first.
The judge provides analysis to help the user make informed decisions.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any

import httpx

from lm_optimizer.domain.models import OptimizationRun, ConfigurationResult
from lm_optimizer.services.judge_prompt import PromptBuilder, JudgeRequest


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
    # Added by service after evaluation
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        d = asdict(self)
        return d


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

        # Helper to get quality
        def get_quality(c):
            return c.quality_score.overall if c.quality_score else 0

        # Score based on priority
        if comparison.priority == "speed":
            best = max(passed, key=lambda c: c.get_avg_generation_tok_s() or 0)
            reasoning = "Highest generation speed among passing configs"
        elif comparison.priority == "quality":
            best = max(passed, key=lambda c: get_quality(c))
            reasoning = "Highest quality score among passing configs"
        elif comparison.priority == "context":
            best = max(passed, key=lambda c: c.context_length)
            reasoning = "Longest context among passing configs"
        else:  # balanced
            # Weighted score: speed * quality
            def balanced_score(c):
                speed = c.get_avg_generation_tok_s() or 0
                qual = get_quality(c)
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
                    best = max(within_limit, key=lambda c: c.get_avg_generation_tok_s() or 0)
                elif comparison.priority == "quality":
                    best = max(within_limit, key=lambda c: get_quality(c))
                else:
                    best = max(within_limit, key=lambda c: (c.get_avg_generation_tok_s() or 0) * get_quality(c))
                warnings.append(f"Auto-selected alternative within VRAM limit: {best.context_length} ctx")

        alternatives = [c.id for c in passed if c.id != best.id][:3]
        best_speed = best.get_avg_generation_tok_s()
        best_qual = get_quality(best)

        return JudgeVerdict(
            recommended_config_id=best.id,
            reasoning=reasoning,
            confidence=0.85,
            pros=[
                f"Speed: {best_speed:.1f} tok/s" if best_speed else "Speed: N/A",
                f"Quality: {best_qual:.3f}" if best_qual else "Quality: N/A",
                f"Context: {best.context_length}",
                f"VRAM: {best.peak_vram_gb:.1f}GB" if best.peak_vram_gb else "VRAM: N/A",
            ],
            cons=warnings,
            alternative_config_ids=alternatives,
            warnings=warnings,
        )

    async def close(self):
        pass


class LocalJudge(ModelJudge):
    """Judge using a local model via LM Studio."""

    def __init__(self, config: JudgeConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url="http://127.0.0.1:1234/v1",
            timeout=60.0,
        )

    def _build_prompt(self, comparison: ModelComparison) -> str:
        """Build prompt for local judge using PromptBuilder."""
        from lm_optimizer.services.judge_prompt import RunData, JudgeRequest

        configs = []
        for cfg in comparison.configs:
            # Extract metrics and quality from ConfigurationResult
            metrics = []
            if hasattr(cfg, 'metrics') and cfg.metrics:
                for m in cfg.metrics:
                    metrics.append({
                        "test_name": m.get("test_name", "unknown"),
                        "category": m.get("category", "unknown"),
                        "success": m.get("success", True),
                        "generation_tok_s": m.get("generation_tok_s"),
                        "prompt_tok_s": m.get("prompt_tok_s"),
                        "estimated_ttft_ms": m.get("estimated_ttft_ms"),
                        "prompt_text": m.get("prompt_text", ""),
                        "output_text": m.get("output_text", ""),
                        "thinking_text": m.get("thinking_text", ""),
                    })

            cfg_dict = {
                "id": cfg.id,
                "config_json": json.dumps({
                    "context_length": cfg.context_length,
                    "gpu_ratio": getattr(cfg.config, 'gpu_ratio', 0.8),
                    "eval_batch_size": getattr(cfg.config, 'eval_batch_size', 256),
                    "flash_attention": getattr(cfg.config, 'flash_attention', True),
                    "offload_kv_cache_to_gpu": getattr(cfg.config, 'offload_kv_cache_to_gpu', True),
                }),
                "metrics_json": json.dumps(metrics),
                "quality_json": json.dumps({
                    "overall": cfg.quality_score.overall if cfg.quality_score else 0,
                    "checks_passed": cfg.quality_score.checks_passed if cfg.quality_score else 0,
                    "checks_total": cfg.quality_score.checks_total if cfg.quality_score else 0,
                    "by_test": getattr(cfg.quality_score, 'by_test', {}) if cfg.quality_score else {},
                }),
                "stability_score": cfg.stability_score,
                "peak_vram_gb": cfg.peak_vram_gb,
                "peak_ram_gb": cfg.peak_ram_gb,
                "score": getattr(cfg, 'score', None),
                "status": cfg.status,
            }
            configs.append(cfg_dict)

        run_data = [RunData(
            run_id="local",
            model_id=comparison.hardware_constraints.get("model_id", "unknown"),
            quality_threshold=0.95,
            configs=configs,
        )]

        request = JudgeRequest(
            run_ids=[],
            use_case=comparison.use_case,
            priority=comparison.priority,
            backend="local",
            judge_model=self.config.model or "local",
            top_n_detail=3,
            include_reasoning_matrix=True,
            fastest_context_preset=False,
        )
        return PromptBuilder.build(run_data, request)

    async def evaluate(self, comparison: ModelComparison) -> JudgeVerdict:
        prompt = self._build_prompt(comparison)

        # Prepare request payload
        payload = {
            "model": self.config.model or "local",
            "messages": [
                {"role": "system", "content": "You are an expert LLM optimization analyst. Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }

        max_retries = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                response = await self.client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()

                content = data["choices"][0]["message"]["content"]
                verdict_data = json.loads(content)

                return JudgeVerdict(
                    recommended_config_id=verdict_data.get("recommended_config_id", ""),
                    reasoning=verdict_data.get("reasoning", ""),
                    confidence=float(verdict_data.get("confidence", 0.5)),
                    pros=verdict_data.get("pros", []),
                    cons=verdict_data.get("cons", []),
                    alternative_config_ids=verdict_data.get("alternative_config_ids", []),
                    warnings=verdict_data.get("warnings", []),
                )

            except json.JSONDecodeError as e:
                last_error = e
                # Retry with stricter prompt
                payload["messages"][0]["content"] += " Return ONLY valid JSON, no extra text."
                continue
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise RuntimeError(
                        f"Model '{self.config.model}' not found in LM Studio. "
                        f"Load the model in LM Studio first, or use a different backend."
                    )
                raise
            except httpx.ConnectError as e:
                raise RuntimeError(
                    "Cannot connect to LM Studio at http://127.0.0.1:1234. "
                    "Ensure LM Studio is running and the API server is enabled."
                )

        # If we exhausted retries
        raise RuntimeError(f"Failed to get valid JSON from local judge after {max_retries} attempts: {last_error}")

    async def close(self):
        await self.client.aclose()


class APIJudge(ModelJudge):
    """Judge using an external API (OpenRouter-compatible)."""

    def __init__(self, config: JudgeConfig):
        self.config = config
        base = config.api_base or "https://openrouter.ai/api/v1"
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        self.client = httpx.AsyncClient(
            base_url=base,
            timeout=60.0,
            headers=headers,
        )

    def _build_prompt(self, comparison: ModelComparison) -> str:
        """Build prompt for API judge."""
        from lm_optimizer.services.judge_prompt import RunData, JudgeRequest

        configs = []
        for cfg in comparison.configs:
            metrics = []
            if hasattr(cfg, 'metrics') and cfg.metrics:
                for m in cfg.metrics:
                    metrics.append({
                        "test_name": m.get("test_name", "unknown"),
                        "category": m.get("category", "unknown"),
                        "success": m.get("success", True),
                        "generation_tok_s": m.get("generation_tok_s"),
                        "prompt_tok_s": m.get("prompt_tok_s"),
                        "estimated_ttft_ms": m.get("estimated_ttft_ms"),
                        "prompt_text": m.get("prompt_text", ""),
                        "output_text": m.get("output_text", ""),
                        "thinking_text": m.get("thinking_text", ""),
                    })

            cfg_dict = {
                "id": cfg.id,
                "config_json": json.dumps({
                    "context_length": cfg.context_length,
                    "gpu_ratio": getattr(cfg.config, 'gpu_ratio', 0.8),
                    "eval_batch_size": getattr(cfg.config, 'eval_batch_size', 256),
                    "flash_attention": getattr(cfg.config, 'flash_attention', True),
                    "offload_kv_cache_to_gpu": getattr(cfg.config, 'offload_kv_cache_to_gpu', True),
                }),
                "metrics_json": json.dumps(metrics),
                "quality_json": json.dumps({
                    "overall": cfg.quality_score.overall if cfg.quality_score else 0,
                    "checks_passed": cfg.quality_score.checks_passed if cfg.quality_score else 0,
                    "checks_total": cfg.quality_score.checks_total if cfg.quality_score else 0,
                    "by_test": getattr(cfg.quality_score, 'by_test', {}) if cfg.quality_score else {},
                }),
                "stability_score": cfg.stability_score,
                "peak_vram_gb": cfg.peak_vram_gb,
                "peak_ram_gb": cfg.peak_ram_gb,
                "score": getattr(cfg, 'score', None),
                "status": cfg.status,
            }
            configs.append(cfg_dict)

        run_data = [RunData(
            run_id="api",
            model_id=comparison.hardware_constraints.get("model_id", "unknown"),
            quality_threshold=0.95,
            configs=configs,
        )]

        request = JudgeRequest(
            run_ids=[],
            use_case=comparison.use_case,
            priority=comparison.priority,
            backend="api",
            judge_model=self.config.model,
            top_n_detail=3,
            include_reasoning_matrix=True,
            fastest_context_preset=False,
        )
        return PromptBuilder.build(run_data, request)

    async def evaluate(self, comparison: ModelComparison) -> JudgeVerdict:
        if not self.config.api_key:
            raise ValueError("API key required for API backend")

        prompt = self._build_prompt(comparison)

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "You are an expert LLM optimization analyst. Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = await self.client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()

                content = data["choices"][0]["message"]["content"]
                verdict_data = json.loads(content)

                return JudgeVerdict(
                    recommended_config_id=verdict_data.get("recommended_config_id", ""),
                    reasoning=verdict_data.get("reasoning", ""),
                    confidence=float(verdict_data.get("confidence", 0.5)),
                    pros=verdict_data.get("pros", []),
                    cons=verdict_data.get("cons", []),
                    alternative_config_ids=verdict_data.get("alternative_config_ids", []),
                    warnings=verdict_data.get("warnings", []),
                )

            except json.JSONDecodeError:
                payload["messages"][0]["content"] += " Return ONLY valid JSON, no extra text."
                continue
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise RuntimeError("Invalid API key for OpenRouter")
                raise

        raise RuntimeError(f"Failed to get valid JSON from API judge after {max_retries} attempts")

    async def close(self):
        await self.client.aclose()


def create_judge(config: JudgeConfig) -> ModelJudge:
    """Factory function to create a judge based on config."""
    if config.backend == JudgeBackend.HEURISTIC:
        return HeuristicJudge()
    elif config.backend == JudgeBackend.LOCAL:
        return LocalJudge(config)
    elif config.backend == JudgeBackend.API:
        return APIJudge(config)
    return HeuristicJudge()