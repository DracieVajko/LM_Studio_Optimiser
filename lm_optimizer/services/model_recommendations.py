"""Model parameter recommendations from Hugging Face and LM Studio."""

import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from lm_optimizer.config import config
from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationParameters:
    """Recommended generation parameters for a model."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    typical_p: float | None = None
    mirostat_mode: int | None = None
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None
    seed: int | None = None
    stop_sequences: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != []}

    def to_api_params(self) -> dict[str, Any]:
        """Convert to API parameters for chat completion."""
        return self.to_dict()


@dataclass
class ModelRecommendation:
    """Complete model recommendation including generation and load parameters."""

    model_id: str
    model_name: str
    generation_params: GenerationParameters = field(default_factory=GenerationParameters)
    load_params: dict[str, Any] = field(default_factory=dict)
    source: str = "default"  # "huggingface", "lm_studio", "default", "heuristic"
    confidence: float = 0.5
    notes: list[str] = field(default_factory=list)


class ModelRecommendationService:
    """Fetch and generate model parameter recommendations."""

    # Default generation parameters by model architecture/type
    ARCH_DEFAULTS = {
        "llama": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            min_p=0.05,
        ),
        "mistral": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.05,
            min_p=0.05,
        ),
        "mixtral": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            min_p=0.05,
        ),
        "qwen": GenerationParameters(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.05,
            min_p=0.0,
        ),
        "qwen2": GenerationParameters(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.05,
            min_p=0.0,
        ),
        "phi": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            min_p=0.05,
        ),
        "phi3": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            min_p=0.05,
        ),
        "gemma": GenerationParameters(
            temperature=0.7,
            top_p=0.95,
            top_k=64,
            repetition_penalty=1.0,
            min_p=0.0,
        ),
        "gemma2": GenerationParameters(
            temperature=0.7,
            top_p=0.95,
            top_k=64,
            repetition_penalty=1.0,
            min_p=0.0,
        ),
        "deepseek": GenerationParameters(
            temperature=0.6,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.05,
            min_p=0.05,
        ),
        "deepseek_v2": GenerationParameters(
            temperature=0.6,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.05,
            min_p=0.05,
        ),
        "deepseek_v3": GenerationParameters(
            temperature=0.6,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.05,
            min_p=0.05,
        ),
        "yi": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            min_p=0.05,
        ),
        "nemotron": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.05,
            min_p=0.05,
        ),
        "command-r": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.05,
            min_p=0.05,
        ),
        "starcoder": GenerationParameters(
            temperature=0.2,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.05,
            min_p=0.05,
        ),
        "codeqwen": GenerationParameters(
            temperature=0.2,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.05,
            min_p=0.05,
        ),
        "wizardcoder": GenerationParameters(
            temperature=0.2,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.05,
            min_p=0.05,
        ),
        "gpt2": GenerationParameters(
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            min_p=0.0,
        ),
        "gpt_neox": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            min_p=0.05,
        ),
        "falcon": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            min_p=0.05,
        ),
        "mpt": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            min_p=0.05,
        ),
        "bloom": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            min_p=0.05,
        ),
        "baichuan": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.05,
            min_p=0.05,
        ),
        "internlm": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.05,
            min_p=0.05,
        ),
        "xgen": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            min_p=0.05,
        ),
        "stablelm": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            min_p=0.05,
        ),
        "openchat": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            min_p=0.05,
        ),
        "zephyr": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            min_p=0.05,
        ),
        "dolphin": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            min_p=0.05,
        ),
        "orca": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            min_p=0.05,
        ),
        "vicuna": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            min_p=0.05,
        ),
        "alpaca": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            min_p=0.05,
        ),
    }

    # Task-specific overrides
    TASK_OVERRIDES = {
        "coding": GenerationParameters(
            temperature=0.1,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.05,
        ),
        "reasoning": GenerationParameters(
            temperature=0.3,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
        ),
        "creative": GenerationParameters(
            temperature=0.9,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.05,
        ),
        "factual": GenerationParameters(
            temperature=0.3,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
        ),
        "json": GenerationParameters(
            temperature=0.0,
            top_p=0.95,
            top_k=1,
            repetition_penalty=1.0,
        ),
        "chat": GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
        ),
    }

    def __init__(self):
        self._cache: dict[str, ModelRecommendation] = {}
        self._hf_client: httpx.AsyncClient | None = None

    async def get_recommendation(
        self,
        model_id: str,
        model_name: str,
        architecture: str | None = None,
        quantization: str | None = None,
        task: str = "chat",
    ) -> ModelRecommendation:
        """Get complete parameter recommendation for a model."""
        cache_key = f"{model_id}:{task}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try to fetch from Hugging Face
        hf_rec = await self._fetch_from_huggingface(model_id)
        if hf_rec:
            self._cache[cache_key] = hf_rec
            return hf_rec

        # Try to fetch from LM Studio model info (if available)
        # This would require LM Studio client integration

        # Fall back to architecture-based heuristics
        rec = self._generate_heuristic_recommendation(
            model_id, model_name, architecture, quantization, task
        )
        self._cache[cache_key] = rec
        return rec

    async def _fetch_from_huggingface(self, model_id: str) -> ModelRecommendation | None:
        """Fetch model card from Hugging Face and extract generation parameters."""
        # Extract repo ID from model_id (format varies)
        # Common formats: "user/model", "model", "user/model:quant"
        repo_id = self._extract_hf_repo_id(model_id)
        if not repo_id:
            return None

        try:
            if self._hf_client is None:
                self._hf_client = httpx.AsyncClient(timeout=10.0)

            # Fetch model card README
            url = f"https://huggingface.co/{repo_id}/raw/main/README.md"
            response = await self._hf_client.get(url)
            if response.status_code != 200:
                # Try alternative: model card data endpoint
                url = f"https://huggingface.co/api/models/{repo_id}"
                response = await self._hf_client.get(url)
                if response.status_code != 200:
                    return None
                data = response.json()
                return self._parse_hf_model_card_data(data, model_id)

            readme = response.text
            return self._parse_hf_readme(readme, model_id)

        except Exception as e:
            logger.debug("Failed to fetch from Hugging Face", model_id=model_id, error=str(e))
            return None

    def _extract_hf_repo_id(self, model_id: str) -> str | None:
        """Extract Hugging Face repo ID from model identifier."""
        # Remove quantization suffixes
        model_id = re.sub(r":(q4|q5|q6|q8|fp16|fp32|gguf|gptq|awq).*$", "", model_id, flags=re.IGNORECASE)

        # If it looks like a repo ID (contains /)
        if "/" in model_id and not model_id.startswith("http"):
            parts = model_id.split("/")
            if len(parts) >= 2:
                return "/".join(parts[:2])

        # Known model name mappings
        known_models = {
            "llama-2-7b": "meta-llama/Llama-2-7b-hf",
            "llama-2-13b": "meta-llama/Llama-2-13b-hf",
            "llama-2-70b": "meta-llama/Llama-2-70b-hf",
            "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
            "llama-3-70b": "meta-llama/Meta-Llama-3-70B",
            "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
            "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B",
            "mistral-7b": "mistralai/Mistral-7B-v0.1",
            "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",
            "mixtral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
            "mixtral-8x22b": "mistralai/Mixtral-8x22B-v0.1",
            "qwen-7b": "Qwen/Qwen-7B",
            "qwen-14b": "Qwen/Qwen-14B",
            "qwen-72b": "Qwen/Qwen-72B",
            "qwen2-7b": "Qwen/Qwen2-7B",
            "qwen2-72b": "Qwen/Qwen2-72B",
            "qwen2.5-7b": "Qwen/Qwen2.5-7B",
            "qwen2.5-72b": "Qwen/Qwen2.5-72B",
            "phi-2": "microsoft/phi-2",
            "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
            "phi-3-medium": "microsoft/Phi-3-medium-4k-instruct",
            "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
            "gemma-7b": "google/gemma-7b",
            "gemma-2b": "google/gemma-2b",
            "gemma2-9b": "google/gemma-2-9b",
            "gemma2-27b": "google/gemma-2-27b",
            "deepseek-coder-7b": "deepseek-ai/deepseek-coder-7b-base-v1.5",
            "deepseek-coder-33b": "deepseek-ai/deepseek-coder-33b-base",
            "deepseek-llm-7b": "deepseek-ai/deepseek-llm-7b-base",
            "deepseek-llm-67b": "deepseek-ai/deepseek-llm-67b-base",
            "deepseek-v2": "deepseek-ai/DeepSeek-V2",
            "deepseek-v3": "deepseek-ai/DeepSeek-V3",
            "yi-6b": "01-ai/Yi-6B",
            "yi-34b": "01-ai/Yi-34B",
            "nemotron-3-8b": "nvidia/Nemotron-3-8B",
            "starcoder2-7b": "bigcode/starcoder2-7b",
            "starcoder2-15b": "bigcode/starcoder2-15b",
            "command-r": "CohereForAI/c4ai-command-r-v01",
            "command-r-plus": "CohereForAI/c4ai-command-r-plus",
        }

        model_lower = model_id.lower()
        for key, repo in known_models.items():
            if key in model_lower:
                return repo

        return None

    def _parse_hf_model_card_data(self, data: dict, model_id: str) -> ModelRecommendation | None:
        """Parse Hugging Face model card data API response."""
        try:
            # Look for generation config in model card data
            card_data = data.get("cardData", {})
            if not card_data:
                return None

            gen_config = card_data.get("generation_config", {})
            if not gen_config:
                return None

            params = GenerationParameters(
                temperature=gen_config.get("temperature"),
                top_p=gen_config.get("top_p"),
                top_k=gen_config.get("top_k"),
                repetition_penalty=gen_config.get("repetition_penalty"),
                min_p=gen_config.get("min_p"),
                presence_penalty=gen_config.get("presence_penalty"),
                frequency_penalty=gen_config.get("frequency_penalty"),
                typical_p=gen_config.get("typical_p"),
                mirostat_mode=gen_config.get("mirostat_mode"),
                mirostat_tau=gen_config.get("mirostat_tau"),
                mirostat_eta=gen_config.get("mirostat_eta"),
                seed=gen_config.get("seed"),
                stop_sequences=gen_config.get("stop_sequences", []),
            )

            return ModelRecommendation(
                model_id=model_id,
                model_name=data.get("modelId", model_id),
                generation_params=params,
                source="huggingface",
                confidence=0.9,
                notes=["Fetched from Hugging Face model card generation_config"],
            )
        except Exception as e:
            logger.debug("Failed to parse HF model card data", error=str(e))
            return None

    def _parse_hf_readme(self, readme: str, model_id: str) -> ModelRecommendation | None:
        """Parse Hugging Face README for generation parameters."""
        try:
            # Look for generation config in YAML frontmatter or code blocks
            params = GenerationParameters()
            found_any = False

            # Pattern 1: YAML frontmatter with generation_config
            yaml_match = re.search(
                r"generation_config:\s*\n((?:\s+\w+:\s*.+\n)+)", readme, re.IGNORECASE
            )
            if yaml_match:
                yaml_content = yaml_match.group(1)
                for line in yaml_content.split("\n"):
                    line = line.strip()
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if hasattr(params, key):
                            try:
                                if key in ("top_k", "mirostat_mode", "seed"):
                                    setattr(params, key, int(value))
                                elif key == "stop_sequences":
                                    # Parse as list
                                    setattr(params, key, [s.strip() for s in value.strip("[]").split(",")])
                                else:
                                    setattr(params, key, float(value))
                                found_any = True
                            except ValueError:
                                pass

            # Pattern 2: Python code block with GenerationConfig
            if not found_any:
                python_blocks = re.findall(r"```python\n(.*?)\n```", readme, re.DOTALL)
                for block in python_blocks:
                    if "GenerationConfig" in block or "generation_config" in block.lower():
                        # Extract parameters from Python code
                        for key in [
                            "temperature",
                            "top_p",
                            "top_k",
                            "repetition_penalty",
                            "min_p",
                            "presence_penalty",
                            "frequency_penalty",
                        ]:
                            match = re.search(rf"{key}\s*=\s*([\d.]+)", block)
                            if match:
                                try:
                                    val = float(match.group(1))
                                    if key == "top_k":
                                        val = int(val)
                                    setattr(params, key, val)
                                    found_any = True
                                except ValueError:
                                    pass

            # Pattern 3: JSON code block
            if not found_any:
                json_blocks = re.findall(r"```json\n(.*?)\n```", readme, re.DOTALL)
                for block in json_blocks:
                    try:
                        import json
                        data = json.loads(block)
                        if "generation_config" in data:
                            gen_config = data["generation_config"]
                            for key in [
                                "temperature",
                                "top_p",
                                "top_k",
                                "repetition_penalty",
                                "min_p",
                                "presence_penalty",
                                "frequency_penalty",
                            ]:
                                if key in gen_config:
                                    val = gen_config[key]
                                    if key == "top_k":
                                        val = int(val)
                                    setattr(params, key, val)
                                    found_any = True
                    except Exception:
                        pass

            if found_any:
                return ModelRecommendation(
                    model_id=model_id,
                    model_name=model_id,
                    generation_params=params,
                    source="huggingface",
                    confidence=0.8,
                    notes=["Parsed from Hugging Face README"],
                )

        except Exception as e:
            logger.debug("Failed to parse HF README", error=str(e))

        return None

    def _generate_heuristic_recommendation(
        self,
        model_id: str,
        model_name: str,
        architecture: str | None,
        quantization: str | None,
        task: str,
    ) -> ModelRecommendation:
        """Generate recommendation based on architecture heuristics."""
        arch_lower = (architecture or model_id or model_name or "").lower()

        # Find matching architecture
        base_params = GenerationParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            min_p=0.05,
        )

        for arch_key, arch_params in self.ARCH_DEFAULTS.items():
            if arch_key in arch_lower:
                base_params = arch_params
                break

        # Apply task-specific overrides
        task_override = self.TASK_OVERRIDES.get(task, GenerationParameters())
        for key, value in task_override.to_dict().items():
            if value is not None:
                setattr(base_params, key, value)

        # Quantization-specific adjustments
        if quantization:
            quant_lower = quantization.lower()
            if "q4" in quant_lower or "4bit" in quant_lower:
                # Lower temp for more quantized models
                if base_params.temperature and base_params.temperature > 0.5:
                    base_params.temperature = max(0.3, base_params.temperature - 0.2)
            elif "q8" in quant_lower or "8bit" in quant_lower or "fp16" in quant_lower:
                # Can use slightly higher temp
                pass

        confidence = 0.6 if architecture else 0.4
        notes = [f"Heuristic based on architecture: {architecture or 'unknown'}"]

        return ModelRecommendation(
            model_id=model_id,
            model_name=model_name,
            generation_params=base_params,
            source="heuristic",
            confidence=confidence,
            notes=notes,
        )

    async def close(self):
        """Close HTTP client."""
        if self._hf_client:
            await self._hf_client.aclose()
            self._hf_client = None


# Global instance
model_recommendation_service = ModelRecommendationService()