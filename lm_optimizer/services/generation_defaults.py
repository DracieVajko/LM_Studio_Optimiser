"""Sourced generation defaults per model lineage.

Every entry cites its source. Community fine-tunes without official settings
inherit the closest official base-lineage entry (marked as lineage-inferred).
Benchmarks intentionally use lower/deterministic temps; the .md reports BOTH.
"""

from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)

# Ordered specific -> general. `match`: substrings of the model key (lowercase).
# `arch`: LM Studio architecture strings.
DEFAULTS = [
    {
        "match": ["qwen3.5"],
        "arch": ["qwen35"],
        "lineage": "Qwen3.5 instruct",
        "settings": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "reasoning": "off for benchmarks",
        },
        "source": "Qwen3.5 official (HF discussion #51, instruct-general)",
    },
    {
        "match": ["qwen3.8", "qwen3", "tongyi", "marco"],
        "arch": ["qwen3", "qwen3moe"],
        "lineage": "Qwen3 (lineage-inferred for community names)",
        "settings": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "reasoning": "off for benchmarks",
        },
        "source": "Qwen3 official generation_config.json (non-thinking 0.7/0.8/20/0)",
    },
    {
        "match": ["deepseek", "fin-r1", "fin_r1"],
        "arch": ["qwen2"],
        "lineage": "DeepSeek-R1-Distill (Qwen2.5-based)",
        "settings": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": None,
            "min_p": None,
            "presence_penalty": None,
            "reasoning": "on (reasoning model; benchmarks run off)",
        },
        "source": "DeepSeek-R1-Distill official (0.6/0.95, no system prompt)",
    },
    {
        "match": ["ministral", "mistral"],
        "arch": ["mistral3"],
        "lineage": "Mistral Ministral",
        "settings": {
            "temperature": 0.3,
            "top_p": None,
            "top_k": None,
            "min_p": None,
            "presence_penalty": None,
            "reasoning": "off for benchmarks",
        },
        "source": "Mistral official docs (0.0-0.7 range; 0.2 focused/deterministic)",
    },
    {
        "match": ["gemma"],
        "arch": ["gemma4", "gemma3", "gemma"],
        "lineage": "Gemma (GGUF)",
        "settings": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "min_p": 0.01,
            "presence_penalty": None,
            "reasoning": "off for benchmarks",
        },
        "source": "Gemma team via HF (1.0/0.95) + Unsloth GGUF defaults (top_k 64)",
    },
    {
        "match": ["gpt-oss", "gpt_oss"],
        "arch": ["gpt-oss", "gpt_oss"],
        "lineage": "OpenAI gpt-oss",
        "settings": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "min_p": None,
            "presence_penalty": None,
            "reasoning": "medium default (harmony format required)",
        },
        "source": "OpenAI gpt-oss official README (1.0/1.0, harmony format)",
    },
]

GENERIC = {
    "lineage": "generic fallback",
    "settings": {
        "temperature": 0.3,
        "top_p": None,
        "top_k": None,
        "min_p": None,
        "presence_penalty": None,
        "reasoning": "off for benchmarks",
    },
    "source": "generic fallback (no lineage match; benchmark-suite defaults)",
}


def lookup(model_id: str, architecture: str | None = None) -> dict:
    """Return {lineage, settings, source} for a model key (+ optional arch)."""
    key = (model_id or "").lower()
    arch = (architecture or "").lower()
    for entry in DEFAULTS:
        if any(m in key for m in entry["match"]) or (
            arch and arch in entry["arch"]
        ):
            return entry
    logger.debug("No generation lineage match", model=model_id)
    return GENERIC
