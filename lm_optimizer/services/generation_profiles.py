"""Generation profiles: measured benchmark settings + publisher recommendations.

Two columns, never mixed:
- "ours": temperatures actually used by our benchmark suite (verifiable in
  benchmark/suite.py). top_p/top_k we do NOT sweep: server defaults.
- "publisher": vendor-recommended sampling per architecture family, each
  value tagged with its source. Unknown families get None (check the model
  card) — never invented numbers.
"""

# Verified publisher sources (checked 2026-09-30):
# - Qwen/Unsloth model cards + unsloth.ai Qwen3 docs: thinking 0.6-1.0,
#   precise coding 0.6/0.95/20, instruct 0.7/0.8/20.
# - Mistral docs (sampling): temperature 0.0-0.7 guidance, no fixed triple.
# - Google/Gemma team: general 1.0/0.95/64; deterministic tasks 0.2-0.5,
#   top_p ~0.8-0.9, top_k ~40.

_QWEN_PUBLISHER = {
    "precision": {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                  "note": "precise coding tasks", "source": "unsloth:qwen3-cards"},
    "chat": {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
             "note": "instruct mode", "source": "unsloth:qwen3-cards"},
    "creative": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                 "note": "thinking mode, general tasks", "source": "unsloth:qwen3-cards"},
}

_MISTRAL_PUBLISHER = {
    "precision": {"temperature": 0.2, "top_p": None, "top_k": None,
                  "note": "docs: 0.0-0.7 range, lower = focused; fix one knob",
                  "source": "mistral-docs:sampling"},
    "chat": {"temperature": 0.7, "top_p": None, "top_k": None,
             "note": "docs: higher values more random", "source": "mistral-docs:sampling"},
    "creative": {"temperature": 1.0, "top_p": None, "top_k": None,
                 "note": "docs example uses temperature=1", "source": "mistral-docs:sampling"},
}

_GEMMA_PUBLISHER = {
    "precision": {"temperature": 0.3, "top_p": 0.9, "top_k": 40,
                  "note": "deterministic/fact tasks per Google team",
                  "source": "google:gemma-team"},
    "chat": {"temperature": 1.0, "top_p": 0.95, "top_k": 64,
             "note": "general use per Google team", "source": "google:gemma-team"},
    "creative": {"temperature": 1.0, "top_p": 0.95, "top_k": 64,
                 "note": "general use per Google team", "source": "google:gemma-team"},
}

_Ours = {
    "precision": {"temperature": 0.1, "top_p": None, "top_k": None,
                  "note": "benchmark suite 'precise' style; top_p/top_k server defaults (untuned)",
                  "source": "suite:precise"},
    "chat": {"temperature": None, "top_p": None, "top_k": None,
             "note": "suite defaults per test (0.0-0.7); server defaults otherwise",
             "source": "suite:balanced"},
    "creative": {"temperature": 0.9, "top_p": None, "top_k": None,
                 "note": "benchmark suite 'creative' style", "source": "suite:creative"},
}

_USE = {
    "precision": "Factual work: coding, numbers, anything the model must not invent.",
    "chat": "Everyday chat: balanced defaults.",
    "creative": "Stories, ideas, summarization with variance.",
}

_FAMILIES = {"qwen": _QWEN_PUBLISHER, "mistral": _MISTRAL_PUBLISHER,
             "ministral": _MISTRAL_PUBLISHER, "gemma": _GEMMA_PUBLISHER}


def _family_for(model_id: str, architecture: str | None) -> str | None:
    hay = f"{model_id or ''} {architecture or ''}".lower()
    for key in ("qwen", "mistral", "ministral", "gemma"):
        if key in hay:
            return key
    return None


def profiles_for(model_id: str, architecture: str | None = None) -> dict:
    """Three generation profiles with ours + publisher columns."""
    fam = _family_for(model_id or "", architecture)
    publishers = _FAMILIES.get(fam or "", {})
    out = {}
    for name in ("precision", "chat", "creative"):
        out[name] = {
            "use": _USE[name],
            "ours": dict(_Ours[name]),
            "publisher": dict(publishers[name]) if name in publishers else None,
        }
    return out
