"""Auto-run model eligibility.

Decides whether a discovered model may be handed to the text-generation
optimizer. Embedding-only, encoder-only and reranker-only models cannot be
measured by a generation benchmark: they have no token-generation throughput
and no instruction-following quality, so optimizing them burns hours of VRAM
time and produces meaningless numbers.

Decision order (first signal that is present wins):

1. Provider metadata (``type`` on the model record, or a runtime survey).
2. Architecture / name signals, used only when metadata is absent.
3. Otherwise UNKNOWN -> not eligible (conservative), with a stated reason.

Unknown is never silently assumed to be a generation model. An explicit
opt-in exists for operators who know a model is generative despite a missing or
mis-declared type.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ModelEligibility(str, Enum):
    """Eligibility verdict for auto-run optimization."""

    ELIGIBLE = "eligible"
    EXCLUDED = "excluded"
    UNKNOWN = "unknown"


# Provider ``type`` values that are explicitly non-generative.
NON_GENERATIVE_TYPES = frozenset(
    {
        "embedding",
        "embeddings",
        "text-embedding",
        "bert",
        "encoder",
        "reranker",
        "rerank",
        "retrieval",
        "colbert",
        "classification",
        "audio",
        "whisper",
        "tts",
        "image",
        "vision-encoder",
    }
)

# Provider ``type`` values that are explicitly generative.
GENERATIVE_TYPES = frozenset(
    {
        "llm",
        "llama",
        "gpt",
        "gpt-oss",
        "chat",
        "text-generation",
        "text_generation",
        "completions",
        "completion",
    }
)

# Architecture / name markers for a model that encodes rather than generates.
# Used ONLY when the provider declared no type.
_NON_GENERATIVE_SIGNALS = (
    "embed",
    "embedding",
    "embeddings",
    "nomic-embed",
    "sentence-transformers",
    "sentence_transformers",
    "minilm",
    "bge-",
    "bge_",
    "gte-",
    "gte_",
    "e5-",
    "jina-embeddings",
    "jina_bert",
    "all-minilm",
    "labse",
    "colbert",
    "reranker",
    "rerank-",
    "cross-encoder",
    "cross_encoder",
    "whisper",
    "tts",
    "wav2vec",
)

# Architecture markers for encoder-only transformer stacks that never generate.
_NON_GENERATIVE_ARCHITECTURES = frozenset(
    {
        "bert",
        "roberta",
        "xlm-roberta",
        "distilbert",
        "albert",
        "electra",
        "deberta",
        "deberta-v2",
        "nomic-bert",
        "modernbert",
        "jina-bert",
        "mpnet",
    }
)


@dataclass(frozen=True)
class EligibilityVerdict:
    """Outcome of the eligibility check, with the reason it was reached."""

    model_id: str
    eligibility: ModelEligibility
    reason: str
    source: str

    @property
    def allowed(self) -> bool:
        return self.eligibility is ModelEligibility.ELIGIBLE


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def classify_model_type(
    model_type: str | None,
    *,
    architecture: str | None = None,
    name: str | None = None,
    model_id: str | None = None,
    force_generation: bool = False,
) -> EligibilityVerdict:
    """Classify one model for generation-benchmark eligibility.

    ``force_generation`` is the explicit operator opt-in: it bypasses an
    exclusion but NOT an UNKNOWN verdict when no signal at all exists, because
    an operator override cannot invent evidence that the model generates.
    """
    mid = str(model_id or name or "unknown")
    mt = _norm(model_type)
    arch = _norm(architecture)
    label = f"{_norm(name)} {_norm(model_id)}"
    haystack = f"{label} {arch}"

    if force_generation:
        if mt in NON_GENERATIVE_TYPES:
            return EligibilityVerdict(
                mid,
                ModelEligibility.EXCLUDED,
                f"provider type {model_type!r} is non-generative "
                f"(overridden with --force-generation)",
                "provider-metadata+override",
            )
        if mt:
            return EligibilityVerdict(
                mid, ModelEligibility.ELIGIBLE, "explicit override", "override"
            )
        return EligibilityVerdict(
            mid,
            ModelEligibility.UNKNOWN,
            "no provider type available; override cannot establish generation support",
            "override",
        )

    # 1. Provider metadata is authoritative when present.
    if mt:
        if mt in NON_GENERATIVE_TYPES:
            return EligibilityVerdict(
                mid,
                ModelEligibility.EXCLUDED,
                f"provider type {model_type!r} is non-generative",
                "provider-metadata",
            )
        if mt in GENERATIVE_TYPES:
            return EligibilityVerdict(
                mid, ModelEligibility.ELIGIBLE, f"provider type {model_type!r}", "provider-metadata"
            )
        # A type we do not recognize is not evidence of generation.
        return EligibilityVerdict(
            mid,
            ModelEligibility.UNKNOWN,
            f"unrecognized provider type {model_type!r}",
            "provider-metadata",
        )

    # 2. No metadata: fall back to architecture, then name.
    if arch in _NON_GENERATIVE_ARCHITECTURES:
        return EligibilityVerdict(
            mid,
            ModelEligibility.EXCLUDED,
            f"architecture {architecture!r} is encoder-only",
            "architecture",
        )
    for sig in _NON_GENERATIVE_SIGNALS:
        if sig in haystack:
            return EligibilityVerdict(
                mid,
                ModelEligibility.EXCLUDED,
                f"name/architecture matches non-generative signal {sig!r}",
                "name-signal",
            )

    # 3. Nothing at all -> conservative unknown.
    return EligibilityVerdict(
        mid,
        ModelEligibility.UNKNOWN,
        "provider reported no model type and no non-generative signal matched",
        "none",
    )


def check_model(
    model: Any,
    *,
    provider_type: str | None = None,
    force_generation: bool = False,
) -> EligibilityVerdict:
    """Classify a discovered model (ModelIdentity or raw provider dict)."""
    if isinstance(model, dict):
        quant = model.get("quantization")
        quant_name = quant.get("name") if isinstance(quant, dict) else quant
        del quant_name  # quantization is not an eligibility signal
        return classify_model_type(
            provider_type if provider_type is not None else model.get("type"),
            architecture=model.get("architecture"),
            name=model.get("display_name") or model.get("name"),
            model_id=model.get("key") or model.get("id"),
            force_generation=force_generation,
        )

    return classify_model_type(
        provider_type if provider_type is not None else getattr(model, "model_type", None),
        architecture=getattr(model, "architecture", None),
        name=getattr(model, "name", None),
        model_id=getattr(model, "id", None),
        force_generation=force_generation,
    )


@dataclass
class EligibilitySummary:
    """Reconciling counts for the eligibility pass."""

    discovered: int = 0
    eligible: int = 0
    excluded: int = 0
    unknown: int = 0

    def add(self, verdict: EligibilityVerdict) -> None:
        self.discovered += 1
        if verdict.eligibility is ModelEligibility.ELIGIBLE:
            self.eligible += 1
        elif verdict.eligibility is ModelEligibility.EXCLUDED:
            self.excluded += 1
        else:
            self.unknown += 1

    def reconciled(self) -> bool:
        return self.discovered == self.eligible + self.excluded + self.unknown


def partition_models(
    models: list[Any],
    *,
    provider_types: dict[str, str] | None = None,
    force_generation: bool = False,
) -> tuple[list[Any], list[tuple[Any, EligibilityVerdict]], EligibilitySummary]:
    """Split discovered models into eligible ones and skipped ones.

    Returns ``(eligible, skipped, summary)``. ``skipped`` carries the verdict so
    the caller can report the reason. An excluded model is NOT a failure: it
    was never a candidate.
    """
    types = provider_types or {}
    eligible: list[Any] = []
    skipped: list[tuple[Any, EligibilityVerdict]] = []
    summary = EligibilitySummary()

    for m in models:
        mid = m.get("key") or m.get("id") if isinstance(m, dict) else getattr(m, "id", "")
        verdict = check_model(m, provider_type=types.get(mid), force_generation=force_generation)
        summary.add(verdict)
        if verdict.allowed:
            eligible.append(m)
        else:
            skipped.append((m, verdict))

    return eligible, skipped, summary