"""Fixed deep-research benchmark suite (long tasks)."""

from ..domain.models import BenchmarkCase

# Planted facts for the recall task. Each is a distinctive ASCII string so
# recall accuracy is mechanically checkable by substring match.
RECALL_FACTS: list[str] = [
    "The harbor lighthouse keeper is named MARLOWE.",
    "The supply ship arrives on the 14th of HARVEST month.",
    "The vault access code is 73921.",
    "The botanist catalogued 47 fern species on Gull Island.",
    "The treaty was signed in the year 1848 at Port Ansel.",
]

RECALL_DOCUMENT: str = (
    "FIELD SURVEY OF GULL ISLAND (synthetic document).\n\n"
    "Section 1: Arrival.\n"
    "The survey team reached Gull Island after a three-day crossing. "
    "The harbor lighthouse keeper is named MARLOWE. "
    "He keeps the lamp lit from dusk until dawn and logs every vessel.\n\n"
    "Section 2: Supplies.\n"
    "Provisions arrive by sea. The supply ship arrives on the 14th of HARVEST month. "
    "Crates are stored in the stone depot above the dock.\n\n"
    "Section 3: Research station.\n"
    "The vault access code is 73921. "
    "Only station staff may enter the vault where samples are kept.\n\n"
    "Section 4: Flora.\n"
    "The botanist catalogued 47 fern species on Gull Island. "
    "Most grow on the shaded northern slopes near the stream.\n\n"
    "Section 5: History.\n"
    "The treaty was signed in the year 1848 at Port Ansel. "
    "A plaque at the harbor commemorates the event.\n\n"
    "Section 6: Notes.\n"
    "Winds are strongest in winter. Gulls nest on the eastern cliffs. "
    "Fresh water comes from the hillside spring."
)

_DEEP_PROMPTS: dict[str, str] = {
    "long_context_recall": (
        "Read the document below carefully. Then answer each question "
        "using only facts stated in the document.\n\n"
        "DOCUMENT:\n" + RECALL_DOCUMENT + "\n\n"
        "QUESTIONS:\n"
        "1. Who is the harbor lighthouse keeper?\n"
        "2. When does the supply ship arrive?\n"
        "3. What is the vault access code?\n"
        "4. How many fern species did the botanist catalogue on Gull Island?\n"
        "5. When and where was the treaty signed?\n\n"
        "Answer as a numbered list with one fact per line."
    ),
    "multi_hop": (
        "A relay race has four runners. Ana runs first and hands off to Ben. "
        "Ben is twice as slow as Ana per lap. Cora runs third and her lap time "
        "is the average of Ana's and Ben's lap times. Dana runs last and her lap "
        "time is 10 seconds faster than Cora's. Ana's lap time is 60 seconds. "
        "What is the total team time? Show each runner's lap time step by step, "
        "then give the total in seconds."
    ),
    "json_discipline": (
        "Output a JSON object with exactly these keys: "
        '"project", "version", "stages" (array of exactly 3 strings), '
        'and "meta" (object with keys "author" and "year"). '
        "Use realistic values for a bridge-building project. "
        "No extra text, no markdown fences, just the JSON object."
    ),
    "coding_precision": (
        "Write a Python function `is_sorted_unique(nums: list[int]) -> bool` that returns "
        "True only if the list is strictly increasing (each element greater than the "
        "previous) with no duplicates. Requirements:\n"
        "1. Run in O(n) time complexity\n"
        "2. Use O(1) extra space\n"
        "3. Handle empty and single-element lists (return True)\n"
        "4. No imports\n\n"
        "Provide only the function definition with docstring."
    ),
    "instruction_follow": (
        "Follow these instructions exactly:\n"
        "1. Write exactly 4 lines.\n"
        "2. Each line must start with the word 'Line' followed by its number (Line 1, Line 2, ...).\n"
        "3. Line 2 must contain the word 'compass'.\n"
        "4. Line 4 must end with a period.\n"
        "5. Do not add any extra lines, headers, or explanations."
    ),
}


def _make_case(name: str, category: str, temperature: float, max_tokens: int) -> BenchmarkCase:
    return BenchmarkCase(
        name=name,
        category=category,
        prompt=_DEEP_PROMPTS[name],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=None,
        top_k=None,
    )


# Fixed long-task suite shared by every model in the deep benchmark.
DEEP_CASES: list[BenchmarkCase] = [
    _make_case("long_context_recall", "recall", 0.0, 2048),
    _make_case("multi_hop", "reasoning", 0.3, 2048),
    _make_case("json_discipline", "format", 0.0, 1024),
    _make_case("coding_precision", "coding", 0.1, 2048),
    _make_case("instruction_follow", "instruction", 0.3, 2048),
]


def deep_metrics(metrics) -> dict:
    """Derive deep-suite metrics from one BenchmarkMetrics row.

    recall_accuracy: fraction of RECALL_FACTS found in output_text
    (case-insensitive substring match; 0.0 when output is empty).
    thinking_chars: length of thinking_text (0 when absent, best-effort).
    elapsed_s: (prompt_processing_ms + generation_ms) in seconds.
    gen_tok_s: generation throughput as reported.
    """
    output_text = getattr(metrics, "output_text", "") or ""
    thinking_text = getattr(metrics, "thinking_text", "") or ""
    lowered = output_text.lower()
    if RECALL_FACTS:
        hits = sum(1 for fact in RECALL_FACTS if fact.lower() in lowered)
        recall_accuracy = hits / len(RECALL_FACTS)
    else:
        recall_accuracy = 0.0
    prompt_ms = float(getattr(metrics, "prompt_processing_ms", 0.0) or 0.0)
    generation_ms = float(getattr(metrics, "generation_ms", 0.0) or 0.0)
    return {
        "recall_accuracy": recall_accuracy,
        "thinking_chars": len(thinking_text),
        "elapsed_s": (prompt_ms + generation_ms) / 1000.0,
        "gen_tok_s": float(getattr(metrics, "generation_tok_s", 0.0) or 0.0),
    }
