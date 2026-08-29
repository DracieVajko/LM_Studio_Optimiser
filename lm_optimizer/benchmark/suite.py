"""Benchmark test suite with deterministic prompts."""

from dataclasses import dataclass

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkPrompt:
    """A single benchmark prompt."""

    name: str
    category: str
    prompt: str
    expected_tokens: int | None = None
    min_tokens: int = 10
    max_tokens: int = 512
    temperature: float = 0.7
    stop_sequences: list[str] | None = None


# Fixed benchmark suite - same prompts used for all configurations
BENCHMARK_SUITE: list[BenchmarkPrompt] = [
    # Test A: Short instruction
    BenchmarkPrompt(
        name="short_instruction",
        category="instruction",
        prompt="Write a concise explanation of how a hash table works in 3-4 sentences.",
        expected_tokens=150,
        min_tokens=50,
        max_tokens=256,
        temperature=0.3,
    ),
    # Test B: Medium reasoning
    BenchmarkPrompt(
        name="medium_reasoning",
        category="reasoning",
        prompt=(
            "You are given a sequence: 2, 6, 12, 20, 30, 42, 56. "
            "What is the next number in the sequence? Explain your reasoning step by step."
        ),
        expected_tokens=200,
        min_tokens=100,
        max_tokens=512,
        temperature=0.3,
    ),
    # Test C: Long context (adjustable based on context length)
    BenchmarkPrompt(
        name="long_context",
        category="context",
        prompt=(
            "Below is a document about renewable energy. Please read it carefully and answer the question at the end.\n\n"
            "DOCUMENT:\n"
            "Renewable energy comes from natural sources that are constantly replenished. "
            "Solar energy harnesses sunlight using photovoltaic cells or concentrated solar power systems. "
            "Wind energy captures kinetic energy from wind using turbines. "
            "Hydroelectric power generates electricity from flowing water. "
            "Geothermal energy taps into heat from the Earth's core. "
            "Biomass energy uses organic materials like wood, crops, and waste. "
            "Each source has advantages: solar is widely available, wind is cost-effective in windy areas, "
            "hydro provides consistent baseload power, geothermal is reliable, and biomass can use waste. "
            "Challenges include intermittency for solar and wind, environmental impact of hydro, "
            "geographic limitations for geothermal, and land use for biomass. "
            "Energy storage solutions like batteries, pumped hydro, and thermal storage help address intermittency. "
            "Grid integration requires smart inverters, forecasting, and demand response. "
            "Policy support includes tax credits, renewable portfolio standards, and feed-in tariffs. "
            "The levelized cost of energy for solar and wind has dropped dramatically, "
            "making them competitive with fossil fuels in many regions. "
            "Future trends include offshore floating wind, perovskite solar cells, "
            "green hydrogen from renewable electrolysis, and advanced geothermal systems.\n\n"
            "QUESTION: Summarize the main renewable energy sources, their key advantages, "
            "primary challenges, and two future trends mentioned in the document."
        ),
        expected_tokens=300,
        min_tokens=150,
        max_tokens=1024,
        temperature=0.3,
    ),
    # Test D: Coding
    BenchmarkPrompt(
        name="coding_task",
        category="coding",
        prompt=(
            "Write a Python function `find_duplicates(nums: list[int]) -> list[int]` that returns "
            "all duplicate integers in a list. The function should:\n"
            "1. Run in O(n) time complexity\n"
            "2. Use O(1) extra space (excluding output)\n"
            "3. Handle negative numbers\n"
            "4. Return duplicates in ascending order\n\n"
            "Provide only the function definition with docstring."
        ),
        expected_tokens=200,
        min_tokens=100,
        max_tokens=512,
        temperature=0.1,
        stop_sequences=["```", "def ", "class "],
    ),
    # Test E: Structured output (JSON)
    BenchmarkPrompt(
        name="structured_output",
        category="format",
        prompt=(
            "Output a JSON object with the following structure exactly:\n"
            "{\n"
            '  "name": "string",\n'
            '  "age": integer,\n'
            '  "skills": ["string", "string", "string"],\n'
            '  "address": {\n'
            '    "city": "string",\n'
            '    "country": "string"\n'
            "  }\n"
            "}\n\n"
            "Use realistic data for a software engineer. No extra text, just the JSON."
        ),
        expected_tokens=100,
        min_tokens=50,
        max_tokens=256,
        temperature=0.0,
        stop_sequences=["}"],
    ),
]


@dataclass
class BenchmarkCase:
    """A benchmark test case with context-specific prompts."""

    name: str
    category: str
    prompt: str
    max_tokens: int
    temperature: float
    stop_sequences: list[str] | None
    context_length: int


def create_benchmark_cases(context_length: int, seed: int = 42) -> list[BenchmarkCase]:
    """Create benchmark cases adapted for the given context length."""
    cases = []

    for bp in BENCHMARK_SUITE:
        # For long context test, scale the prompt if context allows
        prompt = bp.prompt
        max_tokens = bp.max_tokens

        if bp.category == "context" and context_length > 8192:
            # Extend the document for larger context windows
            repeat_count = min(context_length // 4000, 4)
            prompt = bp.prompt.replace(
                "DOCUMENT:\n", "DOCUMENT:\n" + "\n".join(["---"] * repeat_count) + "\n"
            )

        case = BenchmarkCase(
            name=bp.name,
            category=bp.category,
            prompt=prompt,
            max_tokens=min(max_tokens, context_length // 4),  # Leave room for prompt
            temperature=bp.temperature,
            stop_sequences=bp.stop_sequences,
            context_length=context_length,
        )
        cases.append(case)

    return cases


def get_benchmark_suite() -> list[BenchmarkPrompt]:
    """Get the fixed benchmark suite."""
    return BENCHMARK_SUITE.copy()
