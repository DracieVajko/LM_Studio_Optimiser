"""Open-web research sandbox (spec §13.2, second variant).

The closed-package variant ships fixed sources. This variant fetches real
sources at run time and validates the synthesis against them:

- every source is fetched and reduced to plain text,
- the model must cite the sources it used by their URL/slug,
- citation PRECISION (cited sources that were actually fetched) and
  RECALL (fetched sources that were used) are both measured,
- invented citations are penalised.

Network is explicitly opt-in (`network_allowed=True`), which is why this
lives behind its own flag and is never part of the default suite.

Pure helpers are offline-testable; only `fetch_sources` touches the network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any


@dataclass
class Source:
    """One fetched web source."""

    url: str
    title: str = ""
    text: str = ""
    error: str = ""

    @property
    def slug(self) -> str:
        """Short citation handle derived from the URL."""
        cleaned = re.sub(r"^https?://", "", self.url)
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", cleaned).strip("-")
        return cleaned[:60].lower()

    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and not self.error


class _TextExtractor(HTMLParser):
    """Extract visible text from HTML (drops script/style/nav noise)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "nav", "header", "footer", "aside"):
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "nav", "header", "footer", "aside"):
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data.strip())

    @property
    def text(self) -> str:
        return " ".join(self.parts)


def html_to_text(html: str) -> tuple[str, str]:
    """(title, text) from HTML; never raises."""
    parser = _TextExtractor()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception:  # noqa: BLE001
        pass
    return parser.title.strip(), re.sub(r"\s+", " ", parser.text).strip()


async def fetch_sources(urls: list[str], timeout_s: float = 20.0) -> list[Source]:
    """Fetch and reduce each URL to text. Errors are recorded per source."""
    import httpx

    out: list[Source] = []
    async with httpx.AsyncClient(
        timeout=timeout_s, follow_redirects=True,
        headers={"User-Agent": "lm-optimizer-research/1.0"},
    ) as client:
        for url in urls:
            try:
                response = await client.get(url)
                response.raise_for_status()
                title, text = html_to_text(response.text)
                out.append(Source(url=url, title=title, text=text))
            except Exception as e:  # noqa: BLE001
                out.append(Source(url=url, error=f"{type(e).__name__}: {e}"[:160]))
    return out


def build_research_prompt(question: str, sources: list[Source], max_chars: int = 4000) -> str:
    """Prompt with the fetched sources and an explicit citation contract."""
    blocks = []
    for source in sources:
        if not source.ok:
            continue
        body = source.text[:max_chars]
        blocks.append(f"[SOURCE {source.slug}] {source.title}\n{body}")
    joined = "\n\n".join(blocks)
    return (
        "Answer the question using ONLY the sources below.\n"
        "Cite every claim by its SOURCE slug in square brackets, e.g. [example-com-page].\n"
        "If the sources do not answer something, say so explicitly.\n"
        "Never invent a source that is not listed.\n\n"
        f"QUESTION: {question}\n\nSOURCES:\n{joined}"
    )


_CITATION_RE = re.compile(r"\[([a-z0-9][a-z0-9-]{2,80})\]")


def extract_citations(answer: str) -> list[str]:
    """Citation slugs used in the answer (order preserved, deduplicated)."""
    seen: list[str] = []
    for match in _CITATION_RE.finditer(answer or ""):
        slug = match.group(1)
        if slug not in seen:
            seen.append(slug)
    return seen


@dataclass
class CitationReport:
    """Precision/recall of the citations in one answer."""

    cited: list[str] = field(default_factory=list)
    valid: list[str] = field(default_factory=list)
    invented: list[str] = field(default_factory=list)
    fetched: list[str] = field(default_factory=list)
    used: list[str] = field(default_factory=list)
    unused: list[str] = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "cited": self.cited,
            "valid": self.valid,
            "invented": self.invented,
            "fetched": self.fetched,
            "used": self.used,
            "unused": self.unused,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
        }


def validate_citations(answer: str, sources: list[Source]) -> CitationReport:
    """Measure citation precision and recall against the fetched sources.

    precision = valid citations / all citations (invented ones lower it)
    recall    = fetched sources actually cited / fetched sources
    """
    report = CitationReport()
    report.cited = extract_citations(answer)
    report.fetched = [s.slug for s in sources if s.ok]
    valid_set = set(report.fetched)
    report.valid = [c for c in report.cited if c in valid_set]
    report.invented = [c for c in report.cited if c not in valid_set]
    report.used = [slug for slug in report.fetched if slug in report.cited]
    report.unused = [slug for slug in report.fetched if slug not in report.cited]
    if report.cited:
        report.precision = len(report.valid) / len(report.cited)
    if report.fetched:
        report.recall = len(report.used) / len(report.fetched)
    return report


def score_research(answer: str, sources: list[Source], min_recall: float = 0.5) -> dict:
    """Research-task score: hard fail on invented citations, weighted rest."""
    report = validate_citations(answer, sources)
    has_content = len((answer or "").strip()) > 40
    invented_penalty = 0.0 if report.invented else 1.0
    score = (
        (1.0 if has_content else 0.0) * 0.2
        + report.precision * 0.4
        + min(1.0, report.recall / max(min_recall, 0.01)) * 0.4
    ) * invented_penalty
    return {
        "score": round(max(0.0, min(1.0, score)), 4),
        "invented_citations": report.invented,
        "citations": report.as_dict(),
        "passed": bool(has_content and not report.invented and report.recall >= min_recall),
    }
