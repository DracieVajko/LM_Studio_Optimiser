"""Open-web research sandbox tests (P4, spec §13.2)."""
import pytest

from lm_optimizer.services.research_web import (
    Source,
    build_research_prompt,
    extract_citations,
    html_to_text,
    score_research,
    validate_citations,
)


def _source(url, title="Title", text="Body text"):
    return Source(url=url, title=title, text=text)


def test_html_to_text_strips_scripts_and_styles():
    html = """
    <html><head><title>Doc</title>
    <style>body{color:red}</style></head>
    <body><nav>menu</nav><p>Hello world</p>
    <script>alert(1)</script><footer>foot</footer></body></html>
    """
    title, text = html_to_text(html)
    assert title == "Doc"
    assert "Hello world" in text
    assert "alert" not in text
    assert "menu" not in text
    assert "color:red" not in text


def test_html_to_text_never_raises():
    title, text = html_to_text("<<< broken </")
    assert isinstance(title, str)
    assert isinstance(text, str)


def test_extract_citations_dedupes_and_preserves_order():
    answer = "Claim [source-a] and [source-b], again [source-a]."
    assert extract_citations(answer) == ["source-a", "source-b"]


def test_validate_citations_precision_and_recall():
    sources = [_source("https://a.example.com/page"), _source("https://b.example.com/x")]
    slugs = [s.slug for s in sources]
    answer = f"Fact one [{slugs[0]}]. Fact two [invented-source]."
    report = validate_citations(answer, sources)
    assert report.valid == [slugs[0]]
    assert report.invented == ["invented-source"]
    assert report.recall == pytest.approx(0.5)
    assert report.precision == pytest.approx(0.5)
    assert slugs[1] in report.unused


def test_validate_citations_no_citations():
    sources = [_source("https://a.example.com")]
    report = validate_citations("No citations here.", sources)
    assert report.cited == []
    assert report.precision == 0.0
    assert report.recall == 0.0


def test_score_research_penalises_invented_citations():
    sources = [_source("https://a.example.com/page")]
    slug = sources[0].slug
    good = score_research(f"A solid synthesis with facts [{slug}].", sources)
    bad = score_research(f"A solid synthesis with facts [{slug}] [fake-source].", sources)
    assert good["score"] > bad["score"]
    assert bad["invented_citations"] == ["fake-source"]
    assert good["passed"] is True


def test_score_research_requires_content():
    sources = [_source("https://a.example.com/page")]
    result = score_research("", sources)
    assert result["score"] == 0.0
    assert result["passed"] is False


def test_score_research_low_recall_fails():
    sources = [
        _source("https://a.example.com/one"),
        _source("https://b.example.com/two"),
        _source("https://c.example.com/three"),
    ]
    answer = f"Only one source [{sources[0].slug}] and a lot of text to pass the length check."
    result = score_research(answer, sources, min_recall=0.5)
    assert result["passed"] is False
    assert result["citations"]["recall"] == pytest.approx(1 / 3, abs=1e-3)


def test_build_prompt_lists_sources_and_contract():
    sources = [_source("https://a.example.com/page", title="A", text="alpha")]
    prompt = build_research_prompt("What is alpha?", sources)
    assert "QUESTION: What is alpha?" in prompt
    assert sources[0].slug in prompt
    assert "Never invent a source" in prompt


def test_build_prompt_skips_failed_sources():
    sources = [
        _source("https://ok.example.com"),
        Source(url="https://bad.example.com", error="timeout"),
    ]
    prompt = build_research_prompt("q", sources)
    assert "ok-example-com" in prompt
    assert "bad-example-com" not in prompt


def test_source_slug_is_stable():
    a = Source(url="https://Example.COM/Path?q=1")
    b = Source(url="https://Example.COM/Path?q=1")
    assert a.slug == b.slug
    assert a.slug.islower()
