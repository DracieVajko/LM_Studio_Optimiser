"""Web validator tests (P4 static validation, stdlib only)."""
import pytest

from lm_optimizer.services.web_validator import (
    parse_html,
    js_syntax_errors,
    validate_web_files,
)


GOOD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Metrics dashboard</title>
  <style>@media (max-width: 600px) { body { font-size: 14px; } }</style>
</head>
<body>
  <h1>Metrics</h1>
  <img src="chart.png" alt="Monthly metrics chart">
  <label for="filter">Filter</label>
  <input id="filter" type="text">
  <table><tr><td>Metric</td></tr></table>
  <script>function render() { return 1; }</script>
</body>
</html>
"""


def _checks_by_name(files):
    return {c.name: c for c in validate_web_files(files)}


def test_good_page_passes_all_checks():
    checks = _checks_by_name({"index.html": GOOD_HTML})
    for name, check in checks.items():
        assert check.passed, f"{name} failed: {check.detail}"


def test_missing_html_is_critical():
    checks = _checks_by_name({"app.js": "console.log(1)"})
    assert checks["index_html_exists"].passed is False
    assert checks["index_html_exists"].critical is True


def test_external_script_is_flagged():
    html = GOOD_HTML.replace(
        "<script>function render() { return 1; }</script>",
        '<script src="https://cdn.example.com/app.js"></script>',
    )
    checks = _checks_by_name({"index.html": html})
    assert checks["no_external_scripts"].passed is False


def test_external_stylesheet_is_flagged():
    html = GOOD_HTML.replace(
        "<title>Metrics dashboard</title>",
        '<title>Metrics dashboard</title><link rel="stylesheet" href="https://x/y.css">',
    )
    checks = _checks_by_name({"index.html": html})
    assert checks["no_external_styles"].passed is False


def test_missing_viewport_is_flagged():
    html = GOOD_HTML.replace(
        '<meta name="viewport" content="width=device-width, initial-scale=1">', ""
    )
    checks = _checks_by_name({"index.html": html})
    assert checks["has_viewport"].passed is False


def test_multiple_h1_is_flagged():
    html = GOOD_HTML.replace("<h1>Metrics</h1>", "<h1>Metrics</h1><h1>Second</h1>")
    checks = _checks_by_name({"index.html": html})
    assert checks["single_h1"].passed is False


def test_missing_alt_is_flagged():
    html = GOOD_HTML.replace('alt="Monthly metrics chart"', "")
    checks = _checks_by_name({"index.html": html})
    assert checks["images_have_alt"].passed is False


def test_unlabeled_input_is_flagged():
    html = GOOD_HTML.replace('<label for="filter">Filter</label>', "")
    checks = _checks_by_name({"index.html": html})
    assert checks["inputs_labeled"].passed is False


def test_inline_js_syntax_error_is_flagged():
    html = GOOD_HTML.replace(
        "function render() { return 1; }", "function render( { return 1; }"
    )
    checks = _checks_by_name({"index.html": html})
    # Node may be absent in some environments; only assert when it ran.
    if "script 0" in checks["inline_js_parses"].detail:
        assert checks["inline_js_parses"].passed is False


def test_parse_html_collects_facts():
    facts = parse_html(GOOD_HTML)
    assert facts.title.strip() == "Metrics dashboard"
    assert facts.lang == "en"
    assert facts.has_viewport is True
    assert facts.h1_count == 1
    assert facts.has_media_query is True
    assert len(facts.images) == 1
    assert len(facts.inputs) == 1


def test_parse_html_never_raises_on_garbage():
    facts = parse_html("<<<>>> not html at all </")
    assert isinstance(facts.parse_error, str)


def test_js_syntax_errors_empty_for_valid_code():
    assert js_syntax_errors(["const x = 1; console.log(x);"]) == []


def test_js_syntax_errors_ignores_blank_scripts():
    assert js_syntax_errors(["   ", ""]) == []
