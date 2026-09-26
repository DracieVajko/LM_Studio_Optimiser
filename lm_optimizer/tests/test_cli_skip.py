"""--skip for the auto (full-pipeline) command: user-specified model skip list."""

from types import SimpleNamespace


def _targets():
    return [
        SimpleNamespace(id="mistralai/ministral-3-3b"),
        SimpleNamespace(id="qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp"),
        SimpleNamespace(id="openai/gpt-oss-20b"),
    ]


class TestSkipParsing:
    def test_none_and_empty(self):
        from lm_optimizer.cli.main import _parse_skip_models

        assert _parse_skip_models(None) == []
        assert _parse_skip_models("") == []
        assert _parse_skip_models("  ") == []

    def test_comma_list(self):
        from lm_optimizer.cli.main import _parse_skip_models

        assert _parse_skip_models("27b, gpt-oss") == ["27b", "gpt-oss"]


class TestSkipFiltering:
    def test_substring_case_insensitive(self):
        from lm_optimizer.cli.main import _apply_skip_models, _parse_skip_models

        kept, skipped = _apply_skip_models(_targets(), _parse_skip_models("27B"))
        assert [t.id for t in skipped] == ["qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp"]
        assert len(kept) == 2

    def test_no_terms_keeps_all(self):
        from lm_optimizer.cli.main import _apply_skip_models

        kept, skipped = _apply_skip_models(_targets(), [])
        assert len(kept) == 3 and skipped == []

    def test_auto_help_lists_skip(self):
        import typer

        from lm_optimizer.cli.main import app

        info = typer.main.get_command(app)
        params = [p.name for p in info.commands["auto"].params]
        assert "skip" in params
