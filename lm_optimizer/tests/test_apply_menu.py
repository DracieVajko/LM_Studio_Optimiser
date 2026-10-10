"""apply-menu picker: entry collection, choice resolution, registration.

No servers, no models, no DB writes: pure helpers + typer wiring only.
"""

import pytest


def _presets():
    return [
        {"id": "p1", "model_id": "mistralai/ministral-3-3b", "profile": "balanced",
         "config": {"context_length": 32768}},
        {"id": "p2", "model_id": "qwen/qwen3-coder-30b", "profile": "speed",
         "config": {"context_length": 8192}},
        {"id": "p3", "model_id": "", "profile": "balanced", "config": {}},
    ]


def _modelfiles():
    return [
        {"path": "ollama-configs/qwen35-best.modelfile.md", "from": "qwen3.5:4b",
         "num_ctx": 131072, "temperature": 0.7},
        {"path": "", "from": "qwen3.5:4b"},
    ]


class TestCollectEntries:
    def test_presets_become_lmstudio_entries(self):
        from lm_optimizer.cli.main import _collect_apply_entries

        entries = _collect_apply_entries(_presets(), [])
        assert [(e.backend, e.model, e.profile) for e in entries] == [
            ("lmstudio", "mistralai/ministral-3-3b", "balanced"),
            ("lmstudio", "qwen/qwen3-coder-30b", "speed"),
        ]
        assert "32768" in entries[0].detail

    def test_blank_model_id_skipped(self):
        from lm_optimizer.cli.main import _collect_apply_entries

        assert len(_collect_apply_entries(_presets(), [])) == 2

    def test_modelfiles_become_ollama_entries(self):
        from lm_optimizer.cli.main import _collect_apply_entries

        entries = _collect_apply_entries([], _modelfiles())
        assert len(entries) == 1
        assert entries[0].backend == "ollama"
        assert "qwen3.5:4b" in entries[0].label
        assert entries[0].path.endswith(".modelfile.md")

    def test_empty_inputs(self):
        from lm_optimizer.cli.main import _collect_apply_entries

        assert _collect_apply_entries([], []) == []
        assert _collect_apply_entries(None, None) == []


class TestResolveChoice:
    def test_valid(self):
        from lm_optimizer.cli.main import _collect_apply_entries, _resolve_apply_choice

        entries = _collect_apply_entries(_presets(), [])
        assert _resolve_apply_choice(entries, 2).model == "qwen/qwen3-coder-30b"

    @pytest.mark.parametrize("choice", [0, -1, 3, 99])
    def test_invalid_is_none(self, choice):
        from lm_optimizer.cli.main import _collect_apply_entries, _resolve_apply_choice

        entries = _collect_apply_entries(_presets(), [])
        assert _resolve_apply_choice(entries, choice) is None

    def test_empty_list_is_none(self):
        from lm_optimizer.cli.main import _resolve_apply_choice

        assert _resolve_apply_choice([], 1) is None


class TestScanConfigDir:
    def test_parses_valid_and_skips_broken(self, tmp_path, monkeypatch):
        import lm_optimizer.backends.ollama.modelfile as mf
        from lm_optimizer.cli.main import _scan_ollama_config_dir

        good = tmp_path / "a-best.modelfile.md"
        good.write_text("FROM qwen3.5:4b\nPARAMETER num_ctx 8192\n", encoding="utf-8")
        bad = tmp_path / "bad.modelfile.md"
        bad.write_text("PARAMETER num_ctx abc\n", encoding="utf-8")
        monkeypatch.setattr(mf, "OLLAMA_CONFIG_DIR", tmp_path)

        found = _scan_ollama_config_dir()
        assert len(found) == 1
        assert found[0]["from"] == "qwen3.5:4b"
        assert found[0]["num_ctx"] == 8192

    def test_missing_dir_is_empty(self, tmp_path, monkeypatch):
        import lm_optimizer.backends.ollama.modelfile as mf
        from lm_optimizer.cli.main import _scan_ollama_config_dir

        monkeypatch.setattr(mf, "OLLAMA_CONFIG_DIR", tmp_path / "nope")
        assert _scan_ollama_config_dir() == []


class TestRegistration:
    def test_apply_menu_registered_with_backend(self):
        import typer

        from lm_optimizer.cli.main import app

        info = typer.main.get_command(app)
        assert "apply-menu" in info.commands
        params = [p.name for p in info.commands["apply-menu"].params]
        assert "backend" in params
