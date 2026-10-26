"""Static regressions for Web UI wiring that HTTP tests cannot see.

These guard the bugs found on 2026-10-26/27:
- sandbox presets initialised before PRESETS existed (script aborted)
- history bulk actions must be wired to the visible controls
- custom profile must actually send custom_weights + the speed floor
- experiments builder must expose the speed floor and quality context
"""

from pathlib import Path

UI = Path(__file__).resolve().parents[1] / "lm_optimizer" / "ui"


def _read(rel: str) -> str:
    return (UI / rel).read_text(encoding="utf-8")


def test_sandbox_presets_initialised_after_definition():
    js = _read("static/js/sandbox.js")
    def_pos = js.index("var PRESETS")
    init_pos = js.index("renderPresets();", def_pos)
    assert def_pos < init_pos, "renderPresets() must run after PRESETS is assigned"
    assert "No own prompt? Try a preset" in _read("templates/sandbox.html")


def test_history_bulk_controls_and_handlers():
    js = _read("static/js/history.js")
    for needle in (
        "run-select",
        "select-all-runs",
        "bulk-delete",
        "bulk-rerun",
        "bulk-compare",
        "selectedIds()",
        "API.deleteRun",
        "API.startOptimization",
        "API.judge",
    ):
        assert needle in js, f"history.js missing {needle}"


def test_dashboard_sends_custom_weights_and_speed_floor():
    js = _read("static/js/app.js")
    assert "_customWeights()" in js
    assert "custom_weights" in js
    assert "min-generation-tok-s" in js
    assert "min_generation_tok_s" in js
    # 0-1 priorities, not percentages
    ui = _read("static/js/ui.js")
    assert 'id="w-ttft"' in ui
    assert "must total 100%" not in ui
    assert "filterModels" not in ui  # dead search helper removed


def test_experiments_exposes_speed_floor_and_quality_context():
    js = _read("static/js/experiments.js")
    html = _read("templates/experiments.html")
    assert 'id="ex-min-tps"' in html
    assert 'id="ex-quality-ctx"' in html
    assert "min_generation_tok_s" in js
    assert "quality_context" in js


def test_brewery_hosts_the_experiment_builder():
    html = _read("templates/lab.html")
    assert "lab-panel-experiments" in html
    assert "/experiments?embed=1" in html
    js = _read("static/js/experiments.js")
    assert "embed=1" in js
