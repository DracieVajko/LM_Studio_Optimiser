"""Experiment Builder wizard + granular API tests (P6, spec §19.3)."""
import pytest
from fastapi.testclient import TestClient

from lm_optimizer.api.main import app
from lm_optimizer.services.optimizer import compose_granular_skips


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# --- granular skip composition shared by CLI and web -----------------------

def test_compose_speed_only():
    skips = compose_granular_skips({"speed_only": True})
    assert {"quality", "recovery", "validation", "context"} <= set(skips)
    assert "speed" not in skips


def test_compose_quality_only():
    skips = compose_granular_skips({"quality_only": True})
    assert {"speed", "recovery", "validation", "context"} <= set(skips)
    assert "quality" not in skips


def test_compose_reasoning_only():
    skips = compose_granular_skips({"reasoning_sweep": True})
    assert {"speed", "recovery", "validation", "context"} <= set(skips)


def test_compose_mtp_only_keeps_speed():
    skips = compose_granular_skips({"mtp_test": True})
    assert {"quality", "recovery", "validation", "context"} <= set(skips)
    assert "speed" not in skips


def test_compose_merges_user_skips():
    skips = compose_granular_skips({"quality_only": True, "skip_phases": ["recovery"]})
    assert "recovery" in skips
    assert "speed" in skips


def test_compose_no_flags_is_empty():
    assert compose_granular_skips({}) == []
    assert compose_granular_skips(None) == []


# --- web endpoint accepts granular flags ----------------------------------

def test_optimize_schema_accepts_granular_flags():
    from lm_optimizer.api.schemas import AdvancedSettingsSchema

    schema = AdvancedSettingsSchema(
        speed_only=True,
        quality_lab=True,
        speed_context=16384,
    )
    assert schema.speed_only is True
    assert schema.quality_lab is True
    assert schema.speed_context == 16384


def test_experiments_page_renders(client):
    response = client.get("/experiments")
    assert response.status_code == 200, response.text
    assert "Experiment" in response.text
    assert "/static/js/experiments.js" in response.text


def test_experiments_nav_present(client):
    for path in ("/", "/lab", "/settings"):
        response = client.get(path)
        if response.status_code != 200:
            continue
        assert 'href="/experiments"' in response.text, f"missing nav on {path}"
