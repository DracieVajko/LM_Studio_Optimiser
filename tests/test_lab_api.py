"""Lab API + UI smoke tests (P6 web surface for the P2-P5 workflows)."""
import pytest
from fastapi.testclient import TestClient

from lm_optimizer.api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_lab_recommend_endpoint(client):
    response = client.get("/api/lab/recommend?min_tps=3&min_context=8192")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "hardware" in payload
    assert isinstance(payload["recommendations"], list)
    for rec in payload["recommendations"]:
        assert "model" in rec
        assert "fit_class" in rec
        assert rec["fit_class"] in (
            "gpu_resident",
            "hybrid_interactive",
            "hybrid_quality",
            "not_recommended",
        )


def test_lab_recommend_rejects_unknown_use_case(client):
    response = client.get("/api/lab/recommend?use_case=not-a-case")
    assert response.status_code == 400


def test_lab_quality_tasks_endpoint(client):
    response = client.get("/api/lab/quality-tasks")
    assert response.status_code == 200, response.text
    tasks = response.json()["tasks"]
    assert tasks
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids))
    for task in tasks:
        assert task["prompt"].strip()
        assert task["validators"] or task["hard_gates"]


def test_lab_duel_types_endpoint(client):
    response = client.get("/api/lab/duel-types")
    assert response.status_code == 200, response.text
    types = response.json()["types"]
    ids = {t["id"] for t in types}
    assert {"micro_quality", "web", "research", "blender"} <= ids
    for t in types:
        assert t["tasks"]


def test_lab_sandbox_tasks_endpoint(client):
    response = client.get("/api/lab/sandbox-tasks")
    assert response.status_code == 200, response.text
    tasks = response.json()["tasks"]
    ids = {t["id"] for t in tasks}
    assert {"web_dashboard", "research_synthesis", "blender_skyblock"} == ids
    for task in tasks:
        assert task["checks"]
        assert task["max_repair_attempts"] >= 1
        assert task["network_allowed"] is False


def test_lab_sandbox_run_validates_input(client):
    assert client.post("/api/lab/sandbox-run", json={"task": "nope", "model": "m"}).status_code == 400
    assert client.post("/api/lab/sandbox-run", json={"task": "web_dashboard"}).status_code == 400


def test_lab_sandbox_unknown_job(client):
    assert client.get("/api/lab/sandbox-run/not-a-job").status_code == 404


def test_lab_page_renders(client):
    response = client.get("/lab")
    assert response.status_code == 200, response.text
    assert "Lab" in response.text
    assert "/static/js/lab.js" in response.text


def test_lab_nav_present_on_other_pages(client):
    for path in ("/", "/history", "/settings"):
        response = client.get(path)
        if response.status_code != 200:
            continue
        # Lab and experiments now live under one "Brewery" entry.
        assert 'href="/brewery"' in response.text, f"Brewery link missing on {path}"
