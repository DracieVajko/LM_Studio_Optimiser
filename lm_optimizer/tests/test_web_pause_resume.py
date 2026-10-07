"""Tests for Web UI Pause/Resume functionality."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from fastapi.testclient import TestClient

from lm_optimizer.api.main import app
from lm_optimizer.api.main import app as fastapi_app


class TestWebPauseResume:
    """Tests for web UI pause/resume functionality."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @patch("lm_optimizer.api.routes.run_repo")
    @patch("lm_optimizer.api.routes.checkpoint_dir")
    @patch("lm_optimizer.api.routes.Path")
    @patch("lm_optimizer.api.routes.orjson")
    @patch("lm_optimizer.api.routes.datetime")
    def test_pause_run_endpoint(self, mock_datetime, mock_orjson, mock_path, mock_checkpoint_dir, mock_run_repo, client):
        """Test POST /api/runs/{run_id}/pause endpoint."""
        run_id = uuid4()
        mock_run = MagicMock()
        mock_run.status = "running"
        mock_run.id = uuid4()
        mock_run_repo.get.return_value = mock_run

        with patch("lm_optimizer.api.routes.checkpoint_dir") as mock_checkpoint_dir, \
             patch("lm_optimizer.api.routes.Path") as mock_path, \
             patch("lm_optimizer.api.routes.orjson") as mock_orjson, \
             patch("lm_optimizer.api.routes.datetime") as mock_datetime:

            mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
            mock_path.return_value = MagicMock()

            response = client.post(f"/api/runs/{uuid4()}/pause")

            assert response.status_code == 200
            assert response.json()["success"] is True
            assert "Pause requested" in response.json()["message"]

    @patch("lm_optimizer.api.routes.run_repo")
    def test_pause_not_found(self, mock_run_repo, client):
        """Test pause on non-existent run returns 404."""
        run_repo = MagicMock()
        run_repo.get.return_value = None

        with patch("lm_optimizer.api.routes.run_repo", run_repo):
            response = client.post(f"/api/runs/{uuid4()}/pause")
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    @patch("lm_optimizer.api.routes.run_repo")
    def test_pause_non_running(self, mock_run_repo, client):
        """Test pause on non-running run returns 409."""
        mock_run = MagicMock()
        mock_run.status = "paused"
        run_repo = MagicMock()
        run_repo.get.return_value = MagicMock(status="paused")

        with patch("lm_optimizer.api.routes.run_repo", run_repo):
            response = client.post(f"/api/runs/{uuid4()}/pause")
            assert response.status_code == 409
            assert "not running" in response.json()["detail"].lower()

    @patch("lm_optimizer.api.routes.run_repo")
    @patch("lm_optimizer.api.routes.load_checkpoint")
    @patch("lm_optimizer.api.routes.BackgroundTasks")
    @patch("lm_optimizer.api.routes.resume_run_from_checkpoint")
    def test_resume_run_endpoint(self, mock_resume_from_checkpoint, mock_background_tasks, mock_load_checkpoint, mock_run_repo, client):
        """Test POST /api/runs/{run_id}/resume endpoint."""
        run_id = uuid4()
        mock_run = MagicMock()
        mock_run.status = "paused"
        mock_run.id = uuid4()
        mock_run_repo.get.return_value = MagicMock(id=uuid4(), status="paused")
        mock_load_checkpoint.return_value = {"completed_candidate_ids": [1, 2, 3]}

        with patch("lm_optimizer.api.routes.resume_run_from_checkpoint") as mock_resume:
            mock_resume.return_value = {"completed_candidate_ids": [1, 2, 3]}
            response = client.post(f"/api/runs/{uuid4()}/resume")
            assert response.status_code == 200
            assert response.json()["success"] is True

    @patch("lm_optimizer.api.routes.run_repo")
    def test_resume_not_found(self, mock_run_repo, client):
        """Test resume on non-existent run returns 404."""
        run_repo = MagicMock()
        run_repo.get.return_value = None

        with patch("lm_optimizer.api.routes.run_repo", run_repo):
            response = client.post(f"/api/runs/{uuid4()}/resume")
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    @patch("lm_optimizer.api.routes.run_repo")
    @patch("lm_optimizer.api.routes.load_checkpoint")
    def test_resume_not_paused(self, mock_load_checkpoint, mock_run_repo, client):
        """Test resume on non-paused run returns 409."""
        mock_run = MagicMock()
        mock_run.status = "running"
        run_repo = MagicMock()
        run_repo.get.return_value = MagicMock(status="running")
        mock_load_checkpoint.return_value = {"completed_candidate_ids": []}

        with patch("lm_optimizer.api.routes.run_repo", run_repo), \
             patch("lm_optimizer.api.routes.load_checkpoint", mock_load_checkpoint):
            response = client.post(f"/api/runs/{uuid4()}/resume")
            assert response.status_code == 409
            assert "not paused" in response.json()["detail"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])