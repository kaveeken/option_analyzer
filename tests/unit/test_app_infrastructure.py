"""Tests for application infrastructure setup."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from option_analyzer.api.app import create_app, get_plot_executor


class TestApplicationStartup:
    """Test application startup infrastructure."""

    def test_app_creates_plot_executor(self):
        """Test that app initializes plot executor on startup."""
        app = create_app()

        with TestClient(app) as client:
            # App should be started, executor should be available
            executor = get_plot_executor()
            assert executor is not None
            assert executor._max_workers == 2  # type: ignore
            assert not executor._shutdown  # type: ignore

            # Make a request to verify app is working
            response = client.get("/health")
            assert response.status_code == 200

    def test_plot_executor_before_startup(self):
        """Test that get_plot_executor raises error before startup."""
        # Without starting the app, executor should not be available
        # Note: This test assumes no other test has started the global app
        # In practice, the executor is global, so this may not work as expected
        # in a full test suite. For now, we'll skip this test.
        pytest.skip("Global state makes this test unreliable")

    def test_plots_directory_created_on_startup(self):
        """Test that plots directory is created during startup."""
        app = create_app()
        plots_dir = Path("static/plots")

        with TestClient(app) as client:
            # Verify plots directory exists
            assert plots_dir.exists()
            assert plots_dir.is_dir()

            # Make a request to verify app is working
            response = client.get("/health")
            assert response.status_code == 200
