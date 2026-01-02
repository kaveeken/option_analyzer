"""Tests for plotting utilities."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from option_analyzer.utils.plotting import (
    PlotGenerationError,
    cleanup_plot,
    ensure_plots_directory,
    run_plot_operation,
)


@pytest.fixture
def plot_executor():
    """Create a thread pool executor for tests."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-plot")
    yield executor
    executor.shutdown(wait=True)


@pytest.fixture
def temp_plots_dir(tmp_path):
    """Create a temporary plots directory."""
    plots_dir = tmp_path / "plots"
    return plots_dir


class TestPlotOperations:
    """Test async plot operations."""

    async def test_run_plot_operation_success(self, plot_executor):
        """Test successful plot generation."""

        def create_simple_plot():
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 4, 9])
            return fig

        fig = await run_plot_operation(plot_executor, create_simple_plot)

        assert fig is not None
        assert len(fig.axes) == 1

        cleanup_plot(fig)

    async def test_run_plot_operation_failure(self, plot_executor):
        """Test plot generation with error."""

        def failing_plot():
            raise ValueError("Intentional error")

        with pytest.raises(PlotGenerationError) as exc_info:
            await run_plot_operation(plot_executor, failing_plot)

        assert "Failed to generate plot" in str(exc_info.value)
        assert "Intentional error" in str(exc_info.value)

    async def test_concurrent_plot_operations(self, plot_executor):
        """Test multiple concurrent plot operations."""

        def create_plot(value):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [value, value * 2, value * 3])
            return fig

        # Create multiple plots concurrently
        tasks = [
            run_plot_operation(plot_executor, lambda v=i: create_plot(v))
            for i in range(1, 4)
        ]

        figs = await asyncio.gather(*tasks)

        assert len(figs) == 3
        for fig in figs:
            assert fig is not None
            cleanup_plot(fig)


class TestDirectoryManagement:
    """Test plot directory management."""

    def test_ensure_plots_directory_creates_dir(self, temp_plots_dir):
        """Test directory creation."""
        assert not temp_plots_dir.exists()

        result = ensure_plots_directory(temp_plots_dir)

        assert temp_plots_dir.exists()
        assert temp_plots_dir.is_dir()
        assert result == temp_plots_dir.absolute()

    def test_ensure_plots_directory_exists(self, temp_plots_dir):
        """Test with existing directory."""
        temp_plots_dir.mkdir(parents=True)
        assert temp_plots_dir.exists()

        result = ensure_plots_directory(temp_plots_dir)

        assert temp_plots_dir.exists()
        assert result == temp_plots_dir.absolute()

    def test_ensure_plots_directory_creates_parents(self, tmp_path):
        """Test nested directory creation."""
        nested_dir = tmp_path / "parent" / "child" / "plots"
        assert not nested_dir.exists()

        result = ensure_plots_directory(nested_dir)

        assert nested_dir.exists()
        assert result == nested_dir.absolute()


class TestCleanup:
    """Test plot cleanup."""

    def test_cleanup_plot_success(self):
        """Test successful cleanup."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        # Should not raise
        cleanup_plot(fig)

        # Verify figure is closed
        assert plt.fignum_exists(fig.number) is False

    def test_cleanup_plot_already_closed(self):
        """Test cleanup of already closed figure."""
        fig, ax = plt.subplots()
        plt.close(fig)

        # Should not raise even if already closed
        cleanup_plot(fig)
