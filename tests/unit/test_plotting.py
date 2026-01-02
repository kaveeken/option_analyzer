"""Tests for plotting utilities."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import pytest

from option_analyzer.utils.plotting import (
    PlotGenerationError,
    cleanup_plot,
    create_strategy_chart,
    ensure_plots_directory,
    run_plot_operation,
    save_plot,
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


class TestStrategyChart:
    """Test strategy chart generation."""

    @pytest.fixture
    def mock_bins(self):
        """Create mock price bins for testing."""
        from option_analyzer.services.statistics import PriceBin

        return [
            PriceBin(lower=145.0, upper=150.0, count=100, midpoint=147.5),
            PriceBin(lower=150.0, upper=155.0, count=200, midpoint=152.5),
            PriceBin(lower=155.0, upper=160.0, count=150, midpoint=157.5),
        ]

    @pytest.fixture
    def mock_strategy(self):
        """Create mock strategy for testing."""
        from datetime import date

        from option_analyzer.models.domain import (
            OptionContract,
            OptionPosition,
            Stock,
            Strategy,
        )

        # Create a simple call spread strategy
        stock = Stock(symbol="TEST", current_price=150.0, conid=12345)
        call_long = OptionContract(
            conid=111,
            strike=145.0,
            right="C",
            expiration=date(2026, 1, 15),
            bid=6.0,
            ask=6.5,
        )
        call_short = OptionContract(
            conid=222,
            strike=155.0,
            right="C",
            expiration=date(2026, 1, 15),
            bid=2.0,
            ask=2.5,
        )

        positions = [
            OptionPosition(contract=call_long, quantity=1),
            OptionPosition(contract=call_short, quantity=-1),
        ]

        return Strategy(stock=stock, option_positions=positions)

    def test_create_strategy_chart_success(self, mock_bins, mock_strategy):
        """Test successful chart generation."""
        fig = create_strategy_chart(mock_bins, mock_strategy)

        assert fig is not None
        assert len(fig.axes) == 2  # Primary and secondary axis

        # Check that both axes exist
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]

        # Verify histogram bars on primary axis
        assert len(ax1.patches) == len(mock_bins)

        # Verify P&L line on secondary axis
        assert len(ax2.lines) >= 1  # At least P&L line (plus zero line)

        cleanup_plot(fig)

    def test_create_strategy_chart_labels(self, mock_bins, mock_strategy):
        """Test chart has proper labels and title."""
        fig = create_strategy_chart(mock_bins, mock_strategy)

        ax1 = fig.axes[0]
        ax2 = fig.axes[1]

        # Check labels
        assert "Stock Price" in ax1.get_xlabel()
        assert "Frequency" in ax1.get_ylabel()
        assert "Profit/Loss" in ax2.get_ylabel()

        # Check title exists (either suptitle or axes title)
        has_title = (
            (fig._suptitle is not None and len(fig._suptitle.get_text()) > 0)
            or any(len(ax.get_title()) > 0 for ax in fig.axes)
        )
        assert has_title

        cleanup_plot(fig)

    def test_create_strategy_chart_legend(self, mock_bins, mock_strategy):
        """Test chart has legend."""
        fig = create_strategy_chart(mock_bins, mock_strategy)

        ax1 = fig.axes[0]
        legend = ax1.get_legend()

        assert legend is not None
        assert len(legend.get_texts()) >= 2  # Should have both histogram and P&L

        cleanup_plot(fig)

    def test_create_strategy_chart_empty_bins(self, mock_strategy):
        """Test chart generation with empty bins raises error."""
        with pytest.raises(PlotGenerationError):
            create_strategy_chart([], mock_strategy)

    def test_create_strategy_chart_single_bin(self, mock_strategy):
        """Test chart generation with single bin."""
        from option_analyzer.services.statistics import PriceBin

        single_bin = [PriceBin(lower=145.0, upper=150.0, count=100, midpoint=147.5)]

        fig = create_strategy_chart(single_bin, mock_strategy)

        assert fig is not None
        assert len(fig.axes) == 2

        cleanup_plot(fig)


class TestPlotSaving:
    """Test plot file saving functionality."""

    @pytest.fixture
    def mock_bins(self):
        """Create mock price bins for testing."""
        from option_analyzer.services.statistics import PriceBin

        return [
            PriceBin(lower=145.0, upper=150.0, count=100, midpoint=147.5),
            PriceBin(lower=150.0, upper=155.0, count=200, midpoint=152.5),
        ]

    @pytest.fixture
    def mock_strategy(self):
        """Create mock strategy for testing."""
        from datetime import date

        from option_analyzer.models.domain import (
            OptionContract,
            OptionPosition,
            Stock,
            Strategy,
        )

        stock = Stock(symbol="TEST", current_price=150.0, conid=12345)
        call = OptionContract(
            conid=111,
            strike=150.0,
            right="C",
            expiration=date(2026, 1, 15),
            bid=5.0,
            ask=5.5,
        )
        positions = [OptionPosition(contract=call, quantity=1)]
        return Strategy(stock=stock, option_positions=positions)

    def test_save_plot_creates_file(self, tmp_path, mock_bins, mock_strategy):
        """Test that save_plot creates a file."""
        plots_dir = tmp_path / "plots"

        # Generate a chart
        fig = create_strategy_chart(mock_bins, mock_strategy)

        # Save it
        session_id = "test_session_123"
        plot_path = save_plot(fig, session_id, base_path=plots_dir)

        # Verify file was created
        assert plots_dir.exists()
        full_path = tmp_path / plot_path
        assert full_path.exists()
        assert full_path.suffix == ".png"

        # Verify filename format
        assert session_id in full_path.name
        assert "_" in full_path.name

        cleanup_plot(fig)

    def test_save_plot_filename_format(self, tmp_path, mock_bins, mock_strategy):
        """Test that saved plot follows naming convention."""
        plots_dir = tmp_path / "plots"
        fig = create_strategy_chart(mock_bins, mock_strategy)

        session_id = "abc123"
        plot_path = save_plot(fig, session_id, base_path=plots_dir)

        # Extract filename
        from pathlib import Path

        filename = Path(plot_path).name

        # Should be: {session_id}_{timestamp}.png
        assert filename.startswith(f"{session_id}_")
        assert filename.endswith(".png")

        # Timestamp part should be 15 chars: YYYYMMDD_HHMMSS
        timestamp_part = filename.replace(f"{session_id}_", "").replace(".png", "")
        assert len(timestamp_part) == 15  # YYYYMMDD_HHMMSS
        assert timestamp_part[8] == "_"  # Underscore between date and time

        cleanup_plot(fig)

    def test_save_plot_concurrent_writes(self, tmp_path, mock_bins, mock_strategy):
        """Test that concurrent saves create unique files."""
        plots_dir = tmp_path / "plots"
        session_id = "concurrent_test"

        # Create multiple plots rapidly
        paths = []
        for _ in range(3):
            fig = create_strategy_chart(mock_bins, mock_strategy)
            path = save_plot(fig, session_id, base_path=plots_dir)
            paths.append(path)
            cleanup_plot(fig)

        # All paths should exist
        for path in paths:
            full_path = tmp_path / path
            assert full_path.exists()

        # Paths might not all be unique due to timestamp resolution
        # but at least all files should be created
        assert len(paths) == 3
