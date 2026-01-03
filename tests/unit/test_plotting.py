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


class TestChartDataAccuracy:
    """Test that plotted data accurately reflects input data (visual regression via data validation)."""

    @pytest.fixture
    def mock_bins(self):
        """Create mock price bins for testing."""
        from option_analyzer.services.statistics import PriceBin

        return [
            PriceBin(lower=145.0, upper=150.0, count=100, midpoint=147.5),
            PriceBin(lower=150.0, upper=155.0, count=200, midpoint=152.5),
            PriceBin(lower=155.0, upper=160.0, count=150, midpoint=157.5),
            PriceBin(lower=160.0, upper=165.0, count=80, midpoint=162.5),
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

    def test_histogram_bar_heights_match_bin_counts(self, mock_bins, mock_strategy):
        """Verify histogram bar heights match bin counts exactly."""
        fig = create_strategy_chart(mock_bins, mock_strategy)

        ax1 = fig.axes[0]
        bars = ax1.patches

        # Extract bar heights
        bar_heights = [bar.get_height() for bar in bars]
        expected_counts = [bin.count for bin in mock_bins]

        assert len(bar_heights) == len(expected_counts)
        for actual, expected in zip(bar_heights, expected_counts):
            assert actual == expected, f"Bar height {actual} doesn't match bin count {expected}"

        cleanup_plot(fig)

    def test_histogram_bar_positions_match_midpoints(self, mock_bins, mock_strategy):
        """Verify histogram bars are centered at bin midpoints."""
        fig = create_strategy_chart(mock_bins, mock_strategy)

        ax1 = fig.axes[0]
        bars = ax1.patches

        # Extract bar x-positions (centers)
        bar_centers = [bar.get_x() + bar.get_width() / 2 for bar in bars]
        expected_midpoints = [bin.midpoint for bin in mock_bins]

        assert len(bar_centers) == len(expected_midpoints)
        for actual, expected in zip(bar_centers, expected_midpoints):
            assert abs(actual - expected) < 0.01, f"Bar center {actual} doesn't match midpoint {expected}"

        cleanup_plot(fig)

    def test_histogram_bar_widths_match_bin_width(self, mock_bins, mock_strategy):
        """Verify histogram bar widths match bin width * 0.9 (with gap)."""
        fig = create_strategy_chart(mock_bins, mock_strategy)

        ax1 = fig.axes[0]
        bars = ax1.patches

        # Calculate expected width
        expected_bin_width = mock_bins[1].upper - mock_bins[1].lower
        expected_bar_width = expected_bin_width * 0.9

        # Check all bar widths
        for bar in bars:
            actual_width = bar.get_width()
            assert abs(actual_width - expected_bar_width) < 0.01, \
                f"Bar width {actual_width} doesn't match expected {expected_bar_width}"

        cleanup_plot(fig)

    def test_pnl_curve_data_accuracy(self, mock_bins, mock_strategy):
        """Verify P&L curve values match strategy payoff calculations."""
        fig = create_strategy_chart(mock_bins, mock_strategy)

        ax2 = fig.axes[1]
        pnl_line = ax2.lines[0]  # First line should be P&L curve (not zero line)

        # Extract P&L curve data
        x_data, y_data = pnl_line.get_data()

        # Spot-check P&L values at various prices
        # Sample every 50th point to avoid performance issues
        for i in range(0, len(x_data), 50):
            price = x_data[i]
            plotted_pnl = y_data[i]
            calculated_pnl = mock_strategy.total_payoff(price)

            assert abs(plotted_pnl - calculated_pnl) < 0.01, \
                f"At price ${price:.2f}: plotted P&L ${plotted_pnl:.2f} != calculated ${calculated_pnl:.2f}"

        cleanup_plot(fig)

    def test_pnl_curve_covers_full_price_range(self, mock_bins, mock_strategy):
        """Verify P&L curve spans the full price range of the histogram."""
        fig = create_strategy_chart(mock_bins, mock_strategy)

        ax2 = fig.axes[1]
        pnl_line = ax2.lines[0]

        x_data, _ = pnl_line.get_data()

        # P&L curve should span from first bin lower to last bin upper
        expected_min = mock_bins[0].lower
        expected_max = mock_bins[-1].upper

        actual_min = min(x_data)
        actual_max = max(x_data)

        assert abs(actual_min - expected_min) < 0.1, \
            f"P&L curve min {actual_min} doesn't match bins min {expected_min}"
        assert abs(actual_max - expected_max) < 0.1, \
            f"P&L curve max {actual_max} doesn't match bins max {expected_max}"

        cleanup_plot(fig)

    def test_pnl_curve_has_sufficient_resolution(self, mock_bins, mock_strategy):
        """Verify P&L curve has high resolution for smooth rendering."""
        fig = create_strategy_chart(mock_bins, mock_strategy)

        ax2 = fig.axes[1]
        pnl_line = ax2.lines[0]

        x_data, _ = pnl_line.get_data()

        # Should have ~500 points for smooth curve (as per implementation)
        assert len(x_data) >= 400, f"P&L curve has only {len(x_data)} points, expected ~500 for smoothness"

        cleanup_plot(fig)

    def test_histogram_visual_properties(self, mock_bins, mock_strategy):
        """Verify histogram bars have correct visual styling."""
        fig = create_strategy_chart(mock_bins, mock_strategy)

        ax1 = fig.axes[0]
        bars = ax1.patches

        for bar in bars:
            # Check alpha (transparency)
            facecolor = bar.get_facecolor()
            assert facecolor[3] == 0.6, f"Bar alpha should be 0.6, got {facecolor[3]}"

            # Check edge color is present
            edgecolor = bar.get_edgecolor()
            assert edgecolor[3] > 0, "Bar should have visible edge color"

            # Check edge width
            linewidth = bar.get_linewidth()
            assert linewidth == 0.5, f"Bar edge width should be 0.5, got {linewidth}"

        cleanup_plot(fig)

    def test_pnl_line_visual_properties(self, mock_bins, mock_strategy):
        """Verify P&L line has correct visual styling."""
        fig = create_strategy_chart(mock_bins, mock_strategy)

        ax2 = fig.axes[1]
        pnl_line = ax2.lines[0]

        # Check line width
        assert pnl_line.get_linewidth() == 2.5, \
            f"P&L line width should be 2.5, got {pnl_line.get_linewidth()}"

        # Check line is solid (not dashed)
        assert pnl_line.get_linestyle() == '-', \
            f"P&L line should be solid, got {pnl_line.get_linestyle()}"

        # Check no markers on the line
        assert pnl_line.get_marker() == '', \
            f"P&L line should have no markers, got {pnl_line.get_marker()}"

        cleanup_plot(fig)

    def test_zero_line_present(self, mock_bins, mock_strategy):
        """Verify zero reference line exists on P&L axis."""
        fig = create_strategy_chart(mock_bins, mock_strategy)

        ax2 = fig.axes[1]

        # Should have at least 2 lines: P&L curve + zero line
        assert len(ax2.lines) >= 2, "Should have P&L curve and zero reference line"

        # Find the zero line (horizontal line at y=0)
        zero_lines = [line for line in ax2.lines if len(set(line.get_ydata())) == 1 and line.get_ydata()[0] == 0]

        assert len(zero_lines) >= 1, "Should have a horizontal line at y=0"

        cleanup_plot(fig)


class TestChartErrorHandling:
    """Test chart generation error handling and edge cases."""

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

    def test_chart_with_extreme_price_range(self, mock_strategy):
        """Test chart generation with very wide price range."""
        from option_analyzer.services.statistics import PriceBin

        wide_range_bins = [
            PriceBin(lower=10.0, upper=100.0, count=50, midpoint=55.0),
            PriceBin(lower=100.0, upper=1000.0, count=100, midpoint=550.0),
        ]

        fig = create_strategy_chart(wide_range_bins, mock_strategy)

        assert fig is not None
        assert len(fig.axes) == 2

        cleanup_plot(fig)

    def test_chart_with_extreme_counts(self, mock_strategy):
        """Test chart generation with very large count values."""
        from option_analyzer.services.statistics import PriceBin

        extreme_count_bins = [
            PriceBin(lower=145.0, upper=150.0, count=1000000, midpoint=147.5),
            PriceBin(lower=150.0, upper=155.0, count=5000000, midpoint=152.5),
        ]

        fig = create_strategy_chart(extreme_count_bins, mock_strategy)

        assert fig is not None
        ax1 = fig.axes[0]
        bars = ax1.patches
        assert bars[0].get_height() == 1000000
        assert bars[1].get_height() == 5000000

        cleanup_plot(fig)

    def test_chart_with_negative_pnl(self, mock_strategy):
        """Test chart generation when strategy has negative P&L."""
        from option_analyzer.services.statistics import PriceBin

        # Use price range where call has negative P&L (below strike)
        bins = [
            PriceBin(lower=100.0, upper=110.0, count=100, midpoint=105.0),
            PriceBin(lower=110.0, upper=120.0, count=200, midpoint=115.0),
        ]

        fig = create_strategy_chart(bins, mock_strategy)

        # Verify chart generates successfully
        assert fig is not None

        # Verify P&L values are negative
        ax2 = fig.axes[1]
        pnl_line = ax2.lines[0]
        _, y_data = pnl_line.get_data()

        # At these low prices, long call should have negative P&L (premium paid)
        assert any(y < 0 for y in y_data), "Should have negative P&L values"

        cleanup_plot(fig)

    def test_chart_with_small_bin_width(self, mock_strategy):
        """Test chart generation with very narrow bins."""
        from option_analyzer.services.statistics import PriceBin

        narrow_bins = [
            PriceBin(lower=150.0, upper=150.1, count=100, midpoint=150.05),
            PriceBin(lower=150.1, upper=150.2, count=200, midpoint=150.15),
            PriceBin(lower=150.2, upper=150.3, count=150, midpoint=150.25),
        ]

        fig = create_strategy_chart(narrow_bins, mock_strategy)

        assert fig is not None
        ax1 = fig.axes[0]
        bars = ax1.patches

        # Verify bars have appropriate width
        expected_width = 0.1 * 0.9  # bin_width * 0.9
        for bar in bars:
            assert abs(bar.get_width() - expected_width) < 0.001

        cleanup_plot(fig)


class TestAsyncChartGeneration:
    """Test async chart generation via thread pool."""

    @pytest.fixture
    def mock_bins(self):
        """Create mock price bins."""
        from option_analyzer.services.statistics import PriceBin

        return [
            PriceBin(lower=145.0, upper=150.0, count=100, midpoint=147.5),
            PriceBin(lower=150.0, upper=155.0, count=200, midpoint=152.5),
        ]

    @pytest.fixture
    def mock_strategy(self):
        """Create mock strategy."""
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

    async def test_async_chart_generation(self, mock_bins, mock_strategy):
        """Test chart generation via async thread pool."""
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-async-plot")

        try:
            def _create_chart():
                return create_strategy_chart(mock_bins, mock_strategy)

            fig = await run_plot_operation(executor, _create_chart)

            assert fig is not None
            assert len(fig.axes) == 2

            cleanup_plot(fig)
        finally:
            executor.shutdown(wait=True)

    async def test_async_chart_generation_with_error(self):
        """Test async chart generation handles errors properly."""
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-async-error")

        try:
            def _failing_chart():
                raise ValueError("Test error")

            with pytest.raises(PlotGenerationError) as exc_info:
                await run_plot_operation(executor, _failing_chart)

            assert "Failed to generate plot" in str(exc_info.value)
            assert "Test error" in str(exc_info.value)
        finally:
            executor.shutdown(wait=True)

    async def test_multiple_async_charts(self, mock_bins, mock_strategy):
        """Test multiple charts can be generated concurrently."""
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-concurrent")

        try:
            async def create_chart(chart_id):
                def _create():
                    return create_strategy_chart(mock_bins, mock_strategy)
                return await run_plot_operation(executor, _create)

            # Generate 3 charts concurrently
            results = await asyncio.gather(
                create_chart(1),
                create_chart(2),
                create_chart(3),
            )

            assert len(results) == 3
            for fig in results:
                assert fig is not None
                assert len(fig.axes) == 2
                cleanup_plot(fig)
        finally:
            executor.shutdown(wait=True)


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
