"""
Plot generation utilities with async support.

Provides thread-safe matplotlib operations using the application's
thread pool executor.
"""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend for thread safety
matplotlib.use("Agg")

T = TypeVar("T")


class PlotGenerationError(Exception):
    """Raised when plot generation fails."""

    pass


async def run_plot_operation(
    executor: ThreadPoolExecutor,
    plot_func: Callable[[], T],
) -> T:
    """
    Run a matplotlib plotting operation in a thread pool.

    Args:
        executor: Thread pool executor for running the plot operation
        plot_func: Synchronous function that performs plotting operations

    Returns:
        Result from plot_func

    Raises:
        PlotGenerationError: If plot generation fails

    Example:
        >>> async def generate_chart():
        ...     def _plot():
        ...         fig, ax = plt.subplots()
        ...         ax.plot([1, 2, 3], [1, 4, 9])
        ...         return fig
        ...     executor = get_plot_executor()
        ...     return await run_plot_operation(executor, _plot)
    """
    loop = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(executor, plot_func)
        return result
    except Exception as e:
        raise PlotGenerationError(f"Failed to generate plot: {e}") from e


def ensure_plots_directory(base_path: Path = Path("static/plots")) -> Path:
    """
    Ensure the plots directory exists.

    Args:
        base_path: Base directory for plots (default: static/plots)

    Returns:
        Absolute path to the plots directory

    Raises:
        PlotGenerationError: If directory cannot be created
    """
    try:
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path.absolute()
    except Exception as e:
        raise PlotGenerationError(f"Failed to create plots directory: {e}") from e


def cleanup_plot(fig: matplotlib.figure.Figure) -> None:
    """
    Clean up matplotlib figure resources.

    Args:
        fig: Matplotlib figure to clean up

    Note:
        Always call this after saving a plot to prevent memory leaks.
        Use in a try/finally block to ensure cleanup happens.
    """
    try:
        plt.close(fig)
    except Exception:
        # Ignore cleanup errors
        pass


def create_strategy_chart(bins: list, strategy) -> matplotlib.figure.Figure:
    """
    Generate dual-axis chart with price distribution histogram and P&L curve.

    Creates a matplotlib figure with:
    - Primary axis: Histogram showing strike price distribution
    - Secondary axis: P&L curve showing strategy payoff across price range

    Args:
        bins: List of PriceBin objects with lower, upper, count, and midpoint
        strategy: Strategy object with total_payoff(price) method

    Returns:
        Matplotlib Figure object with the dual-axis chart

    Raises:
        PlotGenerationError: If chart generation fails

    Example:
        >>> fig = create_strategy_chart(price_bins, strategy)
        >>> fig.savefig('strategy_analysis.png')
        >>> cleanup_plot(fig)

    Note:
        This function should be called within run_plot_operation() for
        thread-safe async execution.
    """
    try:
        # Validate inputs
        if not bins:
            raise ValueError("Cannot create chart with empty bins list")

        # Create figure and primary axis
        fig, ax1 = plt.subplots(figsize=(12, 7))

        # Extract data from bins
        midpoints = [bin.midpoint for bin in bins]
        counts = [bin.count for bin in bins]
        bin_width = bins[1].upper - bins[1].lower if len(bins) > 1 else 1.0

        # Plot histogram on primary axis
        ax1.bar(
            midpoints,
            counts,
            width=bin_width * 0.9,
            alpha=0.6,
            color='steelblue',
            edgecolor='darkblue',
            linewidth=0.5,
            label='Price Distribution'
        )
        ax1.set_xlabel('Stock Price at Expiration ($)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency (Simulation Count)', fontsize=12, fontweight='bold', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Create secondary axis for P&L curve
        ax2 = ax1.twinx()

        # Generate high-resolution price points for smooth P&L curve
        # Use many more points than histogram bins for accuracy
        price_min = bins[0].lower
        price_max = bins[-1].upper
        price_step = (price_max - price_min) / 500  # 500 points for smooth curve
        smooth_prices = [price_min + i * price_step for i in range(501)]

        # Calculate P&L at each high-resolution price point
        pnl_values = [strategy.total_payoff(price) for price in smooth_prices]

        # Plot P&L curve on secondary axis
        ax2.plot(
            smooth_prices,
            pnl_values,
            color='darkred',
            linewidth=2.5,
            label='P&L Curve',
            marker='',
            linestyle='-'
        )
        ax2.set_ylabel('Profit/Loss ($)', fontsize=12, fontweight='bold', color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred')

        # Add zero line for P&L reference
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.0, alpha=0.5)

        # Set title
        plt.title(
            'Strategy Analysis: Price Distribution & P&L Curve',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

        # Adjust layout to prevent label cutoff
        fig.tight_layout()

        return fig

    except Exception as e:
        raise PlotGenerationError(f"Failed to create strategy chart: {e}") from e


def save_plot(
    fig: matplotlib.figure.Figure,
    session_id: str,
    base_path: Path = Path("static/plots"),
    session=None,
) -> str:
    """
    Save a matplotlib figure to disk with session-based naming.

    Generates a unique filename using the session ID and timestamp,
    then saves the figure to the plots directory.

    Args:
        fig: Matplotlib figure to save
        session_id: Session identifier for filename
        base_path: Base directory for plots (default: static/plots)
        session: Optional SessionState object for tracking plot files

    Returns:
        Relative path to the saved plot file (e.g., "static/plots/abc123_20260103_120530.png")

    Raises:
        PlotGenerationError: If saving fails

    Example:
        >>> fig = create_strategy_chart(bins, strategy)
        >>> path = save_plot(fig, "abc123")
        >>> # Returns: "static/plots/abc123_20260103_120530.png"

    Note:
        - Filename format: {session_id}_{timestamp}.png
        - Timestamp format: YYYYMMDD_HHMMSS
        - Directory is created if it doesn't exist
        - Thread-safe for concurrent writes (unique timestamps)
        - If session is provided, registers file for automatic cleanup
    """
    try:
        # Ensure directory exists
        plots_dir = ensure_plots_directory(base_path)

        # Generate filename with timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{session_id}_{timestamp}.png"
        filepath = plots_dir / filename

        # Save figure with high quality
        fig.savefig(filepath, dpi=150, bbox_inches="tight")

        # Return relative path for URL construction
        plot_path = str(Path(base_path) / filename)

        # Register file with session for cleanup
        if session is not None:
            session.add_plot_file(plot_path)

        return plot_path

    except Exception as e:
        raise PlotGenerationError(f"Failed to save plot: {e}") from e
