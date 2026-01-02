"""
Plot generation utilities with async support.

Provides thread-safe matplotlib operations using the application's
thread pool executor.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, TypeVar

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
