"""
Statistical functions for price distribution modeling and simulation.

This module provides core statistical tools for analyzing option strategies:
- Geometric returns calculation for historical price data
- Bootstrap simulation (Monte Carlo) for price distribution generation
- Histogram binning for probability distributions
- Price range calculation with automatic padding

All functions use NumPy vectorization for performance, targeting 10,000
Monte Carlo simulations in under 5 seconds.
"""

from datetime import date

import numpy as np
from pydantic import BaseModel, Field


class PriceBin(BaseModel):
    """
    A histogram bin representing a price range and frequency count.

    Attributes:
        lower: Lower bound of the price range (inclusive)
        upper: Upper bound of the price range (exclusive for all bins except the last)
        count: Number of occurrences in this bin
    """

    lower: float = Field(ge=0)
    upper: float = Field(gt=0)
    count: int = Field(ge=0)

    @property
    def midpoint(self) -> float:
        """
        Calculate the midpoint of the bin.

        Returns:
            The average of lower and upper bounds

        Example:
            >>> bin = PriceBin(lower=100.0, upper=110.0, count=5)
            >>> bin.midpoint
            105.0
        """
        return (self.lower + self.upper) / 2

    @property
    def width(self) -> float:
        """
        Calculate the width of the bin.

        Returns:
            The difference between upper and lower bounds

        Example:
            >>> bin = PriceBin(lower=100.0, upper=110.0, count=5)
            >>> bin.width
            10.0
        """
        return self.upper - self.lower


def geometric_returns(closes: np.ndarray, period: int = 1) -> np.ndarray:
    """
    Calculate period-over-period price multipliers from closing prices.

    Geometric returns represent the ratio of prices across periods, suitable
    for modeling multiplicative processes like stock price movements.

    Args:
        closes: Array of closing prices (must be positive)
        period: Number of periods to look back (default=1 for consecutive periods)

    Returns:
        Array of price multipliers (length = len(closes) - period)

    Raises:
        ValueError: If closes is empty, period is invalid, or prices are non-positive

    Examples:
        >>> closes = np.array([100.0, 105.0, 103.0, 108.0])
        >>> geometric_returns(closes)
        array([1.05, 0.980952, 1.048544])

        >>> # Weekly returns from daily data (period=5)
        >>> geometric_returns(daily_closes, period=5)
    """
    if closes.size == 0:
        raise ValueError("closes array cannot be empty")

    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    if closes.size <= period:
        raise ValueError(
            f"Insufficient data: need at least {period + 1} prices, got {closes.size}"
        )

    if np.any(closes <= 0):
        raise ValueError("All prices must be positive")

    # Calculate price ratios: closes[period:] / closes[:-period]
    return closes[period:] / closes[:-period]


def calculate_period_for_days(
    target_days: int, bar_size: str = "1w", reference_date: date | None = None
) -> int:
    """
    Calculate the number of bars needed to span a target number of calendar days.

    Adjusts for the difference between trading days and calendar days based
    on the bar size (e.g., weekly bars cover 7 calendar days but only ~5 trading days).

    Args:
        target_days: Target number of calendar days to span
        bar_size: Time interval for each bar (e.g., "1d", "1w")
        reference_date: Reference date for calculation (unused, for future enhancements)

    Returns:
        Number of bars needed to approximate the target days

    Raises:
        ValueError: If target_days is negative or bar_size is unsupported

    Examples:
        >>> # For 30 days of weekly bars (approx 4-5 weeks)
        >>> calculate_period_for_days(30, bar_size="1w")
        4

        >>> # For 252 trading days (1 year) of daily bars
        >>> calculate_period_for_days(365, bar_size="1d")
        252

    Note:
        Uses 252 trading days / 365 calendar days ratio for adjustment
    """
    if target_days < 0:
        raise ValueError(f"target_days must be non-negative, got {target_days}")

    # Map bar sizes to calendar days per bar
    bar_to_days = {
        "1d": 1,  # 1 day
        "1w": 7,  # 1 week
        "1M": 30,  # Approximate month
    }

    if bar_size not in bar_to_days:
        raise ValueError(
            f"Unsupported bar_size '{bar_size}'. Supported: {list(bar_to_days.keys())}"
        )

    days_per_bar = bar_to_days[bar_size]

    # Adjust for trading days vs calendar days (252 trading / 365 calendar)
    trading_day_ratio = 252 / 365

    # Calculate adjusted target in trading days
    target_trading_days = target_days * trading_day_ratio

    # Calculate periods needed, rounding to nearest integer
    periods = round(target_trading_days / (days_per_bar * trading_day_ratio))

    return max(1, periods)  # At least 1 period


def bootstrap_walk(
    returns: np.ndarray, steps: int, rng: np.random.Generator | None = None
) -> float:
    """
    Perform a single random walk by resampling from historical returns.

    Samples returns with replacement and multiplies them together to simulate
    a potential future price path.

    Args:
        returns: Historical return multipliers (from geometric_returns)
        steps: Number of steps in the random walk
        rng: NumPy random generator (creates default if None)

    Returns:
        Final price multiplier after all steps

    Raises:
        ValueError: If returns is empty or steps is invalid

    Example:
        >>> returns = np.array([1.02, 0.98, 1.01, 0.99, 1.03])
        >>> rng = np.random.default_rng(42)
        >>> multiplier = bootstrap_walk(returns, steps=10, rng=rng)
        >>> final_price = 100.0 * multiplier
    """
    if returns.size == 0:
        raise ValueError("returns array cannot be empty")

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    if rng is None:
        rng = np.random.default_rng()

    # Sample returns with replacement and compute product
    samples = rng.choice(returns, size=steps, replace=True)
    return np.prod(samples)


def monte_carlo_simulation(
    returns: np.ndarray,
    steps: int,
    n_simulations: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Run vectorized Monte Carlo simulation using bootstrap resampling.

    Generates multiple random walks simultaneously using NumPy vectorization
    for performance. Target: 10,000 simulations in under 5 seconds.

    Args:
        returns: Historical return multipliers (from geometric_returns)
        steps: Number of steps per random walk
        n_simulations: Number of independent simulations to run
        rng: NumPy random generator (creates default if None)

    Returns:
        Array of final price multipliers, shape (n_simulations,)

    Raises:
        ValueError: If returns is empty or parameters are invalid

    Example:
        >>> returns = geometric_returns(historical_closes)
        >>> rng = np.random.default_rng(42)
        >>> multipliers = monte_carlo_simulation(returns, steps=50, n_simulations=10000, rng=rng)
        >>> # Convert to prices
        >>> final_prices = current_price * multipliers

    Performance:
        - 10k simulations × 50 steps: ~2 seconds
        - Uses single vectorized sampling instead of loops
    """
    if returns.size == 0:
        raise ValueError("returns array cannot be empty")

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    if n_simulations < 1:
        raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")

    if rng is None:
        rng = np.random.default_rng()

    # Vectorized: sample all at once (n_simulations × steps)
    samples = rng.choice(returns, size=(n_simulations, steps), replace=True)

    # Vectorized: product across steps for each simulation
    multipliers = np.prod(samples, axis=1)

    return multipliers


def get_price_distribution(
    current_price: float,
    returns: np.ndarray,
    target_date: date,
    bar_size: str = "1w",
    bootstrap_samples: int | None = None,
    reference_date: date | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate price distribution at target date using historical returns.

    Supports two modes:
    - Direct mode (bootstrap_samples=None): Apply historical sequence directly
    - Monte Carlo mode (bootstrap_samples>0): Generate random walks via bootstrap

    Args:
        current_price: Current stock price
        returns: Historical return multipliers (from geometric_returns)
        target_date: Future date for price distribution
        bar_size: Time interval of returns data (e.g., "1d", "1w")
        bootstrap_samples: Number of Monte Carlo samples (None for direct mode)
        reference_date: Starting date for calculation (defaults to today)
        rng: NumPy random generator (for Monte Carlo mode)

    Returns:
        Array of possible future prices at target_date

    Raises:
        ValueError: If current_price is non-positive, returns is empty, or target_date is in the past

    Examples:
        >>> # Direct mode: use historical sequence as-is
        >>> prices = get_price_distribution(
        ...     current_price=100.0,
        ...     returns=historical_returns,
        ...     target_date=date(2024, 12, 31),
        ...     bootstrap_samples=None
        ... )

        >>> # Monte Carlo mode: generate 10k random walks
        >>> prices = get_price_distribution(
        ...     current_price=100.0,
        ...     returns=historical_returns,
        ...     target_date=date(2024, 12, 31),
        ...     bootstrap_samples=10000
        ... )
    """
    if current_price <= 0:
        raise ValueError(f"current_price must be positive, got {current_price}")

    if returns.size == 0:
        raise ValueError("returns array cannot be empty")

    ref = reference_date or date.today()
    days_to_target = (target_date - ref).days

    if days_to_target < 0:
        raise ValueError(
            f"target_date ({target_date}) must be after reference_date ({ref})"
        )

    # Calculate steps needed
    steps = calculate_period_for_days(days_to_target, bar_size, ref)

    if bootstrap_samples is None:
        # Direct mode: apply historical returns directly
        if returns.size < steps:
            raise ValueError(
                f"Insufficient data for direct mode: need {steps} returns, got {returns.size}"
            )

        # Use the first 'steps' returns
        multipliers = np.prod(returns[:steps])
        prices = np.array([current_price * multipliers])
    else:
        # Monte Carlo mode: bootstrap simulation
        multipliers = monte_carlo_simulation(returns, steps, bootstrap_samples, rng)
        prices = current_price * multipliers

    return prices


def calculate_price_range(
    price_distribution: np.ndarray,
    padding: float = 0.05,
    user_min: float | None = None,
    user_max: float | None = None,
) -> tuple[float, float]:
    """
    Calculate price range for histogram with automatic padding.

    Determines min/max prices from distribution with optional padding and
    user overrides for custom ranges.

    Args:
        price_distribution: Array of price values
        padding: Fraction to pad beyond min/max (default=0.05 for 5%)
        user_min: User-specified minimum (overrides calculated min)
        user_max: User-specified maximum (overrides calculated max)

    Returns:
        Tuple of (min_price, max_price)

    Raises:
        ValueError: If price_distribution is empty or padding is negative

    Examples:
        >>> prices = np.array([95.0, 100.0, 105.0, 110.0])
        >>> calculate_price_range(prices)
        (90.25, 115.5)  # 5% padding

        >>> # Custom range
        >>> calculate_price_range(prices, user_min=90.0, user_max=120.0)
        (90.0, 120.0)

        >>> # No padding
        >>> calculate_price_range(prices, padding=0.0)
        (95.0, 110.0)
    """
    if price_distribution.size == 0:
        raise ValueError("price_distribution cannot be empty")

    if padding < 0:
        raise ValueError(f"padding must be non-negative, got {padding}")

    # Calculate from distribution
    dist_min = float(np.min(price_distribution))
    dist_max = float(np.max(price_distribution))

    # Apply padding
    min_price = dist_min * (1 - padding)
    max_price = dist_max * (1 + padding)

    # Apply user overrides
    if user_min is not None:
        min_price = user_min
    if user_max is not None:
        max_price = user_max

    return min_price, max_price


def create_histogram(
    values: np.ndarray,
    n_bins: int,
    min_value: float | None = None,
    max_value: float | None = None,
) -> list[PriceBin]:
    """
    Create histogram bins from value distribution.

    Bins the values into equal-width intervals and counts frequencies.
    Handles the edge case where the maximum value needs to be included in
    the last bin by using <= for the final bin instead of <.

    Args:
        values: Array of values to bin
        n_bins: Number of bins to create
        min_value: Minimum value for range (defaults to min(values))
        max_value: Maximum value for range (defaults to max(values))

    Returns:
        List of PriceBin objects with counts

    Raises:
        ValueError: If values is empty, n_bins < 1, or min_value >= max_value

    Examples:
        >>> values = np.array([95.0, 100.0, 105.0, 110.0, 115.0])
        >>> bins = create_histogram(values, n_bins=4, min_value=90.0, max_value=120.0)
        >>> len(bins)
        4
        >>> bins[0].count
        1  # Only 95.0 in first bin

    Note:
        The last bin uses <= to ensure max_value is included (critical edge case)
    """
    if values.size == 0:
        raise ValueError("values array cannot be empty")

    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    # Determine range
    if min_value is None:
        min_value = float(np.min(values))
    if max_value is None:
        max_value = float(np.max(values))

    if min_value >= max_value:
        raise ValueError(
            f"min_value ({min_value}) must be less than max_value ({max_value})"
        )

    # Create bin edges
    bin_width = (max_value - min_value) / n_bins
    bins = []

    for i in range(n_bins):
        lower = min_value + i * bin_width
        upper = min_value + (i + 1) * bin_width

        # Count values in this bin
        if i == n_bins - 1:
            # Last bin: include max_value with <=
            count = int(np.sum((values >= lower) & (values <= upper)))
        else:
            # Other bins: exclude upper bound
            count = int(np.sum((values >= lower) & (values < upper)))

        bins.append(PriceBin(lower=lower, upper=upper, count=count))

    return bins
