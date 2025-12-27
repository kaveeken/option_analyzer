"""
Risk metrics calculation for option strategies.

This module provides probability-weighted risk analysis based on price distributions:
- Expected Value (EV): Probability-weighted average P&L
- Probability of Profit (PoP): Likelihood of positive P&L at expiration
- Max gain/loss: Extreme P&L scenarios
- Breakeven prices: Prices where P&L = 0

All calculations use Strategy.total_payoff() which includes transaction costs.
"""

from pydantic import BaseModel, Field, field_validator

from ..models.domain import Strategy
from .statistics import PriceBin


class RiskMetrics(BaseModel):
    """
    Risk analysis results for an option strategy.

    Attributes:
        expected_value: Probability-weighted average P&L in dollars
        probability_of_profit: Fraction of outcomes with P&L > 0 (0.0 to 1.0)
        max_gain: Maximum possible profit (None if unlimited upside)
        max_loss: Maximum possible loss (None if unlimited downside)
        breakevens: List of breakeven prices at expiration
    """

    expected_value: float
    probability_of_profit: float = Field(ge=0.0, le=1.0)
    max_gain: float | None = None
    max_loss: float | None = None
    breakevens: list[float] = Field(default_factory=list)

    @field_validator("probability_of_profit")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Ensure probability is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"probability_of_profit must be between 0 and 1, got {v}")
        return v


def calculate_expected_value(bins: list[PriceBin], strategy: Strategy) -> float:
    """
    Calculate probability-weighted expected value of a strategy.

    Uses the midpoint of each histogram bin as the representative price and
    weights the P&L by the empirical probability (bin.count / total_count).

    Args:
        bins: Histogram bins representing price distribution
        strategy: Option strategy to analyze

    Returns:
        Expected value in dollars (can be negative)

    Raises:
        ValueError: If bins is empty or all bins have zero count

    Examples:
        >>> bins = [
        ...     PriceBin(lower=95.0, upper=100.0, count=20),
        ...     PriceBin(lower=100.0, upper=105.0, count=30),
        ...     PriceBin(lower=105.0, upper=110.0, count=50),
        ... ]
        >>> ev = calculate_expected_value(bins, strategy)
        >>> print(f"Expected Value: ${ev:.2f}")

    Note:
        Transaction costs are included via strategy.total_payoff()
    """
    if not bins:
        raise ValueError("bins cannot be empty")

    total_count = sum(bin.count for bin in bins)

    if total_count == 0:
        raise ValueError("All bins have zero count")

    expected_value = 0.0

    for bin in bins:
        # Calculate P&L at bin midpoint (includes transaction costs)
        pnl = strategy.total_payoff(bin.midpoint)

        # Weight by empirical probability
        probability = bin.count / total_count
        expected_value += pnl * probability

    return expected_value


def calculate_probability_of_profit(bins: list[PriceBin], strategy: Strategy) -> float:
    """
    Calculate probability of positive P&L at expiration.

    Sums the empirical probabilities of all bins where the strategy has
    positive P&L (after transaction costs).

    Args:
        bins: Histogram bins representing price distribution
        strategy: Option strategy to analyze

    Returns:
        Probability between 0.0 and 1.0

    Raises:
        ValueError: If bins is empty or all bins have zero count

    Examples:
        >>> bins = create_histogram(price_distribution, n_bins=50)
        >>> pop = calculate_probability_of_profit(bins, strategy)
        >>> print(f"Probability of Profit: {pop:.1%}")

    Note:
        - P&L > 0 is considered profitable (zero P&L is not profitable)
        - Transaction costs are included via strategy.total_payoff()
    """
    if not bins:
        raise ValueError("bins cannot be empty")

    total_count = sum(bin.count for bin in bins)

    if total_count == 0:
        raise ValueError("All bins have zero count")

    profitable_count = 0

    for bin in bins:
        # Check if P&L is positive at bin midpoint
        pnl = strategy.total_payoff(bin.midpoint)

        if pnl > 0:
            profitable_count += bin.count

    return profitable_count / total_count


def calculate_risk_metrics(bins: list[PriceBin], strategy: Strategy) -> RiskMetrics:
    """
    Calculate comprehensive risk metrics for a strategy.

    This is the main entry point for risk analysis. It validates the strategy
    before performing calculations and returns a complete RiskMetrics object.

    Args:
        bins: Histogram bins representing price distribution
        strategy: Option strategy to analyze

    Returns:
        RiskMetrics object with EV, PoP, and other risk measures

    Raises:
        ValueError: If bins is empty or invalid
        MixedExpirationError: If strategy has mixed expiration dates
        MissingBidAskError: If strategy lacks required price data

    Examples:
        >>> # Get price distribution
        >>> prices = get_price_distribution(
        ...     current_price=100.0,
        ...     returns=historical_returns,
        ...     target_date=expiration_date,
        ...     bootstrap_samples=10000
        ... )
        >>> bins = create_histogram(prices, n_bins=50)
        >>>
        >>> # Calculate risk metrics
        >>> metrics = calculate_risk_metrics(bins, strategy)
        >>> print(f"Expected Value: ${metrics.expected_value:.2f}")
        >>> print(f"Probability of Profit: {metrics.probability_of_profit:.1%}")

    Note:
        Calls strategy.validate_for_analysis() first to fail fast with clear
        error messages before expensive calculations.
    """
    # Validate strategy before expensive calculations
    strategy.validate_for_analysis()

    # Validate bins
    if not bins:
        raise ValueError("bins cannot be empty")

    # Calculate core metrics
    expected_value = calculate_expected_value(bins, strategy)
    probability_of_profit = calculate_probability_of_profit(bins, strategy)

    # Calculate max gain and max loss across the histogram range
    payoffs = [strategy.total_payoff(bin.midpoint) for bin in bins]
    max_gain = max(payoffs) if payoffs else None
    max_loss = min(payoffs) if payoffs else None

    # TODO: Implement breakeven calculation (future enhancement)
    # Requires finding prices where strategy.total_payoff(price) == 0
    breakevens: list[float] = []

    return RiskMetrics(
        expected_value=expected_value,
        probability_of_profit=probability_of_profit,
        max_gain=max_gain,
        max_loss=max_loss,
        breakevens=breakevens,
    )
