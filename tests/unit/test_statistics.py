"""
Comprehensive unit tests for the Statistics Module.

Tests cover:
- Geometric returns calculation
- Period calculation for different bar sizes
- Bootstrap random walk
- Monte Carlo simulation
- Price distribution generation
- Price range calculation
- Histogram creation
- PriceBin model
"""

from datetime import date, timedelta

import numpy as np
import pytest

from option_analyzer.services.statistics import (
    PriceBin,
    bootstrap_walk,
    calculate_period_for_days,
    calculate_price_range,
    create_histogram,
    geometric_returns,
    get_price_distribution,
    monte_carlo_simulation,
)


class TestPriceBin:
    """Tests for PriceBin model."""

    def test_midpoint_calculation(self):
        """Test midpoint property returns correct average."""
        bin = PriceBin(lower=100.0, upper=110.0, count=5)
        assert bin.midpoint == 105.0

    def test_width_calculation(self):
        """Test width property returns correct difference."""
        bin = PriceBin(lower=100.0, upper=110.0, count=5)
        assert bin.width == 10.0

    def test_validation_lower_bound(self):
        """Test lower bound must be >= 0."""
        with pytest.raises(ValueError):
            PriceBin(lower=-1.0, upper=10.0, count=5)

    def test_validation_upper_bound(self):
        """Test upper bound must be > 0."""
        with pytest.raises(ValueError):
            PriceBin(lower=0.0, upper=0.0, count=5)

    def test_validation_count(self):
        """Test count must be >= 0."""
        with pytest.raises(ValueError):
            PriceBin(lower=100.0, upper=110.0, count=-1)


class TestGeometricReturns:
    """Tests for geometric_returns function."""

    def test_basic_calculation(self, sample_closes):
        """Test basic geometric returns calculation."""
        returns = geometric_returns(sample_closes)

        # Verify length
        assert len(returns) == len(sample_closes) - 1

        # Verify first return: 102.0 / 100.0 = 1.02
        assert returns[0] == pytest.approx(1.02, rel=1e-6)

        # Verify all returns are positive
        assert np.all(returns > 0)

    def test_multi_period_returns(self, sample_closes):
        """Test multi-period returns calculation."""
        # 2-period returns
        returns = geometric_returns(sample_closes, period=2)

        # Length should be len(closes) - period
        assert len(returns) == len(sample_closes) - 2

        # First return: closes[2] / closes[0] = 99.96 / 100.0
        assert returns[0] == pytest.approx(0.9996, rel=1e-4)

    def test_empty_array_error(self):
        """Test error handling for empty array."""
        with pytest.raises(ValueError, match="cannot be empty"):
            geometric_returns(np.array([]))

    def test_invalid_period_error(self):
        """Test error handling for invalid period."""
        closes = np.array([100.0, 105.0, 103.0])

        # Period < 1
        with pytest.raises(ValueError, match="must be >= 1"):
            geometric_returns(closes, period=0)

        # Period too large for data
        with pytest.raises(ValueError, match="Insufficient data"):
            geometric_returns(closes, period=5)

    def test_non_positive_prices_error(self):
        """Test error handling for non-positive prices."""
        closes = np.array([100.0, -5.0, 103.0])

        with pytest.raises(ValueError, match="must be positive"):
            geometric_returns(closes)

    def test_zero_price_error(self):
        """Test error handling for zero price."""
        closes = np.array([100.0, 0.0, 103.0])

        with pytest.raises(ValueError, match="must be positive"):
            geometric_returns(closes)


class TestCalculatePeriodForDays:
    """Tests for calculate_period_for_days function."""

    def test_daily_bars(self):
        """Test period calculation for daily bars."""
        # 365 calendar days -> 365 daily bars
        period = calculate_period_for_days(365, bar_size="1d")
        assert period == 365

        # 252 calendar days -> 252 daily bars
        period = calculate_period_for_days(252, bar_size="1d")
        assert period == 252

    def test_weekly_bars(self):
        """Test period calculation for weekly bars."""
        # 30 calendar days -> ~21 trading days -> ~4 weeks
        period = calculate_period_for_days(30, bar_size="1w")
        assert period == 4

    def test_monthly_bars(self):
        """Test period calculation for monthly bars."""
        # 365 calendar days -> ~12 months
        period = calculate_period_for_days(365, bar_size="1M")
        assert period == 12

    def test_negative_days_error(self):
        """Test error handling for negative days."""
        with pytest.raises(ValueError, match="must be non-negative"):
            calculate_period_for_days(-10, bar_size="1d")

    def test_unsupported_bar_size_error(self):
        """Test error handling for unsupported bar size."""
        with pytest.raises(ValueError, match="Unsupported bar_size"):
            calculate_period_for_days(30, bar_size="1h")

    def test_minimum_period_is_one(self):
        """Test that minimum period returned is 1."""
        # Very small target should still return at least 1
        period = calculate_period_for_days(1, bar_size="1w")
        assert period >= 1


class TestBootstrapWalk:
    """Tests for bootstrap_walk function."""

    def test_deterministic_with_seed(self, sample_returns, fixed_rng):
        """Test bootstrap walk is deterministic with fixed seed."""
        result1 = bootstrap_walk(sample_returns, steps=5, rng=fixed_rng)

        # Reset RNG
        fixed_rng2 = np.random.default_rng(42)
        result2 = bootstrap_walk(sample_returns, steps=5, rng=fixed_rng2)

        assert result1 == pytest.approx(result2, rel=1e-9)

    def test_positive_multiplier(self, sample_returns, fixed_rng):
        """Test that bootstrap walk returns positive multiplier."""
        result = bootstrap_walk(sample_returns, steps=10, rng=fixed_rng)
        assert result > 0

    def test_default_rng(self, sample_returns):
        """Test bootstrap walk works with default RNG."""
        result = bootstrap_walk(sample_returns, steps=5)
        assert result > 0

    def test_empty_returns_error(self):
        """Test error handling for empty returns array."""
        with pytest.raises(ValueError, match="cannot be empty"):
            bootstrap_walk(np.array([]), steps=5)

    def test_invalid_steps_error(self, sample_returns):
        """Test error handling for invalid steps."""
        with pytest.raises(ValueError, match="must be >= 1"):
            bootstrap_walk(sample_returns, steps=0)


class TestMonteCarloSimulation:
    """Tests for monte_carlo_simulation function."""

    def test_output_shape(self, sample_returns, fixed_rng):
        """Test output has correct shape."""
        n_sims = 100
        result = monte_carlo_simulation(sample_returns, steps=5, n_simulations=n_sims, rng=fixed_rng)

        assert result.shape == (n_sims,)

    def test_all_positive_multipliers(self, sample_returns, fixed_rng):
        """Test all multipliers are positive."""
        result = monte_carlo_simulation(sample_returns, steps=10, n_simulations=100, rng=fixed_rng)

        assert np.all(result > 0)

    def test_deterministic_with_seed(self, sample_returns):
        """Test simulation is deterministic with fixed seed."""
        rng1 = np.random.default_rng(42)
        result1 = monte_carlo_simulation(sample_returns, steps=5, n_simulations=50, rng=rng1)

        rng2 = np.random.default_rng(42)
        result2 = monte_carlo_simulation(sample_returns, steps=5, n_simulations=50, rng=rng2)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_empty_returns_error(self):
        """Test error handling for empty returns array."""
        with pytest.raises(ValueError, match="cannot be empty"):
            monte_carlo_simulation(np.array([]), steps=5, n_simulations=100)

    def test_invalid_steps_error(self, sample_returns):
        """Test error handling for invalid steps."""
        with pytest.raises(ValueError, match="must be >= 1"):
            monte_carlo_simulation(sample_returns, steps=0, n_simulations=100)

    def test_invalid_n_simulations_error(self, sample_returns):
        """Test error handling for invalid n_simulations."""
        with pytest.raises(ValueError, match="must be >= 1"):
            monte_carlo_simulation(sample_returns, steps=5, n_simulations=0)

    def test_default_rng(self, sample_returns):
        """Test simulation works with default RNG."""
        result = monte_carlo_simulation(sample_returns, steps=5, n_simulations=10)
        assert len(result) == 10
        assert np.all(result > 0)


class TestGetPriceDistribution:
    """Tests for get_price_distribution function."""

    def test_direct_mode(self, sample_returns):
        """Test direct mode applies historical sequence."""
        current_price = 100.0
        target_date = date.today() + timedelta(days=30)

        # Direct mode (no bootstrap)
        prices = get_price_distribution(
            current_price=current_price,
            returns=sample_returns,
            target_date=target_date,
            bar_size="1w",
            bootstrap_samples=None,
        )

        # Should return single price
        assert len(prices) == 1
        assert prices[0] > 0

    def test_monte_carlo_mode(self, sample_returns, fixed_rng):
        """Test Monte Carlo mode generates multiple samples."""
        current_price = 100.0
        target_date = date.today() + timedelta(days=30)
        n_samples = 100

        prices = get_price_distribution(
            current_price=current_price,
            returns=sample_returns,
            target_date=target_date,
            bar_size="1w",
            bootstrap_samples=n_samples,
            rng=fixed_rng,
        )

        # Should return n_samples prices
        assert len(prices) == n_samples
        assert np.all(prices > 0)

    def test_non_positive_current_price_error(self, sample_returns):
        """Test error handling for non-positive current price."""
        target_date = date.today() + timedelta(days=30)

        with pytest.raises(ValueError, match="must be positive"):
            get_price_distribution(
                current_price=0.0,
                returns=sample_returns,
                target_date=target_date,
            )

    def test_empty_returns_error(self):
        """Test error handling for empty returns array."""
        target_date = date.today() + timedelta(days=30)

        with pytest.raises(ValueError, match="cannot be empty"):
            get_price_distribution(
                current_price=100.0,
                returns=np.array([]),
                target_date=target_date,
            )

    def test_past_target_date_error(self, sample_returns):
        """Test error handling for target date in the past."""
        target_date = date.today() - timedelta(days=30)

        with pytest.raises(ValueError, match="must be after"):
            get_price_distribution(
                current_price=100.0,
                returns=sample_returns,
                target_date=target_date,
            )

    def test_custom_reference_date(self, sample_returns):
        """Test using custom reference date."""
        current_price = 100.0
        ref_date = date(2024, 1, 1)
        target_date = date(2024, 2, 1)

        prices = get_price_distribution(
            current_price=current_price,
            returns=sample_returns,
            target_date=target_date,
            reference_date=ref_date,
            bootstrap_samples=None,
        )

        assert len(prices) == 1
        assert prices[0] > 0

    def test_insufficient_data_direct_mode_error(self, sample_returns):
        """Test error handling for insufficient data in direct mode."""
        current_price = 100.0
        # Target date far in future requiring more steps than available returns
        target_date = date.today() + timedelta(days=365 * 5)  # 5 years

        with pytest.raises(ValueError, match="Insufficient data for direct mode"):
            get_price_distribution(
                current_price=current_price,
                returns=sample_returns,
                target_date=target_date,
                bar_size="1w",
                bootstrap_samples=None,  # Direct mode
            )


class TestCalculatePriceRange:
    """Tests for calculate_price_range function."""

    def test_auto_range_with_padding(self):
        """Test automatic range calculation with padding."""
        prices = np.array([95.0, 100.0, 105.0, 110.0])

        min_price, max_price = calculate_price_range(prices, padding=0.05)

        # Min: 95.0 * 0.95 = 90.25
        # Max: 110.0 * 1.05 = 115.5
        assert min_price == pytest.approx(90.25, rel=1e-6)
        assert max_price == pytest.approx(115.5, rel=1e-6)

    def test_user_min_override(self):
        """Test user-specified minimum override."""
        prices = np.array([95.0, 100.0, 105.0, 110.0])

        min_price, max_price = calculate_price_range(prices, user_min=90.0)

        assert min_price == 90.0
        assert max_price == pytest.approx(115.5, rel=1e-6)

    def test_user_max_override(self):
        """Test user-specified maximum override."""
        prices = np.array([95.0, 100.0, 105.0, 110.0])

        min_price, max_price = calculate_price_range(prices, user_max=120.0)

        assert min_price == pytest.approx(90.25, rel=1e-6)
        assert max_price == 120.0

    def test_no_padding(self):
        """Test range with no padding."""
        prices = np.array([95.0, 100.0, 105.0, 110.0])

        min_price, max_price = calculate_price_range(prices, padding=0.0)

        assert min_price == 95.0
        assert max_price == 110.0

    def test_empty_distribution_error(self):
        """Test error handling for empty distribution."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_price_range(np.array([]))

    def test_negative_padding_error(self):
        """Test error handling for negative padding."""
        prices = np.array([95.0, 100.0, 105.0])

        with pytest.raises(ValueError, match="must be non-negative"):
            calculate_price_range(prices, padding=-0.1)


class TestCreateHistogram:
    """Tests for create_histogram function."""

    def test_basic_binning(self):
        """Test basic histogram creation."""
        values = np.array([95.0, 100.0, 105.0, 110.0, 115.0])
        bins = create_histogram(values, n_bins=4, min_value=90.0, max_value=120.0)

        # Should have 4 bins
        assert len(bins) == 4

        # Each bin should be PriceBin
        assert all(isinstance(b, PriceBin) for b in bins)

        # Total count should match input
        assert sum(b.count for b in bins) == len(values)

    def test_bin_width_uniform(self):
        """Test that bins have uniform width."""
        values = np.array([100.0, 105.0, 110.0])
        bins = create_histogram(values, n_bins=5, min_value=100.0, max_value=150.0)

        # All bins should have width of 10.0
        for bin in bins:
            assert bin.width == pytest.approx(10.0, rel=1e-6)

    def test_max_value_included_in_last_bin(self):
        """Test that max value is included in the last bin."""
        # Create values with one at exact max
        values = np.array([90.0, 95.0, 100.0, 105.0, 110.0, 120.0])
        bins = create_histogram(values, n_bins=4, min_value=90.0, max_value=120.0)

        # Last bin should include the 120.0 value
        assert bins[-1].count >= 1

        # Total count should match
        assert sum(b.count for b in bins) == len(values)

    def test_auto_min_max(self):
        """Test automatic min/max from data."""
        values = np.array([95.0, 100.0, 105.0, 110.0])
        bins = create_histogram(values, n_bins=4)

        # First bin should start at min value
        assert bins[0].lower == 95.0

        # Last bin should end at max value
        assert bins[-1].upper == 110.0

    def test_empty_values_error(self):
        """Test error handling for empty values array."""
        with pytest.raises(ValueError, match="cannot be empty"):
            create_histogram(np.array([]), n_bins=4)

    def test_invalid_n_bins_error(self):
        """Test error handling for invalid n_bins."""
        values = np.array([100.0, 105.0, 110.0])

        with pytest.raises(ValueError, match="must be >= 1"):
            create_histogram(values, n_bins=0)

    def test_min_max_validation_error(self):
        """Test error handling when min >= max."""
        values = np.array([100.0, 105.0, 110.0])

        with pytest.raises(ValueError, match="must be less than"):
            create_histogram(values, n_bins=4, min_value=120.0, max_value=100.0)

    def test_single_bin(self):
        """Test histogram with single bin."""
        values = np.array([100.0, 105.0, 110.0])
        bins = create_histogram(values, n_bins=1, min_value=100.0, max_value=110.0)

        assert len(bins) == 1
        assert bins[0].count == 3
