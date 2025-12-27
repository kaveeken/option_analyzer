"""
Comprehensive unit tests for risk metrics module.

Tests cover all functions and edge cases to achieve >90% coverage.
Issue: option_returns-ukg
"""

from datetime import date

import pytest

from option_analyzer.models.domain import (
    OptionContract,
    OptionPosition,
    Stock,
    Strategy,
)
from option_analyzer.services.risk import (
    RiskMetrics,
    calculate_expected_value,
    calculate_probability_of_profit,
    calculate_risk_metrics,
)
from option_analyzer.services.statistics import PriceBin


# Test Fixtures
@pytest.fixture
def simple_long_call():
    """
    Long call at strike 105, premium paid 2.25 (midpoint of 2.0/2.5).
    Breakeven: 105 + 2.25 = 107.25
    Max loss: -225 (premium paid for 1 contract)
    Max gain: Unlimited
    """
    stock = Stock(symbol="AAPL", current_price=100.0, conid="123")
    contract = OptionContract(
        conid="456",
        strike=105.0,
        right="C",
        expiration=date(2024, 12, 31),
        bid=2.0,
        ask=2.5,
    )
    position = OptionPosition(contract=contract, quantity=1)
    return Strategy(stock=stock, option_positions=[position])


@pytest.fixture
def covered_call():
    """
    Covered call: Long 100 shares at 100, short call at strike 105 with 2.25 premium.
    Max gain: (105-100)*100 + 225 = 725
    Max loss: Unlimited downside (stock can go to zero)
    Breakeven: 100 - 2.25 = 97.75
    """
    stock = Stock(symbol="AAPL", current_price=100.0, conid="123")
    contract = OptionContract(
        conid="456",
        strike=105.0,
        right="C",
        expiration=date(2024, 12, 31),
        bid=2.0,
        ask=2.5,
    )
    # Short call
    position = OptionPosition(contract=contract, quantity=-1)
    return Strategy(stock=stock, option_positions=[position], stock_quantity=100)


@pytest.fixture
def short_put():
    """
    Short put at strike 95 with 2.25 premium collected.
    Max gain: 225 (premium collected)
    Max loss: (95 - 0) * 100 - 225 = 9275 (if stock goes to zero)
    Breakeven: 95 - 2.25 = 92.75
    """
    stock = Stock(symbol="AAPL", current_price=100.0, conid="123")
    contract = OptionContract(
        conid="456",
        strike=95.0,
        right="P",
        expiration=date(2024, 12, 31),
        bid=2.0,
        ask=2.5,
    )
    position = OptionPosition(contract=contract, quantity=-1)
    return Strategy(stock=stock, option_positions=[position])


class TestRiskMetrics:
    """Tests for RiskMetrics model."""

    def test_risk_metrics_creation(self):
        """Test basic RiskMetrics model creation."""
        metrics = RiskMetrics(
            expected_value=100.0,
            probability_of_profit=0.65,
            max_gain=500.0,
            max_loss=-200.0,
            breakevens=[105.0, 110.0],
        )

        assert metrics.expected_value == 100.0
        assert metrics.probability_of_profit == 0.65
        assert metrics.max_gain == 500.0
        assert metrics.max_loss == -200.0
        assert metrics.breakevens == [105.0, 110.0]

    def test_probability_validation(self):
        """Test that probability_of_profit is validated to be between 0 and 1."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RiskMetrics(expected_value=0.0, probability_of_profit=1.5)

        with pytest.raises(ValidationError):
            RiskMetrics(expected_value=0.0, probability_of_profit=-0.1)


class TestCalculateExpectedValue:
    """Tests for calculate_expected_value function."""

    def test_simple_ev_calculation(self, simple_long_call):
        """Test EV calculation with realistic bins and verify weighted probability."""
        # Long call: strike 105, premium paid = ask * 100 = 2.5 * 100 = 250
        # Transaction cost = 0.65
        # At 97.5: intrinsic=0, payoff = -250 + 0 - 0.65 = -250.65
        # At 102.5: intrinsic=0, payoff = -250 + 0 - 0.65 = -250.65
        # At 107.5: intrinsic=250, payoff = -250 + 250 - 0.65 = -0.65
        # At 112.5: intrinsic=750, payoff = -250 + 750 - 0.65 = 499.35
        bins = [
            PriceBin(lower=95.0, upper=100.0, count=25),   # midpoint 97.5
            PriceBin(lower=100.0, upper=105.0, count=25),  # midpoint 102.5
            PriceBin(lower=105.0, upper=110.0, count=25),  # midpoint 107.5
            PriceBin(lower=110.0, upper=115.0, count=25),  # midpoint 112.5
        ]

        ev = calculate_expected_value(bins, simple_long_call)

        # Expected: 0.25*(-250.65) + 0.25*(-250.65) + 0.25*(-0.65) + 0.25*(499.35)
        # = -62.6625 - 62.6625 - 0.1625 + 124.8375 = -0.65
        assert isinstance(ev, float)
        assert ev == pytest.approx(-0.65, abs=0.01)

    def test_weighted_probability_validation(self, short_put):
        """Test that probabilities are correctly weighted by bin counts."""
        # Short put: strike 95, premium received = bid * 100 = 2.0 * 100 = 200
        # Transaction cost = 0.65
        # At 90: intrinsic=-500 (short put loses when price drops), payoff = 200 - 500 - 0.65 = -300.65
        # At 100: intrinsic=0 (put expires worthless), payoff = 200 + 0 - 0.65 = 199.35
        bins = [
            PriceBin(lower=85.0, upper=95.0, count=30),    # midpoint 90
            PriceBin(lower=95.0, upper=105.0, count=70),   # midpoint 100
        ]

        ev = calculate_expected_value(bins, short_put)

        # Expected: 0.30*(-300.65) + 0.70*(199.35) = -90.195 + 139.545 = 49.35
        assert ev == pytest.approx(49.35, abs=0.01)

    def test_empty_bins_error(self, simple_long_call):
        """Test that empty bins raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_expected_value([], simple_long_call)

    def test_zero_count_error(self, simple_long_call):
        """Test that all-zero counts raise ValueError."""
        bins = [
            PriceBin(lower=95.0, upper=100.0, count=0),
            PriceBin(lower=100.0, upper=105.0, count=0),
        ]

        with pytest.raises(ValueError, match="zero count"):
            calculate_expected_value(bins, simple_long_call)


class TestCalculateProbabilityOfProfit:
    """Tests for calculate_probability_of_profit function."""

    def test_all_profitable_scenarios(self, short_put):
        """Test PoP when all bins are profitable (should be 1.0)."""
        # Short put: strike 95, premium 2.25 collected
        # At prices > 95, all bins are profitable (put expires worthless, keep premium)
        bins = [
            PriceBin(lower=95.0, upper=100.0, count=40),   # midpoint 97.5: +225
            PriceBin(lower=100.0, upper=105.0, count=60),  # midpoint 102.5: +225
        ]

        pop = calculate_probability_of_profit(bins, short_put)

        # All bins are profitable, so PoP = 1.0
        assert pop == pytest.approx(1.0)

    def test_no_profitable_scenarios(self, simple_long_call):
        """Test PoP when no bins are profitable (should be 0.0)."""
        # Long call: strike 105, premium 2.25
        # Breakeven is at 107.25, so below that all bins are losses
        bins = [
            PriceBin(lower=95.0, upper=100.0, count=50),   # midpoint 97.5: -225
            PriceBin(lower=100.0, upper=105.0, count=50),  # midpoint 102.5: -225
        ]

        pop = calculate_probability_of_profit(bins, simple_long_call)

        # No bins are profitable, so PoP = 0.0
        assert pop == pytest.approx(0.0)

    def test_partial_profitable_scenarios(self, simple_long_call):
        """Test PoP with mix of profitable and unprofitable bins."""
        # Long call: strike 105, premium paid = 250, transaction cost = 0.65
        # Breakeven needs price where intrinsic > 250.65
        # That's at price > 105 + 2.5065 = 107.5065
        bins = [
            PriceBin(lower=95.0, upper=100.0, count=20),   # midpoint 97.5: -250.65 (loss)
            PriceBin(lower=100.0, upper=105.0, count=30),  # midpoint 102.5: -250.65 (loss)
            PriceBin(lower=105.0, upper=110.0, count=25),  # midpoint 107.5: -0.65 (still loss!)
            PriceBin(lower=110.0, upper=115.0, count=25),  # midpoint 112.5: 499.35 (profit!)
        ]

        pop = calculate_probability_of_profit(bins, simple_long_call)

        # Only 25 out of 100 samples are profitable (last bin)
        assert pop == pytest.approx(0.25, abs=0.01)

    def test_breakeven_not_counted_as_profitable(self):
        """Test that breakeven (P&L = 0) is NOT counted as profitable."""
        # Create a strategy that breaks even at specific price
        stock = Stock(symbol="AAPL", current_price=100.0, conid="123")
        contract = OptionContract(
            conid="456",
            strike=100.0,
            right="C",
            expiration=date(2024, 12, 31),
            bid=0.0,  # Free option for simplicity
            ask=0.0,
        )
        position = OptionPosition(contract=contract, quantity=1)
        strategy = Strategy(stock=stock, option_positions=[position])

        # At price 100, payoff = max(0, 100-100)*100 - 0 = 0 (breakeven)
        # At price 95, payoff = max(0, 95-100)*100 - 0 = 0 (loss, ATM)
        # At price 105, payoff = max(0, 105-100)*100 - 0 = 500 (profit)
        bins = [
            PriceBin(lower=92.5, upper=97.5, count=30),    # midpoint 95: 0
            PriceBin(lower=97.5, upper=102.5, count=40),   # midpoint 100: 0
            PriceBin(lower=102.5, upper=107.5, count=30),  # midpoint 105: 500
        ]

        pop = calculate_probability_of_profit(bins, strategy)

        # Only the last bin (30/100) is profitable; breakeven doesn't count
        assert pop == pytest.approx(0.30, abs=0.01)

    def test_empty_bins_error(self, simple_long_call):
        """Test that empty bins raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_probability_of_profit([], simple_long_call)

    def test_zero_count_error(self, simple_long_call):
        """Test that all-zero counts raise ValueError."""
        bins = [
            PriceBin(lower=95.0, upper=100.0, count=0),
            PriceBin(lower=100.0, upper=105.0, count=0),
        ]

        with pytest.raises(ValueError, match="zero count"):
            calculate_probability_of_profit(bins, simple_long_call)


class TestCalculateRiskMetrics:
    """Tests for calculate_risk_metrics function."""

    def test_returns_complete_risk_metrics_object(self, simple_long_call):
        """Test that calculate_risk_metrics returns a complete RiskMetrics object."""
        bins = [
            PriceBin(lower=95.0, upper=100.0, count=20),   # -250.65
            PriceBin(lower=100.0, upper=105.0, count=30),  # -250.65
            PriceBin(lower=105.0, upper=110.0, count=25),  # -0.65
            PriceBin(lower=110.0, upper=115.0, count=25),  # +499.35
        ]

        metrics = calculate_risk_metrics(bins, simple_long_call)

        # Verify return type
        assert isinstance(metrics, RiskMetrics)

        # Verify all fields are populated correctly
        assert isinstance(metrics.expected_value, float)
        assert isinstance(metrics.probability_of_profit, float)
        assert 0.0 <= metrics.probability_of_profit <= 1.0

        # Verify PoP calculation (only last bin is profitable: 25 out of 100)
        assert metrics.probability_of_profit == pytest.approx(0.25, abs=0.01)

        # Max gain and loss should be calculated from bins
        assert metrics.max_gain is not None
        assert metrics.max_loss is not None
        assert metrics.max_gain > 0  # Should have some profitable scenarios
        assert metrics.max_loss < 0  # Should have some losing scenarios

        # Max gain should be ~499.35 (highest bin)
        assert metrics.max_gain == pytest.approx(499.35, abs=1.0)
        # Max loss should be -250.65 (premium paid + transaction cost)
        assert metrics.max_loss == pytest.approx(-250.65, abs=1.0)

        # Breakevens list exists (currently empty, but field is present)
        assert isinstance(metrics.breakevens, list)

    def test_validates_strategy_before_calculation(self):
        """Test that strategy validation is called before expensive calculations."""
        # Create strategy with mixed expirations (invalid)
        stock = Stock(symbol="AAPL", current_price=100.0, conid="123")
        contract1 = OptionContract(
            conid="456",
            strike=105.0,
            right="C",
            expiration=date(2024, 12, 31),
            bid=2.0,
            ask=2.5,
        )
        contract2 = OptionContract(
            conid="789",
            strike=110.0,
            right="C",
            expiration=date(2025, 1, 31),  # Different expiration
            bid=3.0,
            ask=3.5,
        )
        position1 = OptionPosition(contract=contract1, quantity=1)
        position2 = OptionPosition(contract=contract2, quantity=1)
        strategy = Strategy(stock=stock, option_positions=[position1, position2])

        bins = [PriceBin(lower=95.0, upper=100.0, count=20)]

        # Should raise MixedExpirationError from validation
        from option_analyzer.utils.exceptions import MixedExpirationError

        with pytest.raises(MixedExpirationError):
            calculate_risk_metrics(bins, strategy)

    def test_empty_bins_error(self, simple_long_call):
        """Test that empty bins raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_risk_metrics([], simple_long_call)

    def test_validates_missing_bid_ask(self):
        """Test that strategies without bid/ask raise MissingBidAskError."""
        from option_analyzer.utils.exceptions import MissingBidAskError

        stock = Stock(symbol="AAPL", current_price=100.0, conid="123")
        # Contract missing bid/ask
        contract = OptionContract(
            conid="456",
            strike=105.0,
            right="C",
            expiration=date(2024, 12, 31),
            bid=None,
            ask=None,
        )
        position = OptionPosition(contract=contract, quantity=1)
        strategy = Strategy(stock=stock, option_positions=[position])

        bins = [PriceBin(lower=95.0, upper=100.0, count=20)]

        with pytest.raises(MissingBidAskError):
            calculate_risk_metrics(bins, strategy)
