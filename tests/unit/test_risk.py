"""
Basic smoke tests for risk metrics module.

These are minimal tests to verify the module works. Comprehensive tests
with >90% coverage are tracked in issue option_returns-ukg.
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

    @pytest.fixture
    def simple_strategy(self):
        """Create a simple long call strategy."""
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

    def test_calculate_ev_simple(self, simple_strategy):
        """Test EV calculation with simple bins."""
        bins = [
            PriceBin(lower=95.0, upper=100.0, count=20),  # OTM - loss
            PriceBin(lower=100.0, upper=105.0, count=30),  # OTM - loss
            PriceBin(lower=105.0, upper=110.0, count=50),  # ITM - profit
        ]

        ev = calculate_expected_value(bins, simple_strategy)

        # EV should be a float (specific value depends on payoff calculation)
        assert isinstance(ev, float)

    def test_calculate_ev_empty_bins(self, simple_strategy):
        """Test that empty bins raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_expected_value([], simple_strategy)

    def test_calculate_ev_zero_counts(self, simple_strategy):
        """Test that all-zero counts raise ValueError."""
        bins = [
            PriceBin(lower=95.0, upper=100.0, count=0),
            PriceBin(lower=100.0, upper=105.0, count=0),
        ]

        with pytest.raises(ValueError, match="zero count"):
            calculate_expected_value(bins, simple_strategy)


class TestCalculateProbabilityOfProfit:
    """Tests for calculate_probability_of_profit function."""

    @pytest.fixture
    def simple_strategy(self):
        """Create a simple long call strategy."""
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

    def test_calculate_pop_simple(self, simple_strategy):
        """Test PoP calculation with simple bins."""
        bins = [
            PriceBin(lower=95.0, upper=100.0, count=20),  # OTM - loss
            PriceBin(lower=100.0, upper=105.0, count=30),  # OTM - loss
            PriceBin(lower=105.0, upper=110.0, count=50),  # ITM - profit
        ]

        pop = calculate_probability_of_profit(bins, simple_strategy)

        # PoP should be between 0 and 1
        assert 0.0 <= pop <= 1.0
        assert isinstance(pop, float)

    def test_calculate_pop_empty_bins(self, simple_strategy):
        """Test that empty bins raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_probability_of_profit([], simple_strategy)


class TestCalculateRiskMetrics:
    """Tests for calculate_risk_metrics function."""

    @pytest.fixture
    def simple_strategy(self):
        """Create a simple long call strategy."""
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

    def test_calculate_risk_metrics_simple(self, simple_strategy):
        """Test full risk metrics calculation."""
        bins = [
            PriceBin(lower=95.0, upper=100.0, count=20),
            PriceBin(lower=100.0, upper=105.0, count=30),
            PriceBin(lower=105.0, upper=110.0, count=50),
        ]

        metrics = calculate_risk_metrics(bins, simple_strategy)

        # Verify return type
        assert isinstance(metrics, RiskMetrics)

        # Verify all fields are populated
        assert isinstance(metrics.expected_value, float)
        assert isinstance(metrics.probability_of_profit, float)
        assert 0.0 <= metrics.probability_of_profit <= 1.0

        # Max gain and loss should be calculated
        assert metrics.max_gain is not None
        assert metrics.max_loss is not None

    def test_calculate_risk_metrics_validates_strategy(self):
        """Test that strategy validation is called."""
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
