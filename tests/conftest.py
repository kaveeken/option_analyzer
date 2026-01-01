"""
Pytest configuration and shared fixtures.
"""

from datetime import date, timedelta

import pytest

from option_analyzer.models.domain import (
    OptionContract,
    Stock,
)


@pytest.fixture
def sample_stock() -> Stock:
    """Create a sample stock for testing."""
    return Stock(
        symbol="AAPL",
        current_price=100.0,
        conid=265598,
        available_expirations=[
            "DEC24",
            "JAN25",
            "FEB25",
        ],
    )


@pytest.fixture
def sample_call() -> OptionContract:
    """Create a sample call option for testing."""
    return OptionContract(
        conid=12345,
        strike=100.0,
        right="C",
        expiration=date(2024, 12, 20),
        bid=5.0,
        ask=5.5,
        multiplier=100,
    )


@pytest.fixture
def sample_put() -> OptionContract:
    """Create a sample put option for testing."""
    return OptionContract(
        conid=12346,
        strike=100.0,
        right="P",
        expiration=date(2024, 12, 20),
        bid=3.0,
        ask=3.5,
        multiplier=100,
    )


@pytest.fixture
def future_date() -> date:
    """Return a date 30 days in the future."""
    return date.today() + timedelta(days=30)


@pytest.fixture
def sample_returns():
    """Sample historical return multipliers for testing."""
    import numpy as np

    return np.array([1.02, 0.98, 1.01, 0.99, 1.03, 1.00, 0.97, 1.04, 1.01, 0.99])


@pytest.fixture
def sample_closes():
    """Sample closing prices for testing geometric returns."""
    import numpy as np

    return np.array([100.0, 102.0, 99.96, 100.96, 99.95, 102.95, 102.95, 99.86, 103.86, 104.90, 103.85])


@pytest.fixture
def fixed_rng():
    """Fixed RNG for reproducible tests."""
    import numpy as np

    return np.random.default_rng(42)
