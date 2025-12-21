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
        conid="265598",
        available_expirations=[
            date(2024, 12, 20),
            date(2025, 1, 17),
            date(2025, 2, 21),
        ],
    )


@pytest.fixture
def sample_call() -> OptionContract:
    """Create a sample call option for testing."""
    return OptionContract(
        conid="12345",
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
        conid="12346",
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
