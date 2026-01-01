"""
Unit tests for API endpoints.

Tests cover:
- Strategy initialization endpoint (success, no expirations, session creation)
- Stock endpoint (get stock data)
- Option chain endpoint (get option chain)
- Health check endpoint (response format)
"""

from datetime import date
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from option_analyzer.api.app import create_app
from option_analyzer.api.dependencies import get_ibkr_client, get_session_service_dep
from option_analyzer.clients.ibkr import IBKRClient
from option_analyzer.models.domain import OptionChain, OptionContract, Stock
from option_analyzer.services.session import SessionService
from option_analyzer.utils.exceptions import ValidationError


@pytest.fixture
def mock_ibkr_client():
    """Create a mock IBKR client."""
    return Mock(spec=IBKRClient)


@pytest.fixture
def session_service():
    """Create a real session service for testing."""
    return SessionService(ttl_seconds=3600)


@pytest.fixture
def test_client(mock_ibkr_client, session_service):
    """Create a test client with mocked dependencies."""
    app = create_app()

    # Override dependencies
    app.dependency_overrides[get_ibkr_client] = lambda: mock_ibkr_client
    app.dependency_overrides[get_session_service_dep] = lambda: session_service

    return TestClient(app)


class TestStrategyInitEndpoint:
    """Test /api/strategy/init endpoint."""

    def test_strategy_init_success(self, test_client, mock_ibkr_client):
        """Test successful strategy initialization."""
        # Mock IBKR response
        mock_stock = Stock(
            symbol="AAPL",
            current_price=150.25,
            conid=265598,
            available_expirations=["JAN26", "FEB26", "MAR26"],
        )
        mock_ibkr_client.get_stock = AsyncMock(return_value=mock_stock)

        # Make request
        response = test_client.post("/api/strategy/init", json={"symbol": "AAPL"})

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["current_price"] == 150.25
        assert data["target_date"] == "JAN26"  # Earliest expiration
        assert data["available_expirations"] == ["JAN26", "FEB26", "MAR26"]
        assert "session_id" in data
        assert len(data["session_id"]) > 0

        # Verify session cookie was set
        assert "session_id" in response.cookies

    def test_strategy_init_no_expirations(self, test_client, mock_ibkr_client):
        """Test strategy init raises ValidationError when no expirations available."""
        # Mock IBKR response with no expirations
        mock_stock = Stock(
            symbol="TEST",
            current_price=10.0,
            conid=12345,
            available_expirations=[],
        )
        mock_ibkr_client.get_stock = AsyncMock(return_value=mock_stock)

        # Make request
        response = test_client.post("/api/strategy/init", json={"symbol": "TEST"})

        # Verify error response
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "NO_EXPIRATIONS_AVAILABLE" in data["code"]

    def test_strategy_init_creates_session(self, test_client, mock_ibkr_client, session_service):
        """Test that strategy init creates a session in the service."""
        # Mock IBKR response
        mock_stock = Stock(
            symbol="AAPL",
            current_price=150.25,
            conid=265598,
            available_expirations=["JAN26"],
        )
        mock_ibkr_client.get_stock = AsyncMock(return_value=mock_stock)

        # Verify no sessions initially
        assert session_service.session_count() == 0

        # Make request
        response = test_client.post("/api/strategy/init", json={"symbol": "AAPL"})

        # Verify session was created
        assert response.status_code == 200
        assert session_service.session_count() == 1

        # Verify session contains strategy data
        session_id = response.json()["session_id"]
        session = session_service.get_session(session_id)
        assert "strategy" in session.data
        assert session.data["strategy"]["symbol"] == "AAPL"
        assert session.data["strategy"]["target_date"] == "JAN26"
        assert session.data["strategy"]["positions"] == []


class TestGetStockEndpoint:
    """Test /api/stocks/{symbol} endpoint."""

    def test_get_stock(self, test_client, mock_ibkr_client):
        """Test get stock endpoint returns StockResponse format."""
        # Mock IBKR response
        mock_stock = Stock(
            symbol="AAPL",
            current_price=150.25,
            conid=265598,
            available_expirations=["JAN26", "FEB26"],
        )
        mock_ibkr_client.get_stock = AsyncMock(return_value=mock_stock)

        # Make request
        response = test_client.get("/api/stocks/AAPL")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["current_price"] == 150.25
        assert data["conid"] == 265598
        assert data["available_expirations"] == ["JAN26", "FEB26"]

        # Verify IBKR client was called with uppercase symbol
        mock_ibkr_client.get_stock.assert_called_once_with("AAPL")

    def test_get_stock_lowercase_symbol(self, test_client, mock_ibkr_client):
        """Test get stock endpoint converts lowercase symbols to uppercase."""
        # Mock IBKR response
        mock_stock = Stock(
            symbol="AAPL",
            current_price=150.25,
            conid=265598,
            available_expirations=["JAN26"],
        )
        mock_ibkr_client.get_stock = AsyncMock(return_value=mock_stock)

        # Make request with lowercase symbol
        response = test_client.get("/api/stocks/aapl")

        # Verify IBKR client was called with uppercase
        assert response.status_code == 200
        mock_ibkr_client.get_stock.assert_called_once_with("AAPL")


class TestGetOptionChainEndpoint:
    """Test /api/stocks/{symbol}/chains endpoint."""

    def test_get_option_chain(self, test_client, mock_ibkr_client):
        """Test get option chain endpoint returns OptionChainResponse format."""
        # Mock stock response
        mock_stock = Stock(
            symbol="AAPL",
            current_price=150.25,
            conid=265598,
            available_expirations=["JAN26"],
        )
        mock_ibkr_client.get_stock = AsyncMock(return_value=mock_stock)

        # Mock option chain response
        mock_call = OptionContract(
            conid=123456,
            strike=150.0,
            right="C",
            expiration=date(2026, 1, 16),
            bid=2.50,
            ask=2.55,
            multiplier=100,
        )
        mock_put = OptionContract(
            conid=123457,
            strike=150.0,
            right="P",
            expiration=date(2026, 1, 16),
            bid=1.80,
            ask=1.85,
            multiplier=100,
        )
        mock_chain = OptionChain(
            expiration=date(2026, 1, 16),
            calls=[mock_call],
            puts=[mock_put],
        )
        mock_ibkr_client.get_option_chain = AsyncMock(return_value=mock_chain)

        # Make request
        response = test_client.get("/api/stocks/AAPL/chains?month=JAN26")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["expiration"] == "2026-01-16"
        assert len(data["calls"]) == 1
        assert len(data["puts"]) == 1

        # Verify call data
        call = data["calls"][0]
        assert call["conid"] == 123456
        assert call["strike"] == 150.0
        assert call["right"] == "C"
        assert call["bid"] == 2.50
        assert call["ask"] == 2.55
        assert call["multiplier"] == 100

        # Verify put data
        put = data["puts"][0]
        assert put["conid"] == 123457
        assert put["strike"] == 150.0
        assert put["right"] == "P"
        assert put["bid"] == 1.80
        assert put["ask"] == 1.85

        # Verify IBKR client calls
        mock_ibkr_client.get_stock.assert_called_once_with("AAPL")
        mock_ibkr_client.get_option_chain.assert_called_once_with(265598, "JAN26")

    def test_get_option_chain_uppercase_month(self, test_client, mock_ibkr_client):
        """Test option chain endpoint converts month to uppercase."""
        # Mock responses
        mock_stock = Stock(
            symbol="AAPL",
            current_price=150.25,
            conid=265598,
            available_expirations=["JAN26"],
        )
        mock_chain = OptionChain(
            expiration=date(2026, 1, 16),
            calls=[],
            puts=[],
        )
        mock_ibkr_client.get_stock = AsyncMock(return_value=mock_stock)
        mock_ibkr_client.get_option_chain = AsyncMock(return_value=mock_chain)

        # Make request with lowercase month
        response = test_client.get("/api/stocks/AAPL/chains?month=jan26")

        # Verify month was converted to uppercase
        assert response.status_code == 200
        mock_ibkr_client.get_option_chain.assert_called_once_with(265598, "JAN26")


class TestHealthCheckEndpoint:
    """Test /health endpoint."""

    def test_health_check(self, test_client):
        """Test health check endpoint response format."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
