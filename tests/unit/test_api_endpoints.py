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


class TestAddPositionEndpoint:
    """Test POST /api/strategy/positions endpoint."""

    def _create_session_with_strategy(self, test_client, mock_ibkr_client):
        """Helper to create a session with initialized strategy."""
        mock_stock = Stock(
            symbol="AAPL",
            current_price=150.25,
            conid=265598,
            available_expirations=["JAN26"],
        )
        mock_ibkr_client.get_stock = AsyncMock(return_value=mock_stock)

        response = test_client.post("/api/strategy/init", json={"symbol": "AAPL"})
        assert response.status_code == 200
        return response.cookies.get("session_id")

    def test_add_position_success(self, test_client, mock_ibkr_client):
        """Test successfully adding a position to the strategy."""
        # Create session with strategy
        session_id = self._create_session_with_strategy(test_client, mock_ibkr_client)

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
        mock_chain = OptionChain(
            expiration=date(2026, 1, 16),
            calls=[mock_call],
            puts=[],
        )
        mock_ibkr_client.get_option_chain = AsyncMock(return_value=mock_chain)

        # Add position
        response = test_client.post(
            "/api/strategy/positions",
            json={"conid": 123456, "quantity": 2},
            cookies={"session_id": session_id}
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert len(data["positions"]) == 1
        position = data["positions"][0]
        assert position["conid"] == 123456
        assert position["strike"] == 150.0
        assert position["right"] == "C"
        assert position["expiration"] == "2026-01-16"
        assert position["quantity"] == 2
        assert position["bid"] == 2.50
        assert position["ask"] == 2.55

    def test_add_position_negative_quantity(self, test_client, mock_ibkr_client):
        """Test adding a short position (negative quantity)."""
        session_id = self._create_session_with_strategy(test_client, mock_ibkr_client)

        mock_put = OptionContract(
            conid=123457,
            strike=145.0,
            right="P",
            expiration=date(2026, 1, 16),
            bid=1.80,
            ask=1.85,
            multiplier=100,
        )
        mock_chain = OptionChain(
            expiration=date(2026, 1, 16),
            calls=[],
            puts=[mock_put],
        )
        mock_ibkr_client.get_option_chain = AsyncMock(return_value=mock_chain)

        response = test_client.post(
            "/api/strategy/positions",
            json={"conid": 123457, "quantity": -1},
            cookies={"session_id": session_id}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["positions"][0]["quantity"] == -1

    def test_add_position_zero_quantity(self, test_client, mock_ibkr_client):
        """Test that adding a position with zero quantity returns 400."""
        session_id = self._create_session_with_strategy(test_client, mock_ibkr_client)

        response = test_client.post(
            "/api/strategy/positions",
            json={"conid": 123456, "quantity": 0},
            cookies={"session_id": session_id}
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "INVALID_QUANTITY" in data["code"]

    def test_add_position_no_session(self, test_client):
        """Test that adding a position without a session returns 401."""
        response = test_client.post(
            "/api/strategy/positions",
            json={"conid": 123456, "quantity": 2}
        )

        assert response.status_code == 401
        data = response.json()
        assert "error" in data

    def test_add_position_no_strategy_initialized(self, test_client, session_service):
        """Test that adding a position without initializing strategy returns 400."""
        # Create a session but don't initialize strategy
        session = session_service.create_session()

        response = test_client.post(
            "/api/strategy/positions",
            json={"conid": 123456, "quantity": 2},
            cookies={"session_id": session.session_id}
        )

        assert response.status_code == 400
        data = response.json()
        assert "No strategy initialized" in data["error"]

    def test_add_position_contract_not_found(self, test_client, mock_ibkr_client):
        """Test that adding a non-existent contract returns 400."""
        session_id = self._create_session_with_strategy(test_client, mock_ibkr_client)

        # Mock option chain with different contracts
        mock_call = OptionContract(
            conid=999999,
            strike=150.0,
            right="C",
            expiration=date(2026, 1, 16),
            bid=2.50,
            ask=2.55,
            multiplier=100,
        )
        mock_chain = OptionChain(
            expiration=date(2026, 1, 16),
            calls=[mock_call],
            puts=[],
        )
        mock_ibkr_client.get_option_chain = AsyncMock(return_value=mock_chain)

        # Try to add a contract that doesn't exist
        response = test_client.post(
            "/api/strategy/positions",
            json={"conid": 123456, "quantity": 2},
            cookies={"session_id": session_id}
        )

        assert response.status_code == 400
        data = response.json()
        assert "CONTRACT_NOT_FOUND" in data["code"]
        assert "not found in option chain" in data["error"]

    def test_add_position_mixed_expiration(self, test_client, mock_ibkr_client):
        """Test that adding positions with mixed expirations returns 400."""
        session_id = self._create_session_with_strategy(test_client, mock_ibkr_client)

        # Add first position with JAN26 expiration
        mock_call1 = OptionContract(
            conid=123456,
            strike=150.0,
            right="C",
            expiration=date(2026, 1, 16),
            bid=2.50,
            ask=2.55,
            multiplier=100,
        )
        mock_chain1 = OptionChain(
            expiration=date(2026, 1, 16),
            calls=[mock_call1],
            puts=[],
        )
        mock_ibkr_client.get_option_chain = AsyncMock(return_value=mock_chain1)

        response = test_client.post(
            "/api/strategy/positions",
            json={"conid": 123456, "quantity": 2},
            cookies={"session_id": session_id}
        )
        assert response.status_code == 200

        # Try to add second position with FEB26 expiration
        mock_call2 = OptionContract(
            conid=123457,
            strike=150.0,
            right="C",
            expiration=date(2026, 2, 20),
            bid=3.50,
            ask=3.55,
            multiplier=100,
        )
        mock_chain2 = OptionChain(
            expiration=date(2026, 1, 16),  # Still JAN26 for target_date
            calls=[mock_call1, mock_call2],
            puts=[],
        )
        mock_ibkr_client.get_option_chain = AsyncMock(return_value=mock_chain2)

        response = test_client.post(
            "/api/strategy/positions",
            json={"conid": 123457, "quantity": 1},
            cookies={"session_id": session_id}
        )

        assert response.status_code == 400
        data = response.json()
        assert "MIXED_EXPIRATION" in data["code"]

    def test_add_multiple_positions_same_expiration(self, test_client, mock_ibkr_client):
        """Test adding multiple positions with the same expiration succeeds."""
        session_id = self._create_session_with_strategy(test_client, mock_ibkr_client)

        # Mock option chain with multiple contracts
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
            strike=145.0,
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

        # Add first position
        response1 = test_client.post(
            "/api/strategy/positions",
            json={"conid": 123456, "quantity": 2},
            cookies={"session_id": session_id}
        )
        assert response1.status_code == 200
        assert len(response1.json()["positions"]) == 1

        # Add second position with same expiration
        response2 = test_client.post(
            "/api/strategy/positions",
            json={"conid": 123457, "quantity": -1},
            cookies={"session_id": session_id}
        )
        assert response2.status_code == 200
        data = response2.json()
        assert len(data["positions"]) == 2
        assert data["positions"][0]["conid"] == 123456
        assert data["positions"][1]["conid"] == 123457


class TestModifyPositionEndpoint:
    """Test PATCH /api/strategy/positions/{conid} endpoint."""

    def _create_session_with_position(self, test_client, mock_ibkr_client):
        """Helper to create a session with a position."""
        # Initialize strategy
        mock_stock = Stock(
            symbol="AAPL",
            current_price=150.25,
            conid=265598,
            available_expirations=["JAN26"],
        )
        mock_ibkr_client.get_stock = AsyncMock(return_value=mock_stock)
        response = test_client.post("/api/strategy/init", json={"symbol": "AAPL"})
        session_id = response.cookies.get("session_id")

        # Add a position
        mock_call = OptionContract(
            conid=123456,
            strike=150.0,
            right="C",
            expiration=date(2026, 1, 16),
            bid=2.50,
            ask=2.55,
            multiplier=100,
        )
        mock_chain = OptionChain(
            expiration=date(2026, 1, 16),
            calls=[mock_call],
            puts=[],
        )
        mock_ibkr_client.get_option_chain = AsyncMock(return_value=mock_chain)
        test_client.post(
            "/api/strategy/positions",
            json={"conid": 123456, "quantity": 2},
            cookies={"session_id": session_id}
        )

        return session_id

    def test_modify_position_success(self, test_client, mock_ibkr_client):
        """Test successfully modifying a position's quantity."""
        session_id = self._create_session_with_position(test_client, mock_ibkr_client)

        response = test_client.patch(
            "/api/strategy/positions/123456",
            json={"quantity": 5},
            cookies={"session_id": session_id}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["positions"]) == 1
        assert data["positions"][0]["conid"] == 123456
        assert data["positions"][0]["quantity"] == 5

    def test_modify_position_to_negative(self, test_client, mock_ibkr_client):
        """Test modifying a long position to short (flip direction)."""
        session_id = self._create_session_with_position(test_client, mock_ibkr_client)

        response = test_client.patch(
            "/api/strategy/positions/123456",
            json={"quantity": -3},
            cookies={"session_id": session_id}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["positions"][0]["quantity"] == -3

    def test_modify_position_zero_quantity(self, test_client, mock_ibkr_client):
        """Test that modifying to zero quantity returns 400."""
        session_id = self._create_session_with_position(test_client, mock_ibkr_client)

        response = test_client.patch(
            "/api/strategy/positions/123456",
            json={"quantity": 0},
            cookies={"session_id": session_id}
        )

        assert response.status_code == 400
        data = response.json()
        assert "INVALID_QUANTITY" in data["code"]

    def test_modify_position_not_found(self, test_client, mock_ibkr_client):
        """Test that modifying a non-existent position returns 400."""
        session_id = self._create_session_with_position(test_client, mock_ibkr_client)

        response = test_client.patch(
            "/api/strategy/positions/999999",
            json={"quantity": 5},
            cookies={"session_id": session_id}
        )

        assert response.status_code == 400
        data = response.json()
        assert "POSITION_NOT_FOUND" in data["code"]
        assert "not found" in data["error"]

    def test_modify_position_no_session(self, test_client):
        """Test that modifying without a session returns 401."""
        response = test_client.patch(
            "/api/strategy/positions/123456",
            json={"quantity": 5}
        )

        assert response.status_code == 401

    def test_modify_position_no_strategy(self, test_client, session_service):
        """Test that modifying without initialized strategy returns 400."""
        session = session_service.create_session()

        response = test_client.patch(
            "/api/strategy/positions/123456",
            json={"quantity": 5},
            cookies={"session_id": session.session_id}
        )

        assert response.status_code == 400
        data = response.json()
        assert "No strategy initialized" in data["error"]


class TestDeletePositionEndpoint:
    """Test DELETE /api/strategy/positions/{conid} endpoint."""

    def _create_session_with_positions(self, test_client, mock_ibkr_client, count=2):
        """Helper to create a session with multiple positions."""
        # Initialize strategy
        mock_stock = Stock(
            symbol="AAPL",
            current_price=150.25,
            conid=265598,
            available_expirations=["JAN26"],
        )
        mock_ibkr_client.get_stock = AsyncMock(return_value=mock_stock)
        response = test_client.post("/api/strategy/init", json={"symbol": "AAPL"})
        session_id = response.cookies.get("session_id")

        # Add positions
        contracts = [
            OptionContract(
                conid=123456 + i,
                strike=150.0 + (i * 5),
                right="C" if i % 2 == 0 else "P",
                expiration=date(2026, 1, 16),
                bid=2.50,
                ask=2.55,
                multiplier=100,
            )
            for i in range(count)
        ]

        mock_chain = OptionChain(
            expiration=date(2026, 1, 16),
            calls=[c for c in contracts if c.right == "C"],
            puts=[c for c in contracts if c.right == "P"],
        )
        mock_ibkr_client.get_option_chain = AsyncMock(return_value=mock_chain)

        for i in range(count):
            test_client.post(
                "/api/strategy/positions",
                json={"conid": 123456 + i, "quantity": i + 1},
                cookies={"session_id": session_id}
            )

        return session_id

    def test_delete_position_success(self, test_client, mock_ibkr_client):
        """Test successfully deleting a position."""
        session_id = self._create_session_with_positions(test_client, mock_ibkr_client, count=2)

        response = test_client.delete(
            "/api/strategy/positions/123456",
            cookies={"session_id": session_id}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["positions"]) == 1
        assert data["positions"][0]["conid"] == 123457

    def test_delete_last_position(self, test_client, mock_ibkr_client):
        """Test deleting the last position returns empty list."""
        session_id = self._create_session_with_positions(test_client, mock_ibkr_client, count=1)

        response = test_client.delete(
            "/api/strategy/positions/123456",
            cookies={"session_id": session_id}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["positions"]) == 0

    def test_delete_position_not_found(self, test_client, mock_ibkr_client):
        """Test that deleting a non-existent position returns 400."""
        session_id = self._create_session_with_positions(test_client, mock_ibkr_client, count=1)

        response = test_client.delete(
            "/api/strategy/positions/999999",
            cookies={"session_id": session_id}
        )

        assert response.status_code == 400
        data = response.json()
        assert "POSITION_NOT_FOUND" in data["code"]

    def test_delete_position_no_session(self, test_client):
        """Test that deleting without a session returns 401."""
        response = test_client.delete("/api/strategy/positions/123456")

        assert response.status_code == 401

    def test_delete_position_no_strategy(self, test_client, session_service):
        """Test that deleting without initialized strategy returns 400."""
        session = session_service.create_session()

        response = test_client.delete(
            "/api/strategy/positions/123456",
            cookies={"session_id": session.session_id}
        )

        assert response.status_code == 400
        data = response.json()
        assert "No strategy initialized" in data["error"]

    def test_delete_middle_position(self, test_client, mock_ibkr_client):
        """Test deleting a position from the middle of the list."""
        session_id = self._create_session_with_positions(test_client, mock_ibkr_client, count=3)

        response = test_client.delete(
            "/api/strategy/positions/123457",
            cookies={"session_id": session_id}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["positions"]) == 2
        assert data["positions"][0]["conid"] == 123456
        assert data["positions"][1]["conid"] == 123458
