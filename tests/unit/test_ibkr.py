"""
Unit tests for IBKR client.

Tests cover:
- Symbol lookup (get_conid) with caching
- Success cases (single match)
- Error cases (not found, ambiguous, API failures)
- Cache integration and TTL
- Asset type parameter handling
"""

from datetime import timedelta
from unittest.mock import AsyncMock, patch

import pytest

from option_analyzer.clients.cache import InMemoryCache
from option_analyzer.clients.ibkr import IBKRClient
from option_analyzer.config import Settings
from option_analyzer.utils.exceptions import (
    IBKRAPIError,
    IBKRConnectionError,
    SymbolNotFoundError,
)
from option_analyzer.utils.rate_limiter import RateLimiter


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings()


@pytest.fixture
def cache() -> InMemoryCache:
    """Create test cache."""
    return InMemoryCache()


@pytest.fixture
def rate_limiter() -> RateLimiter:
    """Create test rate limiter."""
    return RateLimiter(max_requests=50, per_seconds=1.0)


@pytest.fixture
async def client(
    settings: Settings, cache: InMemoryCache, rate_limiter: RateLimiter
) -> IBKRClient:
    """Create test IBKR client."""
    async with IBKRClient(settings, cache, rate_limiter) as client:
        yield client


class TestGetConid:
    """Test get_conid symbol lookup method."""

    @pytest.mark.asyncio
    async def test_get_conid_success(self, client: IBKRClient) -> None:
        """Test successful symbol lookup returns conid."""
        # Mock successful API response
        client.get_request = AsyncMock(
            return_value=[
                {"conid": 265598, "symbol": "AAPL", "description": "Apple Inc"}
            ]
        )

        conid = await client.get_conid("AAPL")

        assert conid == 265598
        assert isinstance(conid, int)

    @pytest.mark.asyncio
    async def test_get_conid_uses_cache(self, client: IBKRClient) -> None:
        """Test that symbol lookup uses cache on second call."""
        mock_response = [{"conid": 265598, "symbol": "AAPL"}]
        client.get_request = AsyncMock(return_value=mock_response)

        # First call - hits API
        conid1 = await client.get_conid("AAPL")

        # Second call - should use cache
        conid2 = await client.get_conid("AAPL")

        # Should only call API once
        assert client.get_request.call_count == 1
        assert conid1 == conid2 == 265598

    @pytest.mark.asyncio
    async def test_get_conid_cache_ttl_24h(self, client: IBKRClient) -> None:
        """Test that cache uses 24-hour TTL."""
        with patch.object(client._cache, "set") as mock_cache_set:
            client.get_request = AsyncMock(
                return_value=[{"conid": 265598, "symbol": "AAPL"}]
            )

            await client.get_conid("AAPL")

            # Verify cache.set was called with 24h TTL
            assert mock_cache_set.called
            call_args = mock_cache_set.call_args
            # Third argument should be the TTL
            assert call_args[0][2] == timedelta(hours=24)

    @pytest.mark.asyncio
    async def test_get_conid_symbol_not_found_empty_list(
        self, client: IBKRClient
    ) -> None:
        """Test that empty results raise SymbolNotFoundError."""
        # Mock empty response
        client.get_request = AsyncMock(return_value=[])

        with pytest.raises(SymbolNotFoundError) as exc_info:
            await client.get_conid("INVALID")

        assert "INVALID" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_conid_invalid_response_format(self, client: IBKRClient) -> None:
        """Test that malformed API response raises SymbolNotFoundError."""
        # Mock malformed response (not a list of dicts)
        client.get_request = AsyncMock(return_value={"error": "bad format"})

        with pytest.raises(SymbolNotFoundError):
            await client.get_conid("AAPL")

    @pytest.mark.asyncio
    async def test_get_conid_invalid_response_not_list(
        self, client: IBKRClient
    ) -> None:
        """Test that non-list response raises SymbolNotFoundError."""
        # Mock response that's not a list
        client.get_request = AsyncMock(return_value="not a list")

        with pytest.raises(SymbolNotFoundError):
            await client.get_conid("AAPL")

    @pytest.mark.asyncio
    async def test_get_conid_missing_conid_field(self, client: IBKRClient) -> None:
        """Test that missing conid field raises appropriate error."""
        # Mock response with dict but no conid field
        client.get_request = AsyncMock(return_value=[{"symbol": "AAPL", "description": "Apple Inc"}])

        with pytest.raises(KeyError):
            await client.get_conid("AAPL")

    @pytest.mark.asyncio
    async def test_get_conid_connection_error(self, client: IBKRClient) -> None:
        """Test that connection failures raise IBKRConnectionError."""
        # Mock connection error
        client.get_request = AsyncMock(side_effect=IBKRConnectionError())

        with pytest.raises(IBKRConnectionError):
            await client.get_conid("AAPL")

    @pytest.mark.asyncio
    async def test_get_conid_api_error_500(self, client: IBKRClient) -> None:
        """Test that server errors raise IBKRAPIError."""
        # Mock API error after retries exhausted
        client.get_request = AsyncMock(
            side_effect=IBKRAPIError(
                "Error with status code 500: Internal Server Error"
            )
        )

        with pytest.raises(IBKRAPIError) as exc_info:
            await client.get_conid("AAPL")

        assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_conid_api_error_400(self, client: IBKRClient) -> None:
        """Test that client errors raise IBKRAPIError."""
        # Mock API error for bad request
        client.get_request = AsyncMock(
            side_effect=IBKRAPIError("Error with status code 400: Bad Request")
        )

        with pytest.raises(IBKRAPIError) as exc_info:
            await client.get_conid("AAPL")

        assert "400" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_conid_picks_first_result(self, client: IBKRClient) -> None:
        """Test that multiple results returns the first (SMART) conid."""
        # Mock multiple results (ambiguous)
        client.get_request = AsyncMock(
            return_value=[
                {"conid": 265598, "symbol": "AAPL", "exchange": "SMART"},
                {"conid": 123456, "symbol": "AAPL", "exchange": "NASDAQ"},
            ]
        )

        conid = await client.get_conid("AAPL")

        # Should return first result
        assert conid == 265598

    @pytest.mark.asyncio
    async def test_get_conid_default_asset_type_stk(self, client: IBKRClient) -> None:
        """Test that default asset type is STK."""
        client.get_request = AsyncMock(
            return_value=[{"conid": 265598, "symbol": "AAPL"}]
        )

        await client.get_conid("AAPL")

        # Verify the endpoint included default asset_type=STK
        client.get_request.assert_called_once()
        call_args = client.get_request.call_args[0][0]
        assert "assetType=STK" in call_args
        assert "symbol=AAPL" in call_args

    @pytest.mark.asyncio
    async def test_get_conid_custom_asset_type(self, client: IBKRClient) -> None:
        """Test that custom asset_type parameter is passed correctly."""
        client.get_request = AsyncMock(
            return_value=[{"conid": 999999, "symbol": "ES", "assetType": "FUT"}]
        )

        # Request futures contract
        conid = await client.get_conid("ES", asset_type="FUT")

        assert conid == 999999
        # Verify the endpoint included custom asset_type
        client.get_request.assert_called_once()
        call_args = client.get_request.call_args[0][0]
        assert "assetType=FUT" in call_args
        assert "symbol=ES" in call_args

    @pytest.mark.asyncio
    async def test_get_conid_different_symbols_different_cache_entries(
        self, client: IBKRClient
    ) -> None:
        """Test that different symbols create separate cache entries."""
        client.get_request = AsyncMock(
            side_effect=[
                [{"conid": 265598, "symbol": "AAPL"}],
                [{"conid": 456789, "symbol": "MSFT"}],
            ]
        )

        conid1 = await client.get_conid("AAPL")
        conid2 = await client.get_conid("MSFT")

        # Should call API twice (different symbols)
        assert client.get_request.call_count == 2
        assert conid1 == 265598
        assert conid2 == 456789

    @pytest.mark.asyncio
    async def test_get_conid_integer_return_type(self, client: IBKRClient) -> None:
        """Test that conid is returned as integer even if API returns string."""
        # Mock response with string conid
        client.get_request = AsyncMock(
            return_value=[{"conid": "265598", "symbol": "AAPL"}]
        )

        conid = await client.get_conid("AAPL")

        assert conid == 265598
        assert isinstance(conid, int)


class TestGetSearchResults:
    """Test get_search_results helper method."""

    @pytest.mark.asyncio
    async def test_get_search_results_success(self, client: IBKRClient) -> None:
        """Test successful search results."""
        mock_response = [
            {"conid": 265598, "symbol": "AAPL", "description": "Apple Inc"}
        ]
        client.get_request = AsyncMock(return_value=mock_response)

        results = await client.get_search_results("AAPL", "STK", timedelta(hours=24))

        assert results == mock_response
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_search_results_caching(self, client: IBKRClient) -> None:
        """Test that search results are cached."""
        mock_response = [{"conid": 265598, "symbol": "AAPL"}]
        client.get_request = AsyncMock(return_value=mock_response)

        # First call
        results1 = await client.get_search_results("AAPL", "STK", timedelta(hours=24))

        # Second call - should use cache
        results2 = await client.get_search_results("AAPL", "STK", timedelta(hours=24))

        # Should only call API once
        assert client.get_request.call_count == 1
        assert results1 == results2

    @pytest.mark.asyncio
    async def test_get_search_results_no_ttl(self, client: IBKRClient) -> None:
        """Test search results with no TTL."""
        mock_response = [{"conid": 265598, "symbol": "AAPL"}]
        client.get_request = AsyncMock(return_value=mock_response)

        results = await client.get_search_results("AAPL", "STK", None)

        assert results == mock_response

    @pytest.mark.asyncio
    async def test_get_search_results_invalid_response(self, client: IBKRClient) -> None:
        """Test that invalid search results raise SymbolNotFoundError."""
        client.get_request = AsyncMock(return_value={"error": "not a list"})

        with pytest.raises(SymbolNotFoundError):
            await client.get_search_results("INVALID", "STK", timedelta(hours=24))


class TestGetStock:
    """Test get_stock method that combines search and snapshot."""

    @pytest.mark.asyncio
    async def test_get_stock_success(self, client: IBKRClient) -> None:
        """Test successful stock quote retrieval."""
        # Mock search results with option expirations
        mock_search = [
            {
                "conid": 265598,
                "symbol": "AAPL",
                "sections": [
                    {"secType": "STK"},
                    {"secType": "OPT", "months": "DEC24;JAN25;FEB25"},
                ],
            }
        ]
        # Mock market snapshot
        mock_snapshot = [{"conid": 265598, "last": 150.25, "bid": 150.20, "ask": 150.30}]

        client.get_search_results = AsyncMock(return_value=mock_search)
        client.get_market_snapshot = AsyncMock(return_value=mock_snapshot)

        stock = await client.get_stock("AAPL")

        # Verify Stock object fields
        assert stock.symbol == "AAPL"
        assert stock.current_price == 150.25
        assert stock.conid == 265598
        assert stock.available_expirations == ["DEC24", "JAN25", "FEB25"]

    @pytest.mark.asyncio
    async def test_get_stock_uses_5min_cache_for_snapshot(
        self, client: IBKRClient
    ) -> None:
        """Test that market snapshot uses 5-minute cache TTL."""
        mock_search = [
            {
                "conid": 265598,
                "symbol": "AAPL",
                "sections": [{"secType": "OPT", "months": "DEC24"}],
            }
        ]
        mock_snapshot = [{"conid": 265598, "last": 150.0, "bid": 149.95, "ask": 150.05}]

        client.get_search_results = AsyncMock(return_value=mock_search)
        client.get_market_snapshot = AsyncMock(return_value=mock_snapshot)

        await client.get_stock("AAPL")

        # Verify get_market_snapshot was called with 5min TTL
        client.get_market_snapshot.assert_called_once_with(
            265598, timedelta(minutes=5)
        )

    @pytest.mark.asyncio
    async def test_get_stock_no_option_sections(self, client: IBKRClient) -> None:
        """Test stock with no option sections returns empty expirations."""
        mock_search = [
            {
                "conid": 265598,
                "symbol": "AAPL",
                "sections": [{"secType": "STK"}],  # No OPT section
            }
        ]
        mock_snapshot = [{"conid": 265598, "last": 150.0, "bid": 149.95, "ask": 150.05}]

        client.get_search_results = AsyncMock(return_value=mock_search)
        client.get_market_snapshot = AsyncMock(return_value=mock_snapshot)

        stock = await client.get_stock("AAPL")

        assert stock.available_expirations == []

    @pytest.mark.asyncio
    async def test_get_stock_empty_sections(self, client: IBKRClient) -> None:
        """Test stock with empty sections array."""
        mock_search = [
            {
                "conid": 265598,
                "symbol": "AAPL",
                "sections": [],  # Empty sections
            }
        ]
        mock_snapshot = [{"conid": 265598, "last": 150.0, "bid": 149.95, "ask": 150.05}]

        client.get_search_results = AsyncMock(return_value=mock_search)
        client.get_market_snapshot = AsyncMock(return_value=mock_snapshot)

        stock = await client.get_stock("AAPL")

        assert stock.available_expirations == []

    @pytest.mark.asyncio
    async def test_get_stock_multiple_opt_sections(self, client: IBKRClient) -> None:
        """Test that the last OPT section's months are used (loop overwrites)."""
        mock_search = [
            {
                "conid": 265598,
                "symbol": "AAPL",
                "sections": [
                    {"secType": "OPT", "months": "DEC24;JAN25"},
                    {"secType": "OPT", "months": "FEB25;MAR25"},  # Last one wins
                ],
            }
        ]
        mock_snapshot = [{"conid": 265598, "last": 150.0, "bid": 149.95, "ask": 150.05}]

        client.get_search_results = AsyncMock(return_value=mock_search)
        client.get_market_snapshot = AsyncMock(return_value=mock_snapshot)

        stock = await client.get_stock("AAPL")

        # Last OPT section's months (implementation overwrites in loop)
        assert stock.available_expirations == ["FEB25", "MAR25"]

    @pytest.mark.asyncio
    async def test_get_stock_conid_as_string_in_response(
        self, client: IBKRClient
    ) -> None:
        """Test that conid is properly converted even if returned as string."""
        mock_search = [
            {
                "conid": "265598",  # String instead of int
                "symbol": "AAPL",
                "sections": [],
            }
        ]
        mock_snapshot = [{"conid": 265598, "last": 150.0, "bid": 149.95, "ask": 150.05}]

        client.get_search_results = AsyncMock(return_value=mock_search)
        client.get_market_snapshot = AsyncMock(return_value=mock_snapshot)

        stock = await client.get_stock("AAPL")

        assert stock.conid == 265598
        assert isinstance(stock.conid, int)

    @pytest.mark.asyncio
    async def test_get_stock_symbol_not_found(self, client: IBKRClient) -> None:
        """Test that symbol not found errors propagate."""
        client.get_search_results = AsyncMock(
            side_effect=SymbolNotFoundError("INVALID")
        )

        with pytest.raises(SymbolNotFoundError) as exc_info:
            await client.get_stock("INVALID")

        assert "INVALID" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_stock_api_error(self, client: IBKRClient) -> None:
        """Test that API errors from search propagate."""
        client.get_search_results = AsyncMock(
            side_effect=IBKRAPIError("Error with status code 500")
        )

        with pytest.raises(IBKRAPIError) as exc_info:
            await client.get_stock("AAPL")

        assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_stock_snapshot_error(self, client: IBKRClient) -> None:
        """Test that API errors from snapshot propagate."""
        mock_search = [
            {
                "conid": 265598,
                "symbol": "AAPL",
                "sections": [],
            }
        ]
        client.get_search_results = AsyncMock(return_value=mock_search)
        client.get_market_snapshot = AsyncMock(
            side_effect=IBKRAPIError("Snapshot error")
        )

        with pytest.raises(IBKRAPIError) as exc_info:
            await client.get_stock("AAPL")

        assert "Snapshot error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_stock_connection_error(self, client: IBKRClient) -> None:
        """Test that connection errors propagate."""
        client.get_search_results = AsyncMock(side_effect=IBKRConnectionError())

        with pytest.raises(IBKRConnectionError):
            await client.get_stock("AAPL")
