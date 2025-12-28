"""IBKR API client."""

import asyncio
import logging
import types
from datetime import timedelta
from typing import Any

import httpx

from option_analyzer.clients.cache import CacheInterface
from option_analyzer.config import Settings
from option_analyzer.utils.exceptions import IBKRAPIError, IBKRConnectionError, SymbolNotFoundError
from option_analyzer.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class IBKRClient:
    """Async HTTP client for IBKR API with retry logic and rate limiting."""

    def __init__(
        self, settings: Settings, cache: CacheInterface, rate_limiter: RateLimiter
    ) -> None:
        """
        Initialize a httpx.AsyncClient with parameters from Settings,
        and take cache and ratelimiter by reference.
        """
        self.client = httpx.AsyncClient(
            base_url=settings.ibkr_base_url,
            timeout=settings.ibkr_timeout,
            verify=settings.ibkr_verify_ssl,
        )
        self.settings = settings
        self._cache = cache
        self._rate_limiter = rate_limiter

    async def __aenter__(self) -> "IBKRClient":
        return self

    async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: types.TracebackType | None
    ) -> None:
        await self.aclose()

    async def _request(
            self,
            method: str,
            endpoint: str,
            **kwargs: Any # must correspond to httpx.AsyncClient.request parameters
    ) -> Any:
        for attempt in range(self.settings.ibkr_max_retries + 1):
            try:
                await self._rate_limiter.acquire()
                response = await self.client.request(method=method, url=endpoint, **kwargs)
                response.raise_for_status()
                if "application/json" not in response.headers.get("content-type", ""):
                    raise IBKRAPIError("Expected json response not present")
                return response.json()
            except httpx.HTTPStatusError as e:
                await self._handleStatusError(e, attempt)
                continue
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as e:
                if attempt < self.settings.ibkr_max_retries:
                    logger.warning(
                        f"Connection error - retry attempt {attempt + 1}/{self.settings.ibkr_max_retries}"
                    )
                    await asyncio.sleep(self._calculate_backoff(attempt))
                    continue
                raise IBKRConnectionError() from e
        raise IBKRAPIError("Maximum retries reached before response.")

    def _calculate_backoff(self, attempt_count: int) -> float:
        return self.settings.ibkr_retry_delay * 2.0**attempt_count

    async def _handleStatusError(self, error: httpx.HTTPStatusError, attempt: int) -> None:
        status_code = error.response.status_code
        error_body = error.response.text
        if status_code == 429:
            if attempt < self.settings.ibkr_max_retries:
                logger.warning(
                    f"Rate limited - retry attempt {attempt + 1}/{self.settings.ibkr_max_retries}"
                )
                await asyncio.sleep(self._calculate_backoff(attempt))
            else:
                raise IBKRAPIError("Maximum retries reached before response.") from error
        elif status_code >= 500:
            if attempt < self.settings.ibkr_max_retries:
                logger.warning(
                    f"Server error {error.response.status_code} - "
                    f"retry attempt {attempt + 1}/{self.settings.ibkr_max_retries}"
                )
                await asyncio.sleep(self._calculate_backoff(attempt))
            else:
                raise IBKRAPIError(f"Error with status code {status_code}: {error_body}") from error
        elif status_code >= 400:
            raise IBKRAPIError(f"Error with status code {status_code}: {error_body}") from error

    async def get_request(self, endpoint: str, **kwargs: Any) -> Any:
        """Send get request"""
        return await self._request("GET", endpoint, **kwargs)

    async def get_search_results(self, symbol: str, asset_type: str, ttl: timedelta | None) -> list[dict[str, Any]]:
        endpoint = f"iserver/secdef/search?symbol={symbol}&name=false&assetType={asset_type}"
        response = self._cache.get(endpoint)
        if response is None:
            response = await self.get_request(endpoint)
            self._cache.set(endpoint, response, ttl)
        if not isinstance(response, list) or not response:
            raise SymbolNotFoundError(symbol)
        return response

    async def get_conid(self, symbol: str, asset_type: str = "STK") -> int:
        """
        Get contract id for symbol.
        Ambiguous results raise AmbiguousSymbolError,
        and can be retried with specific primary_exchange or asset_type
        """
        result = await self.get_search_results(symbol, asset_type, timedelta(hours=24))
        # the first result is assumed to be the correct/SMART choice, but this is not validated
        return int(result[0]["conid"])

    async def aclose(self) -> None:
        await self.client.aclose()
