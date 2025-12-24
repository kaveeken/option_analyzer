"""IBKR API client."""

import asyncio
import logging
from typing import Any
import httpx
from option_analyzer.config import Settings
from option_analyzer.clients.cache import CacheInterface
from option_analyzer.utils.rate_limiter import RateLimiter
from option_analyzer.utils.exceptions import (
    IBKRAPIError,
    IBKRConnectionError
)

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

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()

    async def _request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        for attempt in range(self.settings.ibkr_max_retries):
            try:
                await self._rate_limiter.acquire()
                response = await self.client.request(method=method, url=endpoint, **kwargs)
                response.raise_for_status()
                if "application/json" not in response.headers.get("content-type", ""):
                    raise IBKRAPIError("Expected json response not present")
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    if attempt < self.settings.ibkr_max_retries - 1:
                        logger.warning("Rate limited - retry with backoff")
                        await asyncio.sleep(self._calculate_backoff(attempt))
                        continue
                    raise IBKRAPIError("Maximum retries reached before response.") from e
                if e.response.status_code >= 500:
                    if attempt < self.settings.ibkr_max_retries - 1:
                        logger.warning("Server error - retry with backoff")
                        await asyncio.sleep(self._calculate_backoff(attempt))
                        continue
                raise IBKRAPIError(f"IBKR API error code {e.response.status_code}") from e
            except httpx.RequestError as e:
                if attempt < self.settings.ibkr_max_retries - 1:
                    logger.warning("Request error - retry with backoff")
                    await asyncio.sleep(self._calculate_backoff(attempt))
                    continue
                raise IBKRConnectionError() from e
        raise IBKRAPIError("Maximum retries reached before response.")

    def _calculate_backoff(self, attempt_count: int) -> float:
        return self.settings.ibkr_retry_delay * 2.0 ** attempt_count

    async def get_request(self, endpoint: str, **kwargs) -> dict[str, Any]:
        """Send get request"""
        return await self._request("GET", endpoint, **kwargs)

    async def aclose(self) -> None:
        await self.client.aclose()
