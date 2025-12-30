"""IBKR API client."""

import asyncio
import logging
import types
from datetime import date, datetime, timedelta
from typing import Any, Literal

import httpx

from option_analyzer.clients.cache import CacheInterface
from option_analyzer.config import Settings
from option_analyzer.models.domain import OptionChain, OptionContract, Stock
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

    async def get_search_results(
            self,
            symbol: str,
            asset_type: str | None=None,
            ttl: timedelta | None=None
    ) -> list[dict[str, Any]]:
        endpoint = f"iserver/secdef/search?symbol={symbol}"
        if asset_type is not None:
            endpoint += f"&assetType={asset_type}"
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
        """
        result = await self.get_search_results(symbol, asset_type, timedelta(hours=24))
        # the first result is assumed to be the correct/SMART choice, but this is not validated
        return int(result[0]["conid"])

    async def get_market_snapshot(self, conid: int|str, ttl: timedelta | None) -> list[dict[str, float|int|None]]:
        """
        Get current market data for given contract id.
        Requested fields:
        31: last price
        84: bid price
        86: ask price
        """
        endpoint = f"iserver/marketdata/snapshot?conids={conid}&fields=31,84,86"
        response = self._cache.get(endpoint)
        if response is None:
            response = await self.get_request(endpoint)
            self._cache.set(endpoint, response, ttl)
        if not isinstance(response, list) or len(response) == 0:
            raise IBKRAPIError("Invalid marked data snapshot")
        return [self._parse_market_snapshot(entry) for entry in response]

    def _parse_market_snapshot(self, snapshot_entry: dict[str, Any]) -> dict[str, float|int|None]:
        return {
            "conid": int(snapshot_entry["conid"]),
            "last": snapshot_entry.get("31", None),
            "bid": snapshot_entry.get("84", None),
            "ask": snapshot_entry.get("86", None)}
        
    async def get_stock(self, symbol: str) -> Stock:
        results = await self.get_search_results(symbol)
        conid = int(results[0]["conid"])
        months = []
        for section in results[0]["sections"]:
            if section["secType"] == "OPT":
                months = section["months"].split(";")
        snapshot = await self.get_market_snapshot(conid, timedelta(minutes=5))
        if len(snapshot) == 0:
            raise IBKRAPIError("Invalid marked data snapshot")
        return Stock(symbol=symbol,
                     current_price=snapshot[0]["last"],
                     conid=conid,
                     available_expirations=months)

    async def get_option_strikes(
            self,
            conid: int,
            month: str,
            ttl: timedelta | None
            ) -> dict[str, list[float]]: # keep strikes as string instead of float
        endpoint = f"iserver/secdef/strikes?conid={conid}&secType=OPT&month={month}"
        response = self._cache.get(endpoint) # duplicate code
        if response is None:
            response = await self.get_request(endpoint)
            self._cache.set(endpoint, response, ttl)
        if not isinstance(response, dict) or len(response) != 2:
            raise IBKRAPIError("Invalid option strikes")
        return response

    async def get_unpriced_option_contract(
            self,
            underlying_conid: int,
            month: str,
            right: Literal["C", "P"],
            strike: float,
            ttl: timedelta | None
            ) -> OptionContract:
        endpoint =\
            f"iserver/secdef/info?conid={underlying_conid}&secType=OPT&month={month}" +\
            f"&strike={strike}&right={right}"
        response = self._cache.get(endpoint) # duplicate code
        if response is None:
            response = await self.get_request(endpoint)
            self._cache.set(endpoint, response, ttl)
        if not isinstance(response, list) or len(response) != 1:
            raise IBKRAPIError("Invalid contract info response")
        unpriced_contract =  OptionContract(
            conid = int(response[0]["conid"]),
            strike = strike,
            right=right,
            expiration = datetime.strptime(response[0]["maturityDate"], "%Y%m%d").date()) # maturityDate is YYYYMMDD
        return unpriced_contract
            
    async def get_option_chain(
            self,
            conid: int,
            month: str,
            ) -> OptionChain:
        """
        Get the option chain for a given conid at a given month.
        """
        strikes = await self.get_option_strikes(conid, month, None) # ttl
        conids_string = ""
        calls = {}
        puts = {}
        for strike in strikes["call"]:
            contract = await self.get_unpriced_option_contract(conid, month, "C", strike, None)
            if len(conids_string) > 0:
                conids_string += ","
            conids_string += str(contract.conid)
            calls[contract.conid] =  contract
        for strike in strikes["put"]:
            contract = await self.get_unpriced_option_contract(conid, month, "P", strike, None)
            conids_string += ","
            conids_string += str(contract.conid)
            puts[contract.conid] =  contract
        market_snapshot = await self.get_market_snapshot(conids_string, None)
        for snapshot_element in market_snapshot:
            contract_conid = snapshot_element["conid"]
            if contract_conid in calls:
                calls[contract_conid].ask = snapshot_element["86"]
                calls[contract_conid].bid = snapshot_element["84"]
            elif contract_conid in puts:
                puts[contract_conid].ask = snapshot_element["86"]
                puts[contract_conid].bid = snapshot_element["84"]
            else:
                raise IBKRAPIError("Market snapshot conid does not correspond to contract")


    async def aclose(self) -> None:
        await self.client.aclose()
