# Endpoint-Specific Rate Limiting for IBKR Client

## Overview
Implement endpoint-specific rate limiting to handle IBKR's differentiated API limits:
- **Global**: 10 requests/second across all endpoints
- **Snapshot**: Each conid in a batched request counts as 1 separate request
- **History**: Max 5 concurrent in-flight requests (future endpoint)

## Current Problem
The existing `RateLimiter` applies uniformly (line 58 in ibkr.py):
```python
await self._rate_limiter.acquire()  # Always consumes 1 token
```

When batching 9 conids in `price_option_chain()` (line 251-252), this should consume 9 tokens but only consumes 1, causing rate limit violations.

**Additional Bug Found**: Line 252 has incorrect slicing: `conids[i:1+CONSECUTIVE_CONIDS]` should be `conids[i:i+CONSECUTIVE_CONIDS]`

## Implementation Approach

### Phase 1: Enhance RateLimiter for Weighted Tokens

**File**: `src/option_analyzer/utils/rate_limiter.py`

Modify `acquire()` method to accept optional `tokens` parameter:

```python
async def acquire(self, tokens: int = 1) -> None:
    """
    Acquire N tokens from the bucket.

    Args:
        tokens: Number of tokens to acquire (default=1)

    Raises:
        ValueError: If tokens < 1 or tokens > max_requests
    """
    if tokens < 1:
        raise ValueError("tokens must be at least 1")
    if tokens > self.max_requests:
        raise ValueError(f"tokens ({tokens}) exceeds max_requests ({self.max_requests})")

    async with self._lock:
        now = time.time()
        self._cleanup_expired_requests(now)

        # Wait until we have enough capacity for all tokens
        while len(self.requests) + tokens > self.max_requests:
            sleep_time = self.requests[0] + self.per_seconds - now
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            now = time.time()
            self._cleanup_expired_requests(now)

        # Record N timestamps (simple approach, backward compatible with cleanup)
        for _ in range(tokens):
            self.requests.append(time.time())
```

**Key Design Decision**: Record multiple timestamps (one per token) rather than weighted tuples. This keeps `_cleanup_expired_requests()` unchanged and maintains simplicity.

### Phase 2: Update IBKRClient._request() for Weighted Tokens

**File**: `src/option_analyzer/clients/ibkr.py`

Add `rate_limit_tokens` parameter to `_request()`:

```python
async def _request(
    self,
    method: str,
    endpoint: str,
    rate_limit_tokens: int = 1,  # NEW parameter
    **kwargs: Any
) -> Any:
    for attempt in range(self.settings.ibkr_max_retries + 1):
        try:
            await self._rate_limiter.acquire(tokens=rate_limit_tokens)  # Pass weight
            response = await self.client.request(method=method, url=endpoint, **kwargs)
            # ... rest unchanged
```

Update `get_request()` to accept and forward the parameter:

```python
async def get_request(self, endpoint: str, rate_limit_tokens: int = 1, **kwargs: Any) -> Any:
    """Send get request"""
    return await self._request("GET", endpoint, rate_limit_tokens=rate_limit_tokens, **kwargs)
```

### Phase 3: Calculate Token Weight in get_market_snapshot()

**File**: `src/option_analyzer/clients/ibkr.py`

Modify `get_market_snapshot()` to count conids and pass weight:

```python
async def get_market_snapshot(self, conid: int|str, ttl: timedelta | None) -> list[dict[str, float|int|None]]:
    """
    Get current market data for given contract id(s).

    Note: Each conid in a batched request counts as 1 token.
    """
    endpoint = f"iserver/marketdata/snapshot?conids={conid}&fields=31,84,86"
    response = self._cache.get(endpoint)
    if response is None:
        # Count conids in the request
        conid_str = str(conid)
        conid_count = len(conid_str.split(','))

        # Acquire tokens equal to number of conids
        response = await self.get_request(endpoint, rate_limit_tokens=conid_count)
        self._cache.set(endpoint, response, ttl)
    # ... rest unchanged
```

### Phase 4: Add History Endpoint Concurrency Limit (Future)

**File**: `src/option_analyzer/clients/ibkr.py`

Add semaphore to `__init__`:

```python
def __init__(self, settings: Settings, cache: CacheInterface, rate_limiter: RateLimiter) -> None:
    # ... existing code ...
    self._history_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent history requests
```

Update `_request()` to handle semaphore for history endpoints:

```python
async def _request(
    self,
    method: str,
    endpoint: str,
    rate_limit_tokens: int = 1,
    **kwargs: Any
) -> Any:
    # Determine if this is a history endpoint
    use_semaphore = 'marketdata/history' in endpoint

    if use_semaphore:
        async with self._history_semaphore:
            return await self._do_request_with_retry(method, endpoint, rate_limit_tokens, **kwargs)
    else:
        return await self._do_request_with_retry(method, endpoint, rate_limit_tokens, **kwargs)

async def _do_request_with_retry(self, method, endpoint, rate_limit_tokens, **kwargs):
    """Extracted retry logic (existing code from _request)"""
    for attempt in range(self.settings.ibkr_max_retries + 1):
        # ... existing retry logic ...
```

### Phase 5: Fix Batching Bug

**File**: `src/option_analyzer/clients/ibkr.py` (line 252)

Fix incorrect slice in `price_option_chain()`:

```python
# BEFORE (BUG):
market_snapshot.append(await self.get_market_snapshot(",".join(conids[i:1+CONSECUTIVE_CONIDS]), None))

# AFTER (FIXED):
market_snapshot.append(await self.get_market_snapshot(",".join(conids[i:i+CONSECUTIVE_CONIDS]), None))
```

Also fix the iteration to properly flatten the nested list structure (line 254 suggests market_snapshot is list of lists).

### Phase 6: Add Comprehensive Tests

**File**: `tests/unit/test_rate_limiter.py`

Add test class for weighted tokens:

```python
class TestWeightedTokens:
    @pytest.mark.asyncio
    async def test_acquire_weighted_tokens(self):
        """Acquiring 3 tokens consumes 3 slots."""
        limiter = RateLimiter(max_requests=10, per_seconds=1.0)
        await limiter.acquire(tokens=3)
        assert len(limiter.requests) == 3

    @pytest.mark.asyncio
    async def test_weighted_blocks_when_insufficient_capacity(self):
        """Weighted acquire blocks if not enough tokens available."""
        limiter = RateLimiter(max_requests=10, per_seconds=1.0)
        await limiter.acquire(tokens=8)  # Use 8 tokens

        start = time.time()
        await limiter.acquire(tokens=5)  # Need 5, only 2 available
        elapsed = time.time() - start

        assert elapsed >= 1.0  # Should wait for window to slide

    @pytest.mark.asyncio
    async def test_weighted_validates_input(self):
        """Test validation of token parameter."""
        limiter = RateLimiter(max_requests=10, per_seconds=1.0)

        with pytest.raises(ValueError, match="at least 1"):
            await limiter.acquire(tokens=0)

        with pytest.raises(ValueError, match="exceeds"):
            await limiter.acquire(tokens=11)

    @pytest.mark.asyncio
    async def test_concurrent_weighted_requests(self):
        """Multiple concurrent weighted requests share the bucket correctly."""
        limiter = RateLimiter(max_requests=10, per_seconds=1.0)

        async def acquire_tokens(n):
            await limiter.acquire(tokens=n)

        start = time.time()
        # Launch 3+4+6=13 tokens concurrently (exceeds 10 capacity)
        await asyncio.gather(
            acquire_tokens(3),
            acquire_tokens(4),
            acquire_tokens(6)
        )
        elapsed = time.time() - start

        # Should block and wait for window
        assert elapsed >= 1.0
        assert len(limiter.requests) == 13
```

**File**: `tests/unit/test_ibkr.py`

Add integration tests for snapshot batching:

```python
class TestSnapshotRateLimiting:
    @pytest.mark.asyncio
    async def test_snapshot_batch_consumes_correct_tokens(self, ibkr_client):
        """Batched snapshot with 3 conids should consume 3 tokens."""
        # Mock the HTTP response
        mock_response = [
            {"conid": 1, "31": 100.0, "84": 99.5, "86": 100.5},
            {"conid": 2, "31": 200.0, "84": 199.5, "86": 200.5},
            {"conid": 3, "31": 300.0, "84": 299.5, "86": 300.5},
        ]

        with patch.object(ibkr_client.client, 'request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value.json.return_value = mock_response
            mock_request.return_value.headers = {"content-type": "application/json"}
            mock_request.return_value.raise_for_status = lambda: None

            # Request snapshot for 3 conids
            await ibkr_client.get_market_snapshot("1,2,3", None)

            # Verify 3 tokens were consumed
            assert len(ibkr_client._rate_limiter.requests) == 3
```

## Critical Files to Modify

1. **`src/option_analyzer/utils/rate_limiter.py`**: Add `tokens` parameter to `acquire()` method (lines 49-73)
2. **`src/option_analyzer/clients/ibkr.py`**:
   - Update `_request()` signature and implementation (lines 50-75)
   - Update `get_request()` to forward tokens parameter (line 103-105)
   - Update `get_market_snapshot()` to calculate conid count (lines 140-155)
   - Fix batching bug in `price_option_chain()` (line 252)
   - Add `_history_semaphore` to `__init__` (lines 23-37)
3. **`tests/unit/test_rate_limiter.py`**: Add `TestWeightedTokens` class
4. **`tests/unit/test_ibkr.py`**: Add `TestSnapshotRateLimiting` class

## Additional Cleanup
- Line 226: Remove or replace `pprint(strikes)` with proper logging
- Add missing `from pprint import pprint` import if keeping debug output

## Backward Compatibility
All changes are backward compatible:
- `tokens=1` default preserves existing behavior
- No breaking changes to public API
- Existing tests continue to pass

## Testing Strategy
1. Unit test weighted token acquisition in isolation
2. Unit test validation and edge cases
3. Integration test snapshot batching consumes correct tokens
4. Manual testing against IBKR API to verify no rate limit violations
