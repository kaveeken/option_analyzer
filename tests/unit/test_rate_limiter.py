"""
Unit tests for rate limiter.

Tests the async rate limiter to ensure it properly throttles requests.
"""

import asyncio
import time

import pytest

from option_analyzer.utils.rate_limiter import RateLimiter


class TestRateLimiter:
    """Test RateLimiter class."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_within_limit(self) -> None:
        """Test that requests within limit are allowed immediately."""
        limiter = RateLimiter(max_requests=5, per_seconds=1.0)

        start = time.time()
        # Make 5 requests (at the limit)
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.time() - start

        # Should complete almost immediately (< 0.1 seconds)
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_over_limit(self) -> None:
        """Test that requests exceeding limit are delayed."""
        limiter = RateLimiter(max_requests=5, per_seconds=1.0)

        start = time.time()
        # Make 10 requests (double the limit)
        for _ in range(10):
            await limiter.acquire()
        elapsed = time.time() - start

        # Should take at least 1 second (waiting for window to pass)
        assert elapsed >= 1.0
        # But not more than 2 seconds (some tolerance for processing)
        assert elapsed < 2.0

    @pytest.mark.asyncio
    async def test_rate_limiter_concurrent_requests(self) -> None:
        """Test rate limiter with concurrent async requests."""
        limiter = RateLimiter(max_requests=3, per_seconds=1.0)

        async def make_request(request_id: int) -> float:
            """Make a single request and return timestamp."""
            await limiter.acquire()
            return time.time()

        start = time.time()
        # Launch 6 concurrent requests
        tasks = [make_request(i) for i in range(6)]
        timestamps = await asyncio.gather(*tasks)
        elapsed = time.time() - start

        # Should take at least 1 second (second batch waits)
        assert elapsed >= 1.0

        # First 3 should complete quickly
        first_batch = timestamps[:3]
        assert max(first_batch) - min(first_batch) < 0.1

        # Last 3 should complete after the window
        last_batch = timestamps[3:]
        assert min(last_batch) - start >= 1.0

    @pytest.mark.asyncio
    async def test_rate_limiter_reset(self) -> None:
        """Test that reset clears all requests."""
        limiter = RateLimiter(max_requests=2, per_seconds=1.0)

        # Make 2 requests (at limit)
        await limiter.acquire()
        await limiter.acquire()

        # Reset
        limiter.reset()

        # Should be able to make 2 more requests immediately
        start = time.time()
        await limiter.acquire()
        await limiter.acquire()
        elapsed = time.time() - start

        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_rate_limiter_different_rates(self) -> None:
        """Test rate limiter with different rate configurations."""
        # Very permissive
        fast_limiter = RateLimiter(max_requests=100, per_seconds=1.0)
        start = time.time()
        for _ in range(50):
            await fast_limiter.acquire()
        elapsed = time.time() - start
        assert elapsed < 0.5  # Should be very fast

        # Very restrictive
        slow_limiter = RateLimiter(max_requests=1, per_seconds=0.5)
        start = time.time()
        for _ in range(3):
            await slow_limiter.acquire()
        elapsed = time.time() - start
        assert elapsed >= 1.0  # Should take at least 1 second (2 waits of 0.5s)
