"""
Rate limiter for API request throttling.

Implements a token bucket algorithm to prevent exceeding rate limits
for external APIs (primarily IBKR).
"""

import asyncio
import time
from collections import deque


class RateLimiter:
    """
    Async rate limiter using token bucket algorithm.

    Ensures that no more than `max_requests` are made within `per_seconds` window.

    Example:
        >>> limiter = RateLimiter(max_requests=50, per_seconds=60)
        >>> async def fetch_data():
        ...     await limiter.acquire()
        ...     # Make API request
    """

    def __init__(self, max_requests: int, per_seconds: float) -> None:
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed
            per_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.requests: deque[float] = deque()
        self._lock = asyncio.Lock()

    def _cleanup_expired_requests(self, now: float) -> None:
        """
        Remove expired requests outside the time window.

        Args:
            now: Current timestamp
        """
        while self.requests and self.requests[0] <= now - self.per_seconds:
            self.requests.popleft()

    async def acquire(self) -> None:
        """
        Acquire permission to make a request.

        Blocks until a request slot is available within the rate limit.

        Note:
            This method is async and may sleep if rate limit is reached.
        """
        async with self._lock:
            now = time.time()
            self._cleanup_expired_requests(now)

            # If at capacity, wait until oldest request expires
            if len(self.requests) >= self.max_requests:
                sleep_time = self.requests[0] + self.per_seconds - now
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                # After sleep, clean up again
                now = time.time()
                self._cleanup_expired_requests(now)

            # Record this request
            self.requests.append(time.time())

    def reset(self) -> None:
        """
        Reset the rate limiter.

        Clears all recorded requests. Useful for testing.
        """
        self.requests.clear()
