"""
Cache interface and implementations.

Provides a protocol-based cache interface with in-memory implementation
supporting TTL (time-to-live) for automatic expiration.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import RLock
from typing import Any, Protocol


class CacheInterface(Protocol):
    """
    Protocol for cache implementations.

    All cache implementations should provide these methods to ensure
    compatibility with the rest of the application.
    """

    def get(self, key: str) -> Any | None:
        """
        Retrieve a value from the cache.

        Args:
            key: Cache key to retrieve

        Returns:
            The cached value, or None if not found or expired
        """
        ...

    def set(self, key: str, value: Any, ttl: timedelta | None = None) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Value to store (can be any type)
            ttl: Optional time-to-live. If None, uses default TTL or never expires
        """
        ...

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.

        Args:
            key: Cache key to remove

        Returns:
            True if the key existed and was removed, False otherwise
        """
        ...

    def clear(self) -> None:
        """Remove all entries from the cache."""
        ...

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key: Cache key to check

        Returns:
            True if the key exists and hasn't expired, False otherwise
        """
        ...


@dataclass
class _CacheEntry:
    """
    Internal cache entry with expiration tracking.

    Attributes:
        value: The cached value
        expires_at: When this entry expires, or None for no expiration
    """

    value: Any
    expires_at: datetime | None = None


class InMemoryCache:
    """
    Thread-safe in-memory cache with TTL support.

    This implementation stores all data in memory using a dictionary.
    Expired entries are lazily cleaned up when accessed.

    Attributes:
        default_ttl: Default TTL for entries (None means no expiration)

    Example:
        >>> cache = InMemoryCache(default_ttl=timedelta(minutes=5))
        >>> cache.set("user:123", {"name": "Alice"})
        >>> cache.get("user:123")
        {'name': 'Alice'}
        >>> cache.set("temp", "value", ttl=timedelta(seconds=1))
        >>> time.sleep(2)
        >>> cache.get("temp")
        None
    """

    def __init__(self, default_ttl: timedelta | None = None) -> None:
        """
        Initialize the cache.

        Args:
            default_ttl: Default TTL for cache entries. None means no expiration.
        """
        self._store: dict[str, _CacheEntry] = {}
        self._default_ttl = default_ttl
        self._lock = RLock()

    def get(self, key: str) -> Any | None:
        """
        Retrieve a value from the cache.

        Args:
            key: Cache key to retrieve

        Returns:
            The cached value, or None if not found or expired

        Note:
            Expired entries are automatically removed when accessed.
        """
        with self._lock:
            entry = self._store.get(key)

            if entry is None:
                return None

            # Check expiration
            if self._is_expired(entry):
                del self._store[key]
                return None

            return entry.value

    def set(self, key: str, value: Any, ttl: timedelta | None = None) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Value to store (can be any type)
            ttl: Optional time-to-live. If None, uses default_ttl from constructor.

        Note:
            If both ttl and default_ttl are None, the entry never expires.
        """
        with self._lock:
            expires_at = self._calculate_expiration(ttl)
            self._store[key] = _CacheEntry(value=value, expires_at=expires_at)

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.

        Args:
            key: Cache key to remove

        Returns:
            True if the key existed and was removed, False otherwise
        """
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._store.clear()

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key: Cache key to check

        Returns:
            True if the key exists and hasn't expired, False otherwise

        Note:
            This does NOT remove expired entries. Use get() to trigger cleanup.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False
            return not self._is_expired(entry)

    def _is_expired(self, entry: _CacheEntry) -> bool:
        """
        Check if a cache entry has expired.

        Args:
            entry: Cache entry to check

        Returns:
            True if expired, False otherwise
        """
        if entry.expires_at is None:
            return False
        return datetime.now() >= entry.expires_at

    def _calculate_expiration(self, ttl: timedelta | None) -> datetime | None:
        """
        Calculate expiration time based on TTL.

        Args:
            ttl: Time-to-live, or None to use default

        Returns:
            Expiration datetime, or None for no expiration
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        if effective_ttl is None:
            return None
        return datetime.now() + effective_ttl
