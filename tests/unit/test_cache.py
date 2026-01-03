"""
Unit tests for cache implementations.

Tests cover:
- Basic cache operations (get, set, delete, clear, exists)
- TTL behavior (default TTL, per-key TTL, expiration)
- Edge cases (non-existent keys, None values, expired entries)
- Thread safety (basic concurrent access)
- Protocol compliance
"""

import time
from datetime import timedelta
from threading import Thread

from option_analyzer.clients.cache import CacheInterface, InMemoryCache


class TestInMemoryCache:
    """Test InMemoryCache implementation."""

    def test_cache_creation(self) -> None:
        """Test basic cache creation."""
        cache = InMemoryCache()
        assert cache is not None

        # With default TTL
        cache_with_ttl = InMemoryCache(default_ttl=timedelta(seconds=60))
        assert cache_with_ttl is not None

    def test_get_set_basic(self) -> None:
        """Test basic get and set operations."""
        cache = InMemoryCache()

        # Set and get a value
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Set and get different types
        cache.set("string", "hello")
        cache.set("int", 42)
        cache.set("list", [1, 2, 3])
        cache.set("dict", {"a": 1, "b": 2})

        assert cache.get("string") == "hello"
        assert cache.get("int") == 42
        assert cache.get("list") == [1, 2, 3]
        assert cache.get("dict") == {"a": 1, "b": 2}

    def test_get_nonexistent_key(self) -> None:
        """Test getting a key that doesn't exist."""
        cache = InMemoryCache()
        assert cache.get("nonexistent") is None

    def test_set_with_none_value(self) -> None:
        """Test setting None as a value."""
        cache = InMemoryCache()
        cache.set("key", None)
        # None is a valid value, should be retrievable
        assert cache.get("key") is None
        # But should exist
        assert cache.exists("key") is True

    def test_overwrite_value(self) -> None:
        """Test overwriting an existing value."""
        cache = InMemoryCache()
        cache.set("key", "value1")
        cache.set("key", "value2")
        assert cache.get("key") == "value2"

    def test_delete_existing_key(self) -> None:
        """Test deleting an existing key."""
        cache = InMemoryCache()
        cache.set("key", "value")
        assert cache.delete("key") is True
        assert cache.get("key") is None

    def test_delete_nonexistent_key(self) -> None:
        """Test deleting a key that doesn't exist."""
        cache = InMemoryCache()
        assert cache.delete("nonexistent") is False

    def test_clear(self) -> None:
        """Test clearing all cache entries."""
        cache = InMemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None

    def test_clear_empty_cache(self) -> None:
        """Test clearing an already empty cache."""
        cache = InMemoryCache()
        cache.clear()  # Should not raise
        assert cache.get("any_key") is None

    def test_exists_present_key(self) -> None:
        """Test exists for a present key."""
        cache = InMemoryCache()
        cache.set("key", "value")
        assert cache.exists("key") is True

    def test_exists_nonexistent_key(self) -> None:
        """Test exists for a nonexistent key."""
        cache = InMemoryCache()
        assert cache.exists("nonexistent") is False

    def test_exists_after_delete(self) -> None:
        """Test exists after deleting a key."""
        cache = InMemoryCache()
        cache.set("key", "value")
        cache.delete("key")
        assert cache.exists("key") is False

    def test_ttl_per_key(self) -> None:
        """Test TTL on individual keys."""
        cache = InMemoryCache()

        # Set with short TTL
        cache.set("short_lived", "value", ttl=timedelta(milliseconds=50))
        assert cache.get("short_lived") == "value"

        # Wait for expiration
        time.sleep(0.1)
        assert cache.get("short_lived") is None

    def test_ttl_per_key_exists(self) -> None:
        """Test exists returns False for expired keys."""
        cache = InMemoryCache()
        cache.set("key", "value", ttl=timedelta(milliseconds=50))

        # Before expiration
        assert cache.exists("key") is True

        # After expiration
        time.sleep(0.1)
        assert cache.exists("key") is False

    def test_default_ttl(self) -> None:
        """Test default TTL on cache instance."""
        cache = InMemoryCache(default_ttl=timedelta(milliseconds=50))

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"

        # Wait for expiration
        time.sleep(0.1)

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_ttl_override_default(self) -> None:
        """Test that per-key TTL overrides default TTL."""
        cache = InMemoryCache(default_ttl=timedelta(milliseconds=100))

        # This should expire quickly despite default TTL
        cache.set("short", "value", ttl=timedelta(milliseconds=50))

        # This should use default TTL
        cache.set("default", "value")

        time.sleep(0.07)

        # Short-lived should be gone
        assert cache.get("short") is None

        # Default should still exist
        assert cache.get("default") == "value"

    def test_no_ttl_permanent(self) -> None:
        """Test that entries with no TTL don't expire."""
        cache = InMemoryCache()  # No default TTL
        cache.set("permanent", "value")  # No per-key TTL

        time.sleep(0.1)

        # Should still exist
        assert cache.get("permanent") == "value"
        assert cache.exists("permanent") is True

    def test_expired_entry_cleanup_on_get(self) -> None:
        """Test that expired entries are removed when accessed via get."""
        cache = InMemoryCache()
        cache.set("key", "value", ttl=timedelta(milliseconds=50))

        # Entry exists
        assert cache.exists("key") is True

        time.sleep(0.1)

        # After expiration, exists still sees it (lazy cleanup)
        assert cache.exists("key") is False

        # But get triggers cleanup
        assert cache.get("key") is None

        # Now it's truly gone
        assert cache.exists("key") is False

    def test_multiple_keys_independent_expiration(self) -> None:
        """Test that different keys expire independently."""
        cache = InMemoryCache()

        cache.set("key1", "value1", ttl=timedelta(milliseconds=50))
        cache.set("key2", "value2", ttl=timedelta(milliseconds=150))
        cache.set("key3", "value3")  # No expiration

        # All present initially
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

        # Wait for first to expire
        time.sleep(0.08)

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

        # Wait for second to expire
        time.sleep(0.1)

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"

    def test_set_updates_expiration(self) -> None:
        """Test that setting an existing key updates its expiration."""
        cache = InMemoryCache()

        cache.set("key", "value1", ttl=timedelta(milliseconds=50))
        time.sleep(0.03)

        # Update with new TTL
        cache.set("key", "value2", ttl=timedelta(milliseconds=100))

        time.sleep(0.05)

        # Original TTL would have expired, but new one hasn't
        assert cache.get("key") == "value2"

    def test_complex_objects(self) -> None:
        """Test caching complex objects."""
        cache = InMemoryCache()

        complex_obj = {
            "nested": {"dict": {"with": ["lists", "and", "values"]}},
            "number": 42,
            "none": None,
        }

        cache.set("complex", complex_obj)
        retrieved = cache.get("complex")

        assert retrieved == complex_obj
        assert retrieved is complex_obj  # Same object reference

    def test_thread_safety_basic(self) -> None:
        """Test basic thread safety with concurrent operations."""
        cache = InMemoryCache()
        errors = []

        def writer(start: int) -> None:
            """Write keys in a loop."""
            try:
                for i in range(100):
                    cache.set(f"key_{start}_{i}", f"value_{start}_{i}")
            except Exception as e:
                errors.append(e)

        def reader(start: int) -> None:
            """Read keys in a loop."""
            try:
                for i in range(100):
                    cache.get(f"key_{start}_{i}")
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            threads.append(Thread(target=writer, args=(i,)))
            threads.append(Thread(target=reader, args=(i,)))

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # No errors should occur
        assert len(errors) == 0

    def test_protocol_compliance(self) -> None:
        """Test that InMemoryCache satisfies CacheInterface protocol."""
        cache: CacheInterface = InMemoryCache()

        # Should be able to use all protocol methods
        cache.set("key", "value")
        assert cache.get("key") == "value"
        assert cache.exists("key") is True
        assert cache.delete("key") is True
        cache.clear()

    def test_empty_string_key(self) -> None:
        """Test using empty string as a key."""
        cache = InMemoryCache()
        cache.set("", "empty_key_value")
        assert cache.get("") == "empty_key_value"
        assert cache.exists("") is True

    def test_large_number_of_entries(self) -> None:
        """Test cache with many entries."""
        cache = InMemoryCache()

        # Add 1000 entries
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")

        # Verify all present
        for i in range(1000):
            assert cache.get(f"key_{i}") == f"value_{i}"

        # Clear and verify
        cache.clear()
        assert cache.get("key_500") is None
