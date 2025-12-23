"""
Client implementations for external services.

This package contains:
- Cache implementations (in-memory, Redis, etc.)
- API clients (IBKR, data providers, etc.)
"""

from .cache import CacheInterface, InMemoryCache

__all__ = ["CacheInterface", "InMemoryCache"]
