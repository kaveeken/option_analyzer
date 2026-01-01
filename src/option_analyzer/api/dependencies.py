"""
FastAPI dependency injection helpers.

Provides reusable dependencies for routes to access configuration,
clients, and services.
"""

from datetime import timedelta
from typing import Annotated

from fastapi import Cookie, Depends

from ..clients.cache import InMemoryCache
from ..clients.ibkr import IBKRClient
from ..config import Settings, get_settings
from ..models.session import SessionState
from ..services.session import SessionService, get_session_service
from ..utils.rate_limiter import RateLimiter

# Global singletons
_cache: InMemoryCache | None = None
_rate_limiter: RateLimiter | None = None


def get_cache() -> InMemoryCache:
    """
    Get or create global cache instance.

    Returns:
        InMemoryCache singleton with 5-minute default TTL
    """
    global _cache
    if _cache is None:
        _cache = InMemoryCache(default_ttl=timedelta(minutes=5))
    return _cache


def get_rate_limiter() -> RateLimiter:
    """
    Get or create global rate limiter instance.

    Returns:
        RateLimiter singleton allowing 50 requests per 60 seconds
    """
    global _rate_limiter
    if _rate_limiter is None:
        # IBKR limits: ~50 requests per minute is a safe default
        _rate_limiter = RateLimiter(max_requests=50, per_seconds=60)
    return _rate_limiter


def get_ibkr_client(
    settings: Annotated[Settings, Depends(get_settings)]
) -> IBKRClient:
    """
    Get IBKR client instance.

    Args:
        settings: Application settings

    Returns:
        Configured IBKRClient instance

    Note:
        Creates a new client instance per request, but reuses
        global cache and rate_limiter singletons.
    """
    return IBKRClient(
        settings=settings,
        cache=get_cache(),
        rate_limiter=get_rate_limiter(),
    )


def get_session_service_dep(
    settings: Annotated[Settings, Depends(get_settings)]
) -> SessionService:
    """
    Get session service instance.

    Args:
        settings: Application settings

    Returns:
        SessionService singleton instance
    """
    return get_session_service(settings)


def get_current_session(
    session_id: Annotated[str | None, Cookie()] = None,
    session_service: Annotated[SessionService, Depends(get_session_service_dep)] = None,
) -> SessionState:
    """
    Get current session from cookie.

    Args:
        session_id: Session ID from cookie
        session_service: Session service instance

    Returns:
        Current SessionState

    Raises:
        SessionExpiredError: If session not found or expired

    Note:
        Use this dependency to require a valid session in an endpoint.
    """
    if session_id is None:
        from ..utils.exceptions import SessionExpiredError
        raise SessionExpiredError("No session ID provided")

    return session_service.get_session(session_id)
