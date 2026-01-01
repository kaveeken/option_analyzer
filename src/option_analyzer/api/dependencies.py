"""
FastAPI dependency injection helpers.

Provides reusable dependencies for routes to access configuration,
clients, and services.
"""

from typing import Annotated

from fastapi import Cookie, Depends

from ..clients.ibkr import IBKRClient
from ..config import Settings, get_settings
from ..models.session import SessionState
from ..services.session import SessionService, get_session_service


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
        This creates a new client instance per request. For production,
        consider connection pooling or client lifecycle management.
    """
    return IBKRClient(
        base_url=settings.ibkr_base_url,
        timeout=settings.ibkr_timeout,
        verify_ssl=settings.ibkr_verify_ssl,
        max_retries=settings.ibkr_max_retries,
        retry_delay=settings.ibkr_retry_delay,
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
