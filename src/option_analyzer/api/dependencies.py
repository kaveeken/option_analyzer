"""
FastAPI dependency injection helpers.

Provides reusable dependencies for routes to access configuration,
clients, and services.
"""

from typing import Annotated

from fastapi import Depends

from ..clients.ibkr import IBKRClient
from ..config import Settings, get_settings


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
