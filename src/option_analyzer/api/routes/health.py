"""
Health check endpoint.

Provides a simple endpoint to verify the API is running.
"""

from fastapi import APIRouter

from ..schemas import HealthCheckResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint.

    Returns:
        Service status and version information

    Example:
        GET /health
        Response: {"status": "healthy", "version": "0.1.0"}
    """
    return HealthCheckResponse(status="healthy", version="0.1.0")
