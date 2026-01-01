"""
API request and response schemas.

These Pydantic models define the contract between the API and clients.
"""

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """
    Standardized error response format.

    Attributes:
        error: Human-readable error message
        code: Machine-readable error code (SNAKE_CASE)
        details: Optional additional context
    """

    error: str = Field(description="Human-readable error message")
    code: str = Field(description="Machine-readable error code")
    details: dict[str, str] | None = Field(
        default=None, description="Optional additional error context"
    )


class HealthCheckResponse(BaseModel):
    """
    Health check endpoint response.

    Attributes:
        status: Current service status
        version: API version
    """

    status: str = Field(description="Service status", examples=["healthy"])
    version: str = Field(description="API version", examples=["0.1.0"])
