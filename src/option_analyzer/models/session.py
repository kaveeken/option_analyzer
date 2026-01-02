"""
Session state model for tracking user analysis sessions.

Sessions store strategy configurations and historical data for
stateful multi-step analysis workflows.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class SessionState(BaseModel):
    """
    User session state for strategy analysis.

    Attributes:
        session_id: Unique session identifier
        created_at: Session creation timestamp
        last_accessed: Last access timestamp (for TTL tracking)
        data: Session data storage (strategy, historical data, etc.)
    """

    session_id: str = Field(description="Unique session identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Session creation timestamp"
    )
    last_accessed: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Last access timestamp"
    )
    data: dict[str, Any] = Field(
        default_factory=dict, description="Session data storage"
    )

    def is_expired(self, ttl_seconds: int) -> bool:
        """
        Check if session has exceeded TTL.

        Args:
            ttl_seconds: Time-to-live in seconds

        Returns:
            True if session is expired, False otherwise
        """
        age_seconds = (datetime.now(UTC) - self.last_accessed).total_seconds()
        return age_seconds > ttl_seconds

    def touch(self) -> None:
        """Update last_accessed timestamp to current time."""
        self.last_accessed = datetime.now(UTC)
