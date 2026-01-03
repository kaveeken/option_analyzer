"""
Session management service.

Handles session lifecycle: creation, retrieval, expiration, and cleanup.
"""

import secrets
from pathlib import Path

from ..config import Settings
from ..models.session import SessionState
from ..utils.exceptions import SessionExpiredError


class SessionService:
    """
    In-memory session management service.

    Attributes:
        _sessions: Dictionary mapping session_id to SessionState
        _ttl_seconds: Session time-to-live in seconds
    """

    def __init__(self, ttl_seconds: int) -> None:
        """
        Initialize session service.

        Args:
            ttl_seconds: Session TTL in seconds
        """
        self._sessions: dict[str, SessionState] = {}
        self._ttl_seconds = ttl_seconds

    def _delete_plot_files(self, session: SessionState) -> int:
        """
        Delete all plot files associated with a session.

        Args:
            session: Session whose plot files should be deleted

        Returns:
            Number of files successfully deleted
        """
        deleted_count = 0
        for plot_path in session.plot_files:
            try:
                filepath = Path(plot_path)
                if filepath.exists():
                    filepath.unlink()
                    deleted_count += 1
            except Exception:
                # Ignore errors - file may already be deleted or inaccessible
                pass
        return deleted_count

    def create_session(self) -> SessionState:
        """
        Create a new session with unique ID.

        Returns:
            New SessionState instance

        Note:
            Uses secrets.token_urlsafe for cryptographically secure IDs.
        """
        session_id = secrets.token_urlsafe(32)
        session = SessionState(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> SessionState:
        """
        Get session by ID, updating last_accessed timestamp.

        Args:
            session_id: Session identifier

        Returns:
            SessionState instance

        Raises:
            SessionExpiredError: If session not found or expired
        """
        session = self._sessions.get(session_id)

        if session is None:
            raise SessionExpiredError(f"Session '{session_id}' not found")

        if session.is_expired(self._ttl_seconds):
            # Remove expired session and its plot files
            self._delete_plot_files(session)
            del self._sessions[session_id]
            raise SessionExpiredError(f"Session '{session_id}' expired")

        # Update last accessed time
        session.touch()
        return session

    def delete_session(self, session_id: str) -> None:
        """
        Delete session by ID and clean up its plot files.

        Args:
            session_id: Session identifier

        Note:
            Does not raise error if session doesn't exist.
        """
        session = self._sessions.pop(session_id, None)
        if session is not None:
            self._delete_plot_files(session)

    def cleanup_expired_sessions(self) -> int:
        """
        Remove all expired sessions and their plot files.

        Returns:
            Number of sessions cleaned up

        Note:
            Should be called periodically by background task.
        """
        expired_ids = [
            session_id
            for session_id, session in self._sessions.items()
            if session.is_expired(self._ttl_seconds)
        ]

        files_deleted = 0
        for session_id in expired_ids:
            session = self._sessions[session_id]
            files_deleted += self._delete_plot_files(session)
            del self._sessions[session_id]

        if files_deleted > 0:
            print(f"Deleted {files_deleted} plot files from {len(expired_ids)} expired sessions")

        return len(expired_ids)

    def session_count(self) -> int:
        """
        Get current number of active sessions.

        Returns:
            Number of sessions in memory
        """
        return len(self._sessions)


# Global session service instance
_session_service: SessionService | None = None


def get_session_service(settings: Settings) -> SessionService:
    """
    Get or create global session service instance.

    Args:
        settings: Application settings

    Returns:
        SessionService singleton instance
    """
    global _session_service
    if _session_service is None:
        _session_service = SessionService(ttl_seconds=settings.session_ttl)
    return _session_service
