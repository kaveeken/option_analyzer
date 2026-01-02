"""
Unit tests for session service.

Tests cover:
- Session creation with unique ID generation
- Session retrieval and timestamp updates
- Session expiration logic
- Expired session cleanup
- Session not found error handling
- Session data storage operations
"""

import time
from datetime import UTC, datetime, timedelta

import pytest

from option_analyzer.models.session import SessionState
from option_analyzer.services.session import SessionService, get_session_service
from option_analyzer.utils.exceptions import SessionExpiredError
from option_analyzer.config import Settings


class TestSessionService:
    """Test SessionService class."""

    def test_create_session(self) -> None:
        """Test session creation with unique ID generation."""
        service = SessionService(ttl_seconds=3600)
        session = service.create_session()

        assert session.session_id is not None
        assert len(session.session_id) > 0
        assert session.session_id in service._sessions
        assert service.session_count() == 1

    def test_create_session_generates_unique_ids(self) -> None:
        """Test that multiple sessions get unique IDs."""
        service = SessionService(ttl_seconds=3600)
        session1 = service.create_session()
        session2 = service.create_session()

        assert session1.session_id != session2.session_id
        assert service.session_count() == 2

    def test_get_session_updates_timestamp(self) -> None:
        """Test that get_session calls touch() to update last_accessed."""
        service = SessionService(ttl_seconds=3600)
        session = service.create_session()
        original_timestamp = session.last_accessed

        # Wait a tiny bit to ensure timestamp would change
        time.sleep(0.01)

        # Get session should update timestamp
        retrieved = service.get_session(session.session_id)

        assert retrieved.last_accessed > original_timestamp

    def test_session_expiration(self) -> None:
        """Test session expiration logic using is_expired()."""
        service = SessionService(ttl_seconds=1)  # 1 second TTL
        session = service.create_session()

        # Session should not be expired immediately
        assert not session.is_expired(1)

        # Wait for expiration
        time.sleep(1.1)

        # Session should now be expired
        assert session.is_expired(1)

    def test_get_session_raises_for_expired(self) -> None:
        """Test that get_session raises SessionExpiredError for expired sessions."""
        service = SessionService(ttl_seconds=1)  # 1 second TTL
        session = service.create_session()
        session_id = session.session_id

        # Wait for expiration
        time.sleep(1.1)

        # Should raise SessionExpiredError and remove from storage
        with pytest.raises(SessionExpiredError, match="expired"):
            service.get_session(session_id)

        # Session should be deleted from storage
        assert service.session_count() == 0

    def test_cleanup_expired_sessions(self) -> None:
        """Test cleanup_expired_sessions returns correct count."""
        service = SessionService(ttl_seconds=1)  # 1 second TTL

        # Create 3 sessions
        session1 = service.create_session()
        session2 = service.create_session()
        session3 = service.create_session()

        assert service.session_count() == 3

        # Wait for expiration
        time.sleep(1.1)

        # Cleanup should remove all 3 expired sessions
        count = service.cleanup_expired_sessions()
        assert count == 3
        assert service.session_count() == 0

    def test_cleanup_expired_sessions_partial(self) -> None:
        """Test cleanup only removes expired sessions."""
        service = SessionService(ttl_seconds=2)  # 2 second TTL

        # Create first session
        session1 = service.create_session()

        # Wait 1 second
        time.sleep(1.1)

        # Create second session (still fresh)
        session2 = service.create_session()

        # Wait another 1 second (session1 expired, session2 still valid)
        time.sleep(1.1)

        # Cleanup should remove only session1
        count = service.cleanup_expired_sessions()
        assert count == 1
        assert service.session_count() == 1
        assert session2.session_id in service._sessions

    def test_session_not_found(self) -> None:
        """Test that SessionExpiredError is raised when session not found."""
        service = SessionService(ttl_seconds=3600)

        with pytest.raises(SessionExpiredError, match="not found"):
            service.get_session("nonexistent-session-id")

    def test_session_data_storage(self) -> None:
        """Test session data dict operations."""
        service = SessionService(ttl_seconds=3600)
        session = service.create_session()

        # Initially empty
        assert len(session.data) == 0

        # Store data
        session.data["strategy"] = "covered_call"
        session.data["stock_symbol"] = "AAPL"
        session.data["count"] = 42

        # Retrieve and verify
        retrieved = service.get_session(session.session_id)
        assert retrieved.data["strategy"] == "covered_call"
        assert retrieved.data["stock_symbol"] == "AAPL"
        assert retrieved.data["count"] == 42
        assert len(retrieved.data) == 3

    def test_delete_session(self) -> None:
        """Test session deletion."""
        service = SessionService(ttl_seconds=3600)
        session = service.create_session()
        session_id = session.session_id

        assert service.session_count() == 1

        service.delete_session(session_id)

        assert service.session_count() == 0
        with pytest.raises(SessionExpiredError):
            service.get_session(session_id)

    def test_delete_session_nonexistent(self) -> None:
        """Test that delete_session doesn't raise error for nonexistent session."""
        service = SessionService(ttl_seconds=3600)

        # Should not raise
        service.delete_session("nonexistent-id")

    def test_cleanup_expired_sessions_no_expired(self) -> None:
        """Test cleanup when no sessions are expired."""
        service = SessionService(ttl_seconds=3600)
        service.create_session()
        service.create_session()

        assert service.session_count() == 2

        count = service.cleanup_expired_sessions()
        assert count == 0
        assert service.session_count() == 2


class TestGetSessionService:
    """Test get_session_service singleton factory."""

    def test_get_session_service_creates_singleton(self) -> None:
        """Test that get_session_service returns same instance."""
        # Reset global
        import option_analyzer.services.session as session_module
        session_module._session_service = None

        settings = Settings(session_ttl=3600)
        service1 = get_session_service(settings)
        service2 = get_session_service(settings)

        assert service1 is service2

    def test_get_session_service_uses_settings(self) -> None:
        """Test that get_session_service uses settings TTL."""
        # Reset global
        import option_analyzer.services.session as session_module
        session_module._session_service = None

        settings = Settings(session_ttl=7200)
        service = get_session_service(settings)

        assert service._ttl_seconds == 7200


class TestSessionState:
    """Test SessionState model."""

    def test_session_state_creation(self) -> None:
        """Test basic session state creation."""
        session = SessionState(session_id="test-123")

        assert session.session_id == "test-123"
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_accessed, datetime)
        assert session.data == {}

    def test_session_state_touch(self) -> None:
        """Test touch() updates last_accessed."""
        session = SessionState(session_id="test-123")
        original = session.last_accessed

        time.sleep(0.01)
        session.touch()

        assert session.last_accessed > original

    def test_session_state_is_expired(self) -> None:
        """Test is_expired() logic."""
        session = SessionState(session_id="test-123")

        # Fresh session should not be expired
        assert not session.is_expired(3600)

        # Simulate old last_accessed by modifying it
        session.last_accessed = datetime.now(UTC) - timedelta(seconds=7200)

        # Should be expired with 3600 second TTL
        assert session.is_expired(3600)

    def test_session_state_is_not_expired_edge_case(self) -> None:
        """Test is_expired() edge case at exact TTL boundary."""
        session = SessionState(session_id="test-123")

        # Set last_accessed to just under TTL seconds ago
        session.last_accessed = datetime.now(UTC) - timedelta(seconds=3599)

        # Should not be expired
        assert not session.is_expired(3600)

        # But significantly past TTL should be expired
        session.last_accessed = datetime.now(UTC) - timedelta(seconds=3700)
        assert session.is_expired(3600)

    def test_session_state_data_storage(self) -> None:
        """Test session data can store various types."""
        session = SessionState(session_id="test-123")

        session.data["string"] = "value"
        session.data["number"] = 42
        session.data["list"] = [1, 2, 3]
        session.data["dict"] = {"nested": "data"}

        assert session.data["string"] == "value"
        assert session.data["number"] == 42
        assert session.data["list"] == [1, 2, 3]
        assert session.data["dict"] == {"nested": "data"}
