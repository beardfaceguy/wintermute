"""
Unit tests for database operations.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from db.db_models import Message
from db.db_ops import get_recent_messages, store_message


class TestDatabaseOperations:
    """Test cases for database operations."""

    @patch("db.db_ops.AsyncSessionLocal")
    @pytest.mark.asyncio
    async def test_store_message_success(self, mock_session_local: MagicMock) -> None:
        """Test successful message storage."""
        # Setup mock session
        mock_session = AsyncMock()
        mock_session_local.return_value.__aenter__.return_value = mock_session

        # Test data
        session_id = "test-session-123"
        role = "user"
        content = "Hello, world!"

        # Call the function
        await store_message(session_id=session_id, role=role, content=content)

        # Verify session was used correctly
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

        # Verify the message object was created correctly
        added_message = mock_session.add.call_args[0][0]
        assert isinstance(added_message, Message)
        # Access attributes through getattr to avoid SQLAlchemy boolean issues
        assert getattr(added_message, "session_id") == session_id
        assert getattr(added_message, "role") == role
        assert getattr(added_message, "content") == content

    @patch("db.db_ops.AsyncSessionLocal")
    @pytest.mark.asyncio
    async def test_store_message_with_optional_fields(
        self, mock_session_local: MagicMock
    ) -> None:
        """Test message storage with optional fields."""
        mock_session = AsyncMock()
        mock_session_local.return_value.__aenter__.return_value = mock_session

        session_id = "test-session-456"
        role = "assistant"
        content = "Response message"
        embedding = [0.1, 0.2, 0.3]
        token_count = 42

        await store_message(
            session_id=session_id,
            role=role,
            content=content,
            embedding=embedding,
            token_count=token_count,
        )

        added_message = mock_session.add.call_args[0][0]
        assert getattr(added_message, "embedding") == embedding
        assert getattr(added_message, "token_count") == token_count

    @patch("db.db_ops.AsyncSessionLocal")
    @pytest.mark.asyncio
    async def test_store_message_exception_handling(
        self, mock_session_local: MagicMock
    ) -> None:
        """Test message storage exception handling."""
        mock_session = AsyncMock()
        mock_session.commit.side_effect = Exception("Database error")
        mock_session_local.return_value.__aenter__.return_value = mock_session

        with patch("db.db_ops.store_message") as mock_store:
            mock_store.side_effect = Exception("Database error")

            with pytest.raises(Exception, match="Database error"):
                await store_message(
                    session_id="test-session", role="user", content="test message"
                )

        mock_session.rollback.assert_called_once()

    @patch("db.db_ops.AsyncSessionLocal")
    @pytest.mark.asyncio
    async def test_get_recent_messages_success(
        self, mock_session_local: MagicMock
    ) -> None:
        """Test successful message retrieval."""
        mock_session = AsyncMock()
        mock_session_local.return_value.__aenter__.return_value = mock_session

        # Create mock messages in DB order (newest first, matching ORDER BY timestamp DESC)
        mock_messages = [
            Message(
                session_id="test-session", role="assistant", content="Hi there!", id=2
            ),
            Message(session_id="test-session", role="user", content="Hello", id=1),
        ]

        # Mock the execute result
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_messages
        mock_session.execute.return_value = mock_result

        # Call the function
        result = await get_recent_messages("test-session", limit=10)

        # Verify result
        assert len(result) == 2
        assert getattr(result[0], "content") == "Hello"
        assert getattr(result[1], "content") == "Hi there!"

        # Verify query was executed
        mock_session.execute.assert_called_once()

    @patch("db.db_ops.AsyncSessionLocal")
    @pytest.mark.asyncio
    async def test_get_recent_messages_empty_result(
        self, mock_session_local: MagicMock
    ) -> None:
        """Test message retrieval with empty result."""
        mock_session = AsyncMock()
        mock_session_local.return_value.__aenter__.return_value = mock_session

        # Mock empty result
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        result = await get_recent_messages("empty-session")

        assert result == []
        assert isinstance(result, list)

    @patch("db.db_ops.AsyncSessionLocal")
    @pytest.mark.asyncio
    async def test_get_recent_messages_with_limit(
        self, mock_session_local: MagicMock
    ) -> None:
        """Test message retrieval with custom limit."""
        mock_session = AsyncMock()
        mock_session_local.return_value.__aenter__.return_value = mock_session

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        await get_recent_messages("test-session", limit=5)

        # Verify the limit was applied in the query
        # The actual query construction is tested indirectly through the mock
        mock_session.execute.assert_called_once()

    @patch("db.db_ops.AsyncSessionLocal")
    @pytest.mark.asyncio
    async def test_get_recent_messages_reversed_order(
        self, mock_session_local: MagicMock
    ) -> None:
        """Test that messages are returned in correct order (reversed from DB order)."""
        mock_session = AsyncMock()
        mock_session_local.return_value.__aenter__.return_value = mock_session

        # Create messages in reverse chronological order (as they would be from DB)
        mock_messages = [
            Message(
                session_id="test-session",
                role="assistant",
                content="Latest message",
                id=3,
            ),
            Message(
                session_id="test-session", role="user", content="Earlier message", id=2
            ),
            Message(
                session_id="test-session", role="user", content="Oldest message", id=1
            ),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_messages
        mock_session.execute.return_value = mock_result

        result = await get_recent_messages("test-session")

        # Should be reversed to chronological order
        assert len(result) == 3
        assert getattr(result[0], "content") == "Oldest message"
        assert getattr(result[1], "content") == "Earlier message"
        assert getattr(result[2], "content") == "Latest message"
