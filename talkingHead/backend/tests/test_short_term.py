"""Tests for memory.short_term — thin delegation layer over db_ops."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.asyncio
async def test_remember_message_delegates_to_store_message():
    mock_store = AsyncMock()
    with patch("memory.short_term.db_ops.store_message", mock_store):
        from memory.short_term import remember_message

        await remember_message(
            session_id="sess-1",
            role="user",
            content="hello",
            embedding=[0.1, 0.2],
            token_count=5,
        )

    mock_store.assert_awaited_once_with("sess-1", "user", "hello", [0.1, 0.2], 5)


@pytest.mark.asyncio
async def test_remember_message_passes_none_optionals():
    mock_store = AsyncMock()
    with patch("memory.short_term.db_ops.store_message", mock_store):
        from memory.short_term import remember_message

        await remember_message(
            session_id="sess-2",
            role="assistant",
            content="world",
            embedding=None,
            token_count=None,
        )

    mock_store.assert_awaited_once_with("sess-2", "assistant", "world", None, None)


@pytest.mark.asyncio
async def test_recall_recent_messages_delegates_to_get_recent():
    sentinel = [{"role": "user", "content": "hi"}]

    mock_get = AsyncMock(return_value=sentinel)
    with patch("memory.short_term.db_ops.get_recent_messages", mock_get):
        from memory.short_term import recall_recent_messages

        result = await recall_recent_messages("sess-1", limit=10)

    mock_get.assert_awaited_once_with("sess-1", 10)
    assert result is sentinel


@pytest.mark.asyncio
async def test_recall_recent_messages_uses_default_limit():
    mock_get = AsyncMock(return_value=[])
    with patch("memory.short_term.db_ops.get_recent_messages", mock_get):
        from memory.short_term import recall_recent_messages

        await recall_recent_messages("sess-3")

    mock_get.assert_awaited_once_with("sess-3", 20)
