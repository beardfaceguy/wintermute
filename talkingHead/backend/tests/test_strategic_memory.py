"""Tests for memory.strategic — async bridge to mcp-memory."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# format_memory_context (pure function, no mocking needed)
# ---------------------------------------------------------------------------

from memory.strategic import format_memory_context


class TestFormatMemoryContext:
    def test_empty_list_returns_empty_string(self):
        assert format_memory_context([]) == ""

    def test_none_like_falsy_returns_empty_string(self):
        assert format_memory_context([]) == ""

    def test_single_memory_has_header_and_footer(self):
        memories = [{"similarity": 0.92, "zone": "live", "text": "hello"}]
        result = format_memory_context(memories)
        assert result.startswith("[Relevant Memory]")
        assert "[End Memory]" in result

    def test_includes_similarity_zone_text(self):
        memories = [
            {"similarity": 0.87, "zone": "cold", "text": "the sky is blue"},
        ]
        result = format_memory_context(memories)
        assert "sim=0.87" in result
        assert "cold" in result
        assert "the sky is blue" in result

    def test_multiple_memories_numbered(self):
        memories = [
            {"similarity": 0.9, "zone": "live", "text": "first"},
            {"similarity": 0.8, "zone": "cold", "text": "second"},
            {"similarity": 0.7, "zone": "archive", "text": "third"},
        ]
        result = format_memory_context(memories)
        assert "1." in result
        assert "2." in result
        assert "3." in result

    def test_missing_fields_use_defaults(self):
        memories = [{}]
        result = format_memory_context(memories)
        assert "sim=0" in result
        assert "?" in result


# ---------------------------------------------------------------------------
# search_relevant_memories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_empty_when_mcp_unavailable():
    with patch("memory.strategic._mcp_memory_available", False):
        from memory.strategic import search_relevant_memories

        result = await search_relevant_memories("test query")
        assert result == []


@pytest.mark.asyncio
async def test_search_calls_memory_search_when_available():
    mock_memory_search = MagicMock(return_value=[{"text": "found", "similarity": 0.9}])

    with (
        patch("memory.strategic._mcp_memory_available", True),
        patch("memory.strategic.memory_search", mock_memory_search, create=True),
    ):
        from memory.strategic import search_relevant_memories

        results = await search_relevant_memories("hello world", limit=5, zone="live", min_trust=0.5)

    mock_memory_search.assert_called_once_with(
        query="hello world", limit=5, zone="live", min_trust=0.5
    )
    assert len(results) == 1
    assert results[0]["text"] == "found"


@pytest.mark.asyncio
async def test_search_returns_empty_on_exception():
    mock_memory_search = MagicMock(side_effect=RuntimeError("db down"))

    with (
        patch("memory.strategic._mcp_memory_available", True),
        patch("memory.strategic.memory_search", mock_memory_search, create=True),
    ):
        from memory.strategic import search_relevant_memories

        result = await search_relevant_memories("test")

    assert result == []


# ---------------------------------------------------------------------------
# store_conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_returns_none_when_mcp_unavailable():
    with patch("memory.strategic._mcp_memory_available", False):
        from memory.strategic import store_conversation

        result = await store_conversation("sess-1", "hi", "hello")
        assert result is None


@pytest.mark.asyncio
async def test_store_calls_memory_add_when_available():
    mock_memory_add = MagicMock(return_value={"id": "mem-42"})

    with (
        patch("memory.strategic._mcp_memory_available", True),
        patch("memory.strategic.memory_add", mock_memory_add, create=True),
    ):
        from memory.strategic import store_conversation

        result = await store_conversation(
            "sess-1", "user says", "assistant says", extra_tags={"mood": "happy"}
        )

    mock_memory_add.assert_called_once()
    call_kwargs = mock_memory_add.call_args
    assert "User: user says\nAssistant: assistant says" == call_kwargs.kwargs["text"]
    assert call_kwargs.kwargs["zone"] == "live"
    tags = call_kwargs.kwargs["tags"]
    assert tags["session_id"] == "sess-1"
    assert tags["type"] == "conversation"
    assert tags["source"] == "talkingHead"
    assert tags["mood"] == "happy"
    assert result == {"id": "mem-42"}


@pytest.mark.asyncio
async def test_store_returns_none_on_exception():
    mock_memory_add = MagicMock(side_effect=ConnectionError("lost connection"))

    with (
        patch("memory.strategic._mcp_memory_available", True),
        patch("memory.strategic.memory_add", mock_memory_add, create=True),
    ):
        from memory.strategic import store_conversation

        result = await store_conversation("sess-1", "hi", "bye")

    assert result is None
