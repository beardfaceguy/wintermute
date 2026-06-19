"""
Extended tests for chat WebSocket endpoint — covers DAG deep search integration,
message size limits, and connection capacity rejection.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocketDisconnect


def _make_mock_ws():
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    ws.close = AsyncMock()
    ws.receive_text = AsyncMock(side_effect=WebSocketDisconnect())
    return ws


class TestDeepSearchIntegration:
    """Verify chat_ws passes deep=True to search_relevant_memories."""

    @pytest.mark.asyncio
    @patch("app.websocket.chat_ws.store_conversation", new_callable=AsyncMock)
    @patch(
        "app.websocket.chat_ws.search_relevant_memories", new_callable=AsyncMock, return_value=[]
    )
    @patch("app.websocket.chat_ws.store_message", new_callable=AsyncMock)
    @patch("app.websocket.chat_ws.get_recent_messages", new_callable=AsyncMock, return_value=[])
    @patch("app.websocket.chat_ws.chat_processor")
    async def test_search_called_with_deep_true(
        self,
        mock_processor,
        mock_get_messages,
        mock_store_message,
        mock_search_memories,
        mock_store_conv,
    ):
        from app.websocket.chat_ws import chat_endpoint

        mock_processor.stream_response = AsyncMock(return_value="response")
        ws = _make_mock_ws()
        ws.receive_text = AsyncMock(
            side_effect=[
                json.dumps({"message": "complex multi-hop question"}),
                WebSocketDisconnect(),
            ]
        )

        with patch("app.websocket.chat_ws.manager") as mock_mgr:
            mock_mgr.connect = AsyncMock(return_value=True)
            mock_mgr.disconnect = MagicMock()
            await chat_endpoint(ws)

        mock_search_memories.assert_called_once()
        call_kwargs = mock_search_memories.call_args
        assert call_kwargs.kwargs.get("deep") is True or (
            len(call_kwargs.args) > 0 and call_kwargs[1].get("deep") is True
        )

    @pytest.mark.asyncio
    @patch("app.websocket.chat_ws.store_conversation", new_callable=AsyncMock)
    @patch("app.websocket.chat_ws.search_relevant_memories", new_callable=AsyncMock)
    @patch("app.websocket.chat_ws.store_message", new_callable=AsyncMock)
    @patch("app.websocket.chat_ws.get_recent_messages", new_callable=AsyncMock, return_value=[])
    @patch("app.websocket.chat_ws.chat_processor")
    async def test_memory_context_injected_into_prompt(
        self,
        mock_processor,
        mock_get_messages,
        mock_store_message,
        mock_search_memories,
        mock_store_conv,
    ):
        """When memories are found, they should appear in the formatted prompt."""
        from app.websocket.chat_ws import chat_endpoint

        mock_search_memories.return_value = [
            {"similarity": 0.95, "zone": "live", "text": "OOM caused by batch size 64"},
        ]
        mock_processor.stream_response = AsyncMock(return_value="ok")
        ws = _make_mock_ws()
        ws.receive_text = AsyncMock(
            side_effect=[json.dumps({"message": "what caused OOM?"}), WebSocketDisconnect()]
        )

        with patch("app.websocket.chat_ws.manager") as mock_mgr:
            mock_mgr.connect = AsyncMock(return_value=True)
            mock_mgr.disconnect = MagicMock()
            await chat_endpoint(ws)

        prompt_arg = mock_processor.stream_response.call_args[0][0]
        assert "[Relevant Memory]" in prompt_arg
        assert "OOM caused by batch size 64" in prompt_arg


class TestMessageSizeLimit:
    @pytest.mark.asyncio
    async def test_oversized_message_rejected(self):
        from app.websocket.chat_ws import chat_endpoint

        ws = _make_mock_ws()
        huge_payload = "x" * (17 * 1024)
        ws.receive_text = AsyncMock(side_effect=[huge_payload, WebSocketDisconnect()])

        with patch("app.websocket.chat_ws.manager") as mock_mgr:
            mock_mgr.connect = AsyncMock(return_value=True)
            mock_mgr.disconnect = MagicMock()
            await chat_endpoint(ws)

        ws.send_text.assert_called_with("Error: Message too large")

    @pytest.mark.asyncio
    async def test_normal_size_message_accepted(self):
        from app.websocket.chat_ws import chat_endpoint

        ws = _make_mock_ws()
        ws.receive_text = AsyncMock(
            side_effect=[json.dumps({"message": "hello"}), WebSocketDisconnect()]
        )

        with (
            patch("app.websocket.chat_ws.manager") as mock_mgr,
            patch("app.websocket.chat_ws.store_message", new_callable=AsyncMock),
            patch(
                "app.websocket.chat_ws.search_relevant_memories",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.websocket.chat_ws.get_recent_messages", new_callable=AsyncMock, return_value=[]
            ),
            patch("app.websocket.chat_ws.chat_processor") as mock_proc,
            patch("app.websocket.chat_ws.store_conversation", new_callable=AsyncMock),
        ):
            mock_mgr.connect = AsyncMock(return_value=True)
            mock_mgr.disconnect = MagicMock()
            mock_proc.stream_response = AsyncMock(return_value="hi")
            await chat_endpoint(ws)

        send_calls = [c.args[0] for c in ws.send_text.call_args_list]
        assert "Error: Message too large" not in send_calls


class TestEndOfStreamSentinel:
    """Verify the assistant-message-complete sentinel is sent after each turn.

    The frontend's TTS hook listens for this exact string to know when to
    synthesize, so it must be emitted on every assistant response.
    """

    @pytest.mark.asyncio
    async def test_sentinel_sent_after_response(self):
        from app.websocket.chat_ws import END_OF_STREAM_SENTINEL, chat_endpoint

        ws = _make_mock_ws()
        ws.receive_text = AsyncMock(
            side_effect=[json.dumps({"message": "hi"}), WebSocketDisconnect()]
        )

        with (
            patch("app.websocket.chat_ws.manager") as mock_mgr,
            patch("app.websocket.chat_ws.store_message", new_callable=AsyncMock),
            patch(
                "app.websocket.chat_ws.search_relevant_memories",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.websocket.chat_ws.get_recent_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("app.websocket.chat_ws.chat_processor") as mock_proc,
            patch("app.websocket.chat_ws.store_conversation", new_callable=AsyncMock),
        ):
            mock_mgr.connect = AsyncMock(return_value=True)
            mock_mgr.disconnect = MagicMock()
            mock_proc.stream_response = AsyncMock(return_value="response text")
            await chat_endpoint(ws)

        send_calls = [c.args[0] for c in ws.send_text.call_args_list]
        assert END_OF_STREAM_SENTINEL in send_calls
        # Sentinel should be the last thing sent for this turn.
        assert send_calls[-1] == END_OF_STREAM_SENTINEL


class TestConnectionCapacity:
    @pytest.mark.asyncio
    async def test_connect_rejected_at_capacity(self):
        from app.websocket.connection_manager import ConnectionManager

        mgr = ConnectionManager()
        # Fill to capacity
        for _ in range(int(__import__("os").getenv("WS_MAX_CONNECTIONS", "200"))):
            ws = _make_mock_ws()
            result = await mgr.connect(ws)
            assert result is True

        # Next connection should be rejected
        overflow_ws = _make_mock_ws()
        result = await mgr.connect(overflow_ws)
        assert result is False
        overflow_ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_endpoint_returns_on_rejected_connect(self):
        """If manager.connect returns False, chat_endpoint should return immediately."""
        from app.websocket.chat_ws import chat_endpoint

        ws = _make_mock_ws()

        with patch("app.websocket.chat_ws.manager") as mock_mgr:
            mock_mgr.connect = AsyncMock(return_value=False)
            mock_mgr.disconnect = MagicMock()
            await chat_endpoint(ws)

        # Should not have tried to receive any messages
        ws.receive_text.assert_not_called()


class TestDAGSearchFallback:
    """Verify that DAG search failures fall back gracefully."""

    @pytest.mark.asyncio
    async def test_dag_failure_falls_back_to_direct(self):
        mock_memory_search = MagicMock(return_value=[{"text": "direct result", "similarity": 0.8}])
        mock_dag = AsyncMock(side_effect=RuntimeError("LLM timeout"))

        with (
            patch("memory.strategic._mcp_memory_available", True),
            patch("memory.strategic._dag_retrieval_available", True),
            patch("memory.strategic.dag_search", mock_dag),
            patch("memory.strategic.memory_search", mock_memory_search, create=True),
        ):
            from memory.strategic import search_relevant_memories

            results = await search_relevant_memories("test query", deep=True)

        assert len(results) == 1
        assert results[0]["text"] == "direct result"
        mock_dag.assert_awaited_once()
        mock_memory_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_dag_unavailable_uses_direct(self):
        mock_memory_search = MagicMock(return_value=[{"text": "found"}])

        with (
            patch("memory.strategic._mcp_memory_available", True),
            patch("memory.strategic._dag_retrieval_available", False),
            patch("memory.strategic.memory_search", mock_memory_search, create=True),
        ):
            from memory.strategic import search_relevant_memories

            results = await search_relevant_memories("test", deep=True)

        assert len(results) == 1
        mock_memory_search.assert_called_once()
