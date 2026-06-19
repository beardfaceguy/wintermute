"""
Unit tests for WebSocket functionality.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocketDisconnect


class TestConnectionManager:
    """Test cases for WebSocket connection manager."""

    @pytest.fixture(autouse=True)
    def _fresh_manager(self):
        """Reset the singleton manager's connection list before each test."""
        from app.websocket.connection_manager import manager

        manager.active_connections.clear()
        yield
        manager.active_connections.clear()

    @pytest.mark.asyncio
    async def test_connect(self, mock_websocket: MagicMock) -> None:
        """Test WebSocket connection."""
        from app.websocket.connection_manager import manager

        await manager.connect(mock_websocket)

        mock_websocket.accept.assert_called_once()
        assert mock_websocket in manager.active_connections

    @pytest.mark.asyncio
    async def test_disconnect(self, mock_websocket: MagicMock) -> None:
        """Test WebSocket disconnection."""
        from app.websocket.connection_manager import manager

        await manager.connect(mock_websocket)
        assert mock_websocket in manager.active_connections

        manager.disconnect(mock_websocket)
        assert mock_websocket not in manager.active_connections

    @pytest.mark.asyncio
    async def test_send_personal_message(self, mock_websocket: MagicMock) -> None:
        """Test sending personal message to specific WebSocket."""
        from app.websocket.connection_manager import manager

        await manager.connect(mock_websocket)
        await manager.send_personal_message("test message", mock_websocket)

        mock_websocket.send_text.assert_called_once_with("test message")

    @pytest.mark.asyncio
    async def test_broadcast(self, mock_websocket: MagicMock) -> None:
        """Test broadcasting message to all connections."""
        from app.websocket.connection_manager import manager

        mock_websocket2 = MagicMock()
        mock_websocket2.accept = AsyncMock()
        mock_websocket2.send_text = AsyncMock()

        await manager.connect(mock_websocket)
        await manager.connect(mock_websocket2)

        await manager.broadcast("broadcast message")

        mock_websocket.send_text.assert_called_once_with("broadcast message")
        mock_websocket2.send_text.assert_called_once_with("broadcast message")


class TestChatWebSocket:
    """Test cases for chat WebSocket endpoint."""

    @pytest.mark.asyncio
    @patch("app.websocket.chat_ws.store_conversation", new_callable=AsyncMock)
    @patch(
        "app.websocket.chat_ws.search_relevant_memories", new_callable=AsyncMock, return_value=[]
    )
    @patch("app.websocket.chat_ws.store_message", new_callable=AsyncMock)
    @patch("app.websocket.chat_ws.get_recent_messages", new_callable=AsyncMock, return_value=[])
    @patch("app.websocket.chat_ws.chat_processor")
    async def test_chat_endpoint_success(
        self,
        mock_chat_processor: MagicMock,
        mock_get_messages: AsyncMock,
        mock_store_message: AsyncMock,
        mock_search_memories: AsyncMock,
        mock_store_conv: AsyncMock,
        mock_websocket: MagicMock,
        sample_message_data: dict[str, Any],
    ) -> None:
        """Test successful chat message processing."""
        from app.websocket.chat_ws import chat_endpoint

        mock_chat_processor.stream_response = AsyncMock(return_value="Assistant response")
        mock_websocket.receive_text = AsyncMock(
            side_effect=[json.dumps(sample_message_data), WebSocketDisconnect()]
        )

        with patch("app.websocket.chat_ws.manager") as mock_manager:
            mock_manager.connect = AsyncMock()
            mock_manager.disconnect = MagicMock()

            await chat_endpoint(mock_websocket)

            mock_manager.connect.assert_called_once_with(mock_websocket)
            assert mock_store_message.call_count >= 2

    @pytest.mark.asyncio
    async def test_chat_endpoint_empty_message(self, mock_websocket: MagicMock) -> None:
        """Test chat endpoint with empty message."""
        from app.websocket.chat_ws import chat_endpoint

        mock_websocket.receive_text = AsyncMock(
            side_effect=[json.dumps({"message": ""}), WebSocketDisconnect()]
        )

        with patch("app.websocket.chat_ws.manager") as mock_manager:
            mock_manager.connect = AsyncMock()
            mock_manager.disconnect = MagicMock()

            await chat_endpoint(mock_websocket)

            mock_websocket.send_text.assert_called_with("Error: Empty message")

    @pytest.mark.asyncio
    async def test_chat_endpoint_invalid_json(self, mock_websocket: MagicMock) -> None:
        """Test chat endpoint with invalid JSON."""
        from app.websocket.chat_ws import chat_endpoint

        mock_websocket.receive_text = AsyncMock(side_effect=["invalid json", WebSocketDisconnect()])

        with patch("app.websocket.chat_ws.manager") as mock_manager:
            mock_manager.connect = AsyncMock()
            mock_manager.disconnect = MagicMock()

            await chat_endpoint(mock_websocket)

            mock_websocket.send_text.assert_called_with("Error: Invalid JSON format")

    @pytest.mark.asyncio
    async def test_chat_endpoint_exception_handling(self, mock_websocket: MagicMock) -> None:
        """Test chat endpoint exception handling."""
        from app.websocket.chat_ws import chat_endpoint

        mock_websocket.receive_text = AsyncMock(side_effect=Exception("Test exception"))

        with patch("app.websocket.chat_ws.manager") as mock_manager:
            mock_manager.connect = AsyncMock()
            mock_manager.disconnect = MagicMock()

            await chat_endpoint(mock_websocket)

            mock_websocket.send_text.assert_called_with("Error: Test exception")

    @pytest.mark.asyncio
    async def test_chat_endpoint_websocket_disconnect(self, mock_websocket: MagicMock) -> None:
        """Test WebSocket disconnect handling."""
        from app.websocket.chat_ws import chat_endpoint

        mock_websocket.receive_text = AsyncMock(side_effect=WebSocketDisconnect())

        with patch("app.websocket.chat_ws.manager") as mock_manager:
            mock_manager.connect = AsyncMock()
            mock_manager.disconnect = MagicMock()

            await chat_endpoint(mock_websocket)

            mock_manager.disconnect.assert_called_once_with(mock_websocket)
