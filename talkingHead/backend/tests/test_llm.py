"""
Unit tests for LLM chat processor.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.chat.llm import ChatProcessor


class TestChatProcessor:
    """Test cases for ChatProcessor."""

    def test_chat_processor_initialization(self) -> None:
        """Test ChatProcessor initialization."""
        processor = ChatProcessor(
            model_url="http://localhost:8000", model_name="test-model"
        )

        assert processor.model_url == "http://localhost:8000"
        assert processor.model_name == "test-model"

    def test_chat_processor_default_config(self) -> None:
        """Test ChatProcessor with default configuration."""
        # Since the module-level variables are set at import time,
        # we'll test that the processor uses the current values
        processor = ChatProcessor()

        # These should match the actual loaded config
        assert processor.model_url is not None
        assert processor.model_name is not None
        assert isinstance(processor.model_url, str)
        assert isinstance(processor.model_name, str)

    @patch("app.chat.llm.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_stream_response_success(self, mock_client: MagicMock) -> None:
        """Test successful streaming response."""
        processor = ChatProcessor("http://test:8000", "test-model")

        # Mock the async context manager
        mock_client_instance = MagicMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        mock_client.return_value.__aexit__.return_value = None

        # Mock the streaming response
        mock_response = MagicMock()
        mock_response.aiter_lines.return_value = [
            'data: {"choices": [{"text": "Hello"}]}',
            'data: {"choices": [{"text": " world"}]}',
            "data: [DONE]",
        ]
        mock_client_instance.stream.return_value.__aenter__.return_value = mock_response
        mock_client_instance.stream.return_value.__aexit__.return_value = None

        # Mock the callback
        mock_callback = AsyncMock()

        # Call the method
        result = await processor.stream_response("test prompt", mock_callback)

        # Verify result
        assert result == "Hello world"

        # Verify callback was called with each token
        assert mock_callback.call_count == 2
        mock_callback.assert_any_call("Hello")
        mock_callback.assert_any_call(" world")

        # Verify the request was made correctly
        mock_client_instance.stream.assert_called_once()
        call_args = mock_client_instance.stream.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "http://test:8000"

        # Verify the JSON payload
        json_data = call_args[1]["json"]
        assert json_data["model"] == "test-model"
        assert json_data["prompt"] == "test prompt"
        assert json_data["stream"] is True
        assert json_data["max_tokens"] == 512
        assert json_data["temperature"] == 0.95
        assert json_data["top_p"] == 0.95

    @patch("app.chat.llm.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_stream_response_empty_response(self, mock_client: MagicMock) -> None:
        """Test streaming response with empty response."""
        processor = ChatProcessor("http://test:8000", "test-model")

        mock_client_instance = MagicMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        mock_client.return_value.__aexit__.return_value = None

        # Mock empty response
        mock_response = MagicMock()
        mock_response.aiter_lines.return_value = ["data: [DONE]"]
        mock_client_instance.stream.return_value.__aenter__.return_value = mock_response
        mock_client_instance.stream.return_value.__aexit__.return_value = None

        mock_callback = AsyncMock()

        result = await processor.stream_response("test prompt", mock_callback)

        assert result == ""
        mock_callback.assert_not_called()

    @patch("app.chat.llm.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_stream_response_parse_error(self, mock_client: MagicMock) -> None:
        """Test streaming response with JSON parse error."""
        processor = ChatProcessor("http://test:8000", "test-model")

        mock_client_instance = MagicMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        mock_client.return_value.__aexit__.return_value = None

        # Mock response with invalid JSON
        mock_response = MagicMock()
        mock_response.aiter_lines.return_value = [
            "data: invalid json",
            'data: {"choices": [{"text": "Valid"}]}',
            "data: [DONE]",
        ]
        mock_client_instance.stream.return_value.__aenter__.return_value = mock_response
        mock_client_instance.stream.return_value.__aexit__.return_value = None

        mock_callback = AsyncMock()

        result = await processor.stream_response("test prompt", mock_callback)

        # Should still process valid JSON
        assert result == "Valid"
        mock_callback.assert_called_once_with("Valid")

    @patch("app.chat.llm.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_stream_response_missing_text(self, mock_client: MagicMock) -> None:
        """Test streaming response with missing text field."""
        processor = ChatProcessor("http://test:8000", "test-model")

        mock_client_instance = MagicMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        mock_client.return_value.__aexit__.return_value = None

        # Mock response with missing text
        mock_response = MagicMock()
        mock_response.aiter_lines.return_value = [
            'data: {"choices": [{}]}',
            'data: {"choices": [{"text": "Valid"}]}',
            "data: [DONE]",
        ]
        mock_client_instance.stream.return_value.__aenter__.return_value = mock_response
        mock_client_instance.stream.return_value.__aexit__.return_value = None

        mock_callback = AsyncMock()

        result = await processor.stream_response("test prompt", mock_callback)

        # Should only process valid responses
        assert result == "Valid"
        mock_callback.assert_called_once_with("Valid")

    @patch("app.chat.llm.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_stream_response_skip_empty_lines(
        self, mock_client: MagicMock
    ) -> None:
        """Test that empty lines and control lines are skipped."""
        processor = ChatProcessor("http://test:8000", "test-model")

        mock_client_instance = MagicMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        mock_client.return_value.__aexit__.return_value = None

        # Mock response with empty lines and control lines
        mock_response = MagicMock()
        mock_response.aiter_lines.return_value = [
            "",
            ":",
            'data: {"choices": [{"text": "Hello"}]}',
            "data: [DONE]",
        ]
        mock_client_instance.stream.return_value.__aenter__.return_value = mock_response
        mock_client_instance.stream.return_value.__aexit__.return_value = None

        mock_callback = AsyncMock()

        result = await processor.stream_response("test prompt", mock_callback)

        assert result == "Hello"
        mock_callback.assert_called_once_with("Hello")

    @patch("app.chat.llm.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_stream_response_exception_handling(
        self, mock_client: MagicMock
    ) -> None:
        """Test exception handling during streaming."""
        processor = ChatProcessor("http://test:8000", "test-model")

        mock_client_instance = MagicMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        mock_client.return_value.__aexit__.return_value = None

        # Mock response that raises an exception
        mock_response = MagicMock()
        mock_response.aiter_lines.side_effect = Exception("Network error")
        mock_client_instance.stream.return_value.__aenter__.return_value = mock_response
        mock_client_instance.stream.return_value.__aexit__.return_value = None

        mock_callback = AsyncMock()

        with pytest.raises(Exception, match="Network error"):
            await processor.stream_response("test prompt", mock_callback)
