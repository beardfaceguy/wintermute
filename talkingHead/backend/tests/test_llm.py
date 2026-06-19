"""
Unit tests for LLM chat processor.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.chat.llm import ChatProcessor


async def _async_iter(items):
    """Helper to turn a list into an async iterator for mocking aiter_lines()."""
    for item in items:
        yield item


class TestChatProcessor:
    """Test cases for ChatProcessor."""

    def test_chat_processor_initialization(self) -> None:
        """Test ChatProcessor initialization."""
        processor = ChatProcessor(model_url="http://localhost:8000", model_name="test-model")

        assert processor.model_url == "http://localhost:8000"
        assert processor.model_name == "test-model"

    def test_chat_processor_default_config(self) -> None:
        """Test ChatProcessor with default configuration."""
        processor = ChatProcessor()

        assert processor.model_url is not None
        assert processor.model_name is not None
        assert isinstance(processor.model_url, str)
        assert isinstance(processor.model_name, str)

    @patch("app.chat.llm.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_stream_response_success(self, mock_client: MagicMock) -> None:
        """Test successful streaming response (chat completions format)."""
        processor = ChatProcessor("http://test:8000/v1/chat/completions", "test-model")

        mock_client_instance = MagicMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        mock_client.return_value.__aexit__.return_value = None

        mock_response = MagicMock()
        mock_response.aiter_lines.return_value = _async_iter(
            [
                'data: {"choices": [{"delta": {"content": "Hello"}}]}',
                'data: {"choices": [{"delta": {"content": " world"}}]}',
                "data: [DONE]",
            ]
        )
        mock_client_instance.stream.return_value.__aenter__.return_value = mock_response
        mock_client_instance.stream.return_value.__aexit__.return_value = None

        mock_callback = AsyncMock()

        result = await processor.stream_response("user: test prompt\nassistant:", mock_callback)

        assert result == "Hello world"

        assert mock_callback.call_count == 2
        mock_callback.assert_any_call("Hello")
        mock_callback.assert_any_call(" world")

        mock_client_instance.stream.assert_called_once()
        call_args = mock_client_instance.stream.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "http://test:8000/v1/chat/completions"

        json_data = call_args[1]["json"]
        assert json_data["model"] == "test-model"
        assert "messages" in json_data
        assert json_data["stream"] is True
        assert json_data["max_tokens"] == 512
        assert json_data["temperature"] == 0.7
        assert json_data["top_p"] == 0.9

    @patch("app.chat.llm.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_stream_response_empty_response(self, mock_client: MagicMock) -> None:
        """Test streaming response with empty response."""
        processor = ChatProcessor("http://test:8000/v1/chat/completions", "test-model")

        mock_client_instance = MagicMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        mock_client.return_value.__aexit__.return_value = None

        mock_response = MagicMock()
        mock_response.aiter_lines.return_value = _async_iter(["data: [DONE]"])
        mock_client_instance.stream.return_value.__aenter__.return_value = mock_response
        mock_client_instance.stream.return_value.__aexit__.return_value = None

        mock_callback = AsyncMock()

        result = await processor.stream_response("user: test\nassistant:", mock_callback)

        assert result == ""
        mock_callback.assert_not_called()

    @patch("app.chat.llm.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_stream_response_parse_error(self, mock_client: MagicMock) -> None:
        """Test streaming response with JSON parse error."""
        processor = ChatProcessor("http://test:8000/v1/chat/completions", "test-model")

        mock_client_instance = MagicMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        mock_client.return_value.__aexit__.return_value = None

        mock_response = MagicMock()
        mock_response.aiter_lines.return_value = _async_iter(
            [
                "data: invalid json",
                'data: {"choices": [{"delta": {"content": "Valid"}}]}',
                "data: [DONE]",
            ]
        )
        mock_client_instance.stream.return_value.__aenter__.return_value = mock_response
        mock_client_instance.stream.return_value.__aexit__.return_value = None

        mock_callback = AsyncMock()

        result = await processor.stream_response("user: test\nassistant:", mock_callback)

        assert result == "Valid"
        mock_callback.assert_called_once_with("Valid")

    @patch("app.chat.llm.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_stream_response_missing_delta(self, mock_client: MagicMock) -> None:
        """Test streaming response with missing delta/content field."""
        processor = ChatProcessor("http://test:8000/v1/chat/completions", "test-model")

        mock_client_instance = MagicMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        mock_client.return_value.__aexit__.return_value = None

        mock_response = MagicMock()
        mock_response.aiter_lines.return_value = _async_iter(
            [
                'data: {"choices": [{"delta": {}}]}',
                'data: {"choices": [{"delta": {"content": "Valid"}}]}',
                "data: [DONE]",
            ]
        )
        mock_client_instance.stream.return_value.__aenter__.return_value = mock_response
        mock_client_instance.stream.return_value.__aexit__.return_value = None

        mock_callback = AsyncMock()

        result = await processor.stream_response("user: test\nassistant:", mock_callback)

        assert result == "Valid"
        mock_callback.assert_called_once_with("Valid")

    @patch("app.chat.llm.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_stream_response_skip_empty_lines(self, mock_client: MagicMock) -> None:
        """Test that empty lines and control lines are skipped."""
        processor = ChatProcessor("http://test:8000/v1/chat/completions", "test-model")

        mock_client_instance = MagicMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        mock_client.return_value.__aexit__.return_value = None

        mock_response = MagicMock()
        mock_response.aiter_lines.return_value = _async_iter(
            [
                "",
                ":",
                'data: {"choices": [{"delta": {"content": "Hello"}}]}',
                "data: [DONE]",
            ]
        )
        mock_client_instance.stream.return_value.__aenter__.return_value = mock_response
        mock_client_instance.stream.return_value.__aexit__.return_value = None

        mock_callback = AsyncMock()

        result = await processor.stream_response("user: test\nassistant:", mock_callback)

        assert result == "Hello"
        mock_callback.assert_called_once_with("Hello")

    @patch("app.chat.llm.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_stream_response_exception_handling(self, mock_client: MagicMock) -> None:
        """Test exception handling during streaming."""
        processor = ChatProcessor("http://test:8000/v1/chat/completions", "test-model")

        mock_client_instance = MagicMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        mock_client.return_value.__aexit__.return_value = None

        mock_response = MagicMock()
        mock_response.aiter_lines.side_effect = Exception("Network error")
        mock_client_instance.stream.return_value.__aenter__.return_value = mock_response
        mock_client_instance.stream.return_value.__aexit__.return_value = None

        mock_callback = AsyncMock()

        with pytest.raises(Exception, match="Network error"):
            await processor.stream_response("user: test\nassistant:", mock_callback)

    @pytest.mark.asyncio
    async def test_no_model_url_returns_error(self) -> None:
        """Test that a missing model URL returns an error message."""
        processor = ChatProcessor.__new__(ChatProcessor)
        processor.model_url = ""
        processor.model_name = ""

        mock_callback = AsyncMock()
        result = await processor.stream_response("test", mock_callback)
        assert "not configured" in result
        mock_callback.assert_called_once()


class TestBuildMessages:
    """Test cases for _build_messages prompt parsing."""

    def test_simple_user_prompt(self) -> None:
        processor = ChatProcessor("http://test:8000", "test-model")
        messages = processor._build_messages("user: hello\nassistant:")
        assert messages == [{"role": "user", "content": "hello"}]

    def test_conversation_history(self) -> None:
        processor = ChatProcessor("http://test:8000", "test-model")
        prompt = "user: hi\nassistant: hey there\nuser: how are you?\nassistant:"
        messages = processor._build_messages(prompt)
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "hi"}
        assert messages[1] == {"role": "assistant", "content": "hey there"}
        assert messages[2] == {"role": "user", "content": "how are you?"}

    def test_memory_block_becomes_system_message(self) -> None:
        processor = ChatProcessor("http://test:8000", "test-model")
        prompt = (
            "[Relevant Memory]\n"
            "  1. [live, sim=0.92] some memory\n"
            "[End Memory]\n"
            "\n"
            "user: what happened?\nassistant:"
        )
        messages = processor._build_messages(prompt)
        assert messages[0]["role"] == "system"
        assert "[Relevant Memory]" in messages[0]["content"]
        assert any(m["role"] == "user" and "what happened?" in m["content"] for m in messages)

    def test_fallback_for_unparseable_prompt(self) -> None:
        processor = ChatProcessor("http://test:8000", "test-model")
        messages = processor._build_messages("just a raw string with no prefixes")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "just a raw string with no prefixes"
