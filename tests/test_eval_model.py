"""
Tests for eval/model.py — backend factory and provider shortcut parsing.

All tests are hermetic (no network calls). We test that make_backend()
constructs the right backend type and sets model_id correctly.
"""

from unittest.mock import MagicMock, patch

import pytest

from eval.model import (
    AnthropicBackend,
    GenerateConfig,
    HFLocalBackend,
    OllamaBackend,
    OpenAICompatBackend,
    make_backend,
)


class TestOllamaBackend:
    """--no-think routing + native /api/chat behavior (#918)."""

    def test_no_think_routes_to_ollama(self):
        b = make_backend("http://localhost:11434", model="qwen3:8b", think=False)
        assert isinstance(b, OllamaBackend)
        assert b.model_id == "qwen3:8b"

    def test_default_http_still_openai_compat(self):
        with patch("openai.OpenAI"):
            b = make_backend("http://localhost:11434", model="qwen3:8b")
        assert isinstance(b, OpenAICompatBackend)

    def test_chat_posts_to_api_chat_with_think_false(self):
        b = make_backend("http://localhost:11434", model="qwen3:8b", think=False)
        fake = MagicMock()
        fake.json.return_value = {"message": {"content": "The answer is 42."}}
        with patch("httpx.post", return_value=fake) as mock_post:
            out = b.chat([{"role": "user", "content": "hi"}], GenerateConfig(max_tokens=16))
        assert out == "The answer is 42."
        assert mock_post.call_args[0][0].endswith("/api/chat")
        body = mock_post.call_args.kwargs["json"]
        assert body["think"] is False
        assert body["model"] == "qwen3:8b"
        assert body["options"]["num_predict"] == 16

    def test_strips_residual_think_block(self):
        b = make_backend("http://localhost:11434", model="qwen3:8b", think=False)
        fake = MagicMock()
        fake.json.return_value = {"message": {"content": "<think>hmm</think>Final: 7"}}
        with patch("httpx.post", return_value=fake):
            out = b.chat([{"role": "user", "content": "x"}], GenerateConfig())
        assert out == "Final: 7"

    def test_empty_content_returns_empty_string(self):
        b = make_backend("http://localhost:11434", model="qwen3:8b", think=True)
        fake = MagicMock()
        fake.json.return_value = {"message": {"content": None}}
        with patch("httpx.post", return_value=fake):
            out = b.chat([{"role": "user", "content": "x"}], GenerateConfig())
        assert out == ""


class TestMakeBackendFactory:
    """make_backend() shortcut routing."""

    def test_http_target_returns_openai_compat(self):
        with patch("openai.OpenAI"):
            b = make_backend("http://localhost:11434", model="mistral:7b")
        assert isinstance(b, OpenAICompatBackend)
        assert b.model_id == "mistral:7b"

    def test_https_target_returns_openai_compat(self):
        with patch("openai.OpenAI"):
            b = make_backend("https://api.openai.com", model="gpt-4o")
        assert isinstance(b, OpenAICompatBackend)
        assert b.model_id == "gpt-4o"

    def test_openai_shortcut(self):
        with patch("openai.OpenAI"):
            b = make_backend("openai:gpt-4o")
        assert isinstance(b, OpenAICompatBackend)
        assert b.model_id == "gpt-4o"

    def test_openai_mini_shortcut(self):
        with patch("openai.OpenAI"):
            b = make_backend("openai:gpt-4o-mini")
        assert b.model_id == "gpt-4o-mini"

    def test_gemini_shortcut(self):
        with patch("openai.OpenAI"):
            b = make_backend("gemini:gemini-2.5-pro")
        assert isinstance(b, OpenAICompatBackend)
        assert b.model_id == "gemini-2.5-pro"

    def test_groq_shortcut(self):
        with patch("openai.OpenAI"):
            b = make_backend("groq:llama-3.3-70b-versatile")
        assert b.model_id == "llama-3.3-70b-versatile"

    def test_mistral_shortcut(self):
        with patch("openai.OpenAI"):
            b = make_backend("mistral:mistral-large-latest")
        assert b.model_id == "mistral-large-latest"

    def test_together_shortcut(self):
        with patch("openai.OpenAI"):
            b = make_backend("together:meta-llama/Llama-3-70b-chat-hf")
        assert b.model_id == "meta-llama/Llama-3-70b-chat-hf"

    def test_anthropic_shortcut(self):
        mock_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            mock_anthropic.Anthropic.return_value = MagicMock()
            b = make_backend("anthropic:claude-sonnet-4-6")
        assert isinstance(b, AnthropicBackend)
        assert b.model_id == "claude-sonnet-4-6"

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("transformers"),
        reason="transformers not installed",
    )
    @pytest.mark.skipif(True, reason="transformers not installed in test env")
    def test_hf_prefix_returns_hf_backend(self):
        with patch("transformers.AutoTokenizer.from_pretrained"), patch("transformers.pipeline"):
            b = make_backend("hf:mistralai/Mistral-7B-Instruct-v0.3")
        assert isinstance(b, HFLocalBackend)
        assert b.model_id == "mistralai/Mistral-7B-Instruct-v0.3"

    @pytest.mark.skipif(True, reason="transformers not installed in test env")
    def test_hf_local_path(self):
        with patch("transformers.AutoTokenizer.from_pretrained"), patch("transformers.pipeline"):
            b = make_backend("hf:/mnt/checkpoints/step3000")
        assert b.model_id == "/mnt/checkpoints/step3000"


class TestAnthropicBackendEmptyResponse:
    """Regression: empty content list should return '' not raise IndexError."""

    def test_empty_content_returns_empty_string(self):
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = []  # empty — triggers the Claude GPQA bug if unchecked
        mock_client.messages.create.return_value = mock_response

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            mock_anthropic.Anthropic.return_value = mock_client
            b = AnthropicBackend("claude-sonnet-4-6")
            result = b.chat([{"role": "user", "content": "hi"}], GenerateConfig(max_tokens=5))

        assert result == ""


class TestGenerateConfig:
    def test_defaults(self):
        cfg = GenerateConfig()
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 512
        assert cfg.system_prompt is None

    def test_custom(self):
        cfg = GenerateConfig(max_tokens=16, temperature=0.5, system_prompt="Be brief.")
        assert cfg.max_tokens == 16
        assert cfg.temperature == 0.5
        assert cfg.system_prompt == "Be brief."
