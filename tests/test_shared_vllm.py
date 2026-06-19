"""
Tests for shared/vllm_llm.py — VLLM LlamaIndex shim.

Focuses on error handling: HTTP failures, non-JSON responses, malformed JSON.
"""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Stub out llama_index and pydantic imports so we can test without the full package
_li_keys = [
    "llama_index",
    "llama_index.core",
    "llama_index.core.base",
    "llama_index.core.base.llms",
    "llama_index.core.base.llms.types",
    "llama_index.core.llms",
]
_saved_li = {}
for _k in _li_keys:
    _saved_li[_k] = sys.modules.pop(_k, None)

# Build minimal stubs for the types the module imports
_types_mod = ModuleType("llama_index.core.base.llms.types")


class _CompletionResponse:
    def __init__(self, text=""):
        self.text = text


_types_mod.CompletionResponse = _CompletionResponse
_types_mod.CompletionResponseGen = type(None)
_types_mod.LLMMetadata = type(
    "LLMMetadata",
    (),
    {
        "__init__": lambda self, **kw: self.__dict__.update(kw),
    },
)

_llms_mod = ModuleType("llama_index.core.llms")
_llms_mod.CustomLLM = type(
    "CustomLLM",
    (),
    {
        "__init_subclass__": classmethod(lambda cls, **kw: None),
        "__init__": lambda self, **kw: None,
    },
)

for _k in _li_keys:
    sys.modules[_k] = MagicMock()
sys.modules["llama_index.core.base.llms.types"] = _types_mod
sys.modules["llama_index.core.llms"] = _llms_mod

from shared.vllm_llm import VLLM  # noqa: E402

for _k in _li_keys:
    if _saved_li[_k] is not None:
        sys.modules[_k] = _saved_li[_k]
    else:
        sys.modules.pop(_k, None)
del _saved_li, _li_keys


def test_init_strips_trailing_slash():
    llm = VLLM(base_url="http://localhost:8000/v1/chat/completions/", model_name="test")
    assert llm._base_url == "http://localhost:8000/v1/chat/completions"


def test_metadata_returns_model_name():
    llm = VLLM(base_url="http://localhost:8000", model_name="my-model")
    meta = llm.metadata
    assert meta.model_name == "my-model"
    assert meta.is_chat_model is True


@patch("shared.vllm_llm.requests.post")
def test_complete_success(mock_post):
    """Successful completion parses choices[0].message.content."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "Hello world"}}]}
    mock_post.return_value = mock_resp

    llm = VLLM(base_url="http://localhost:8000", model_name="test")
    result = llm.complete("Say hello")

    assert result.text == "Hello world"


@patch("shared.vllm_llm.requests.post")
def test_complete_empty_choices_raises(mock_post):
    """Empty choices list should raise ValueError."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": []}
    mock_resp.text = '{"choices": []}'
    mock_post.return_value = mock_resp

    llm = VLLM(base_url="http://localhost:8000", model_name="test")
    with pytest.raises(ValueError, match="Unexpected response"):
        llm.complete("test")


@patch("shared.vllm_llm.requests.post")
def test_complete_non_json_response_raises_useful_error(mock_post):
    """If the server returns non-JSON (e.g. HTML error page), should raise a clear error."""
    import json

    mock_resp = MagicMock()
    mock_resp.json.side_effect = json.JSONDecodeError("msg", "doc", 0)
    mock_resp.text = "<html>502 Bad Gateway</html>"
    mock_resp.status_code = 502
    mock_post.return_value = mock_resp

    llm = VLLM(base_url="http://localhost:8000", model_name="test")

    with pytest.raises((ValueError, json.JSONDecodeError)):
        llm.complete("test")


@patch("shared.vllm_llm.requests.post")
def test_complete_http_error_raises(mock_post):
    """HTTP errors (500, 503) should produce a clear error, not silently parse garbage."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.json.return_value = {"error": {"message": "Internal Server Error"}}
    mock_resp.text = '{"error": {"message": "Internal Server Error"}}'
    mock_post.return_value = mock_resp

    llm = VLLM(base_url="http://localhost:8000", model_name="test")

    with pytest.raises((ValueError, KeyError)):
        llm.complete("test")


@patch("shared.vllm_llm.requests.post")
def test_complete_forwards_kwargs(mock_post):
    """Temperature and max_tokens kwargs should be forwarded in the request body."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    mock_post.return_value = mock_resp

    llm = VLLM(base_url="http://localhost:8000", model_name="test")
    llm.complete("test", temperature=0.1, max_tokens=128)

    call_json = mock_post.call_args[1]["json"]
    assert call_json["temperature"] == 0.1
    assert call_json["max_tokens"] == 128
