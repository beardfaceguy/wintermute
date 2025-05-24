from llama_index.core.llms import CustomLLM
from llama_index.core.base.llms.types import (
    LLMMetadata,
    CompletionResponse,
    CompletionResponseGen,
    ChatMessage,
)
import requests
from typing import Optional, List, Generator
from pydantic import PrivateAttr


class VLLM(CustomLLM):
    _base_url: str = PrivateAttr()
    _model_name: str = PrivateAttr()

    def __init__(self, base_url: str, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        # Construct the chat-style message body
        response = requests.post(
            self._base_url,  # Already contains full path
            headers={"Content-Type": "application/json"},
            json={
                "model": self._model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 256),
                "stream": False,
            },
        )
        try:
            text = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise ValueError(f"Unexpected response from vLLM: {response.text}")
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        # Dummy fallback streaming
        yield self.complete(prompt, **kwargs)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192,
            num_output=1,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self._model_name,
        )
