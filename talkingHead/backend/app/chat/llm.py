import json
import logging
import os
from typing import Awaitable, Callable

import httpx

from shared.config_loader import load_vllm_config

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
logger = logging.getLogger("chat.llm")


class ChatProcessor:
    def __init__(self, model_url: str | None = None, model_name: str | None = None):
        if model_url and model_name:
            self.model_url = model_url
            self.model_name = model_name
        else:
            try:
                url, model = load_vllm_config()
                self.model_url = model_url or url
                self.model_name = model_name or model
            except RuntimeError as e:
                logger.warning("vLLM config not available: %s", e)
                self.model_url = ""
                self.model_name = ""

    async def stream_response(
        self, prompt: str, send_token_callback: Callable[[str], Awaitable[None]]
    ) -> str:
        """
        Stream response from the LLM to the provided send_token_callback function.
        Returns the full assembled assistant response string.
        """
        if not self.model_url:
            msg = "LLM not configured — set VLLM_HOST env var"
            await send_token_callback(msg)
            return msg

        assistant_response = ""

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                self.model_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": True,
                    "max_tokens": 256,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repetition_penalty": 1.3,
                    "frequency_penalty": 0.5,
                },
            ) as response:
                async for line in response.aiter_lines():
                    if not line.strip() or line.startswith(":"):
                        continue
                    if line.strip() == "data: [DONE]":
                        break
                    try:
                        payload = json.loads(line.removeprefix("data: "))
                        content = payload["choices"][0].get("text", "")
                        if content:
                            assistant_response += content
                            await send_token_callback(content)
                    except Exception as e:
                        if DEBUG:
                            print(f"[DEBUG] LLM parse error: {e}")

        return assistant_response
