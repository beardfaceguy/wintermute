import json
import logging
import os
from typing import Awaitable, Callable

import httpx

from shared.config_loader import load_vllm_config, load_inference_config

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
logger = logging.getLogger("chat.llm")

# Streaming HTTP timeouts (seconds). Read timeout is generous because
# token streams can stall briefly between deltas; finite to avoid forever-hangs.
_LLM_CONNECT_TIMEOUT = float(os.getenv("LLM_CONNECT_TIMEOUT", "10"))
_LLM_READ_TIMEOUT = float(os.getenv("LLM_READ_TIMEOUT", "300"))
_LLM_WRITE_TIMEOUT = float(os.getenv("LLM_WRITE_TIMEOUT", "10"))


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

        try:
            self._inference = load_inference_config()
        except Exception:
            self._inference = {
                "max_tokens": 512, "temperature": 0.7,
                "top_p": 0.9, "frequency_penalty": 0.5,
            }

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        """Convert a formatted prompt string into chat messages.

        The prompt follows the convention:
            [Relevant Memory] ... [End Memory]
            user: ...
            assistant: ...
            user: <latest>
            assistant:

        We parse this into proper role/content message dicts so
        instruction-tuned models (Mistral, etc.) get the right format.
        """
        messages: list[dict[str, str]] = []
        system_parts: list[str] = []
        lines = prompt.split("\n")

        i = 0
        # Collect memory block as system context
        if lines and lines[0].startswith("[Relevant Memory]"):
            while i < len(lines):
                system_parts.append(lines[i])
                if lines[i].strip() == "[End Memory]":
                    i += 1
                    break
                i += 1

        if system_parts:
            messages.append({"role": "system", "content": "\n".join(system_parts)})

        # Parse role-prefixed lines
        current_role = None
        current_content: list[str] = []

        for line in lines[i:]:
            if line.startswith("user: "):
                if current_role and current_content:
                    messages.append({"role": current_role, "content": "\n".join(current_content)})
                current_role = "user"
                current_content = [line[6:]]
            elif line.startswith("assistant:"):
                if current_role and current_content:
                    messages.append({"role": current_role, "content": "\n".join(current_content)})
                current_role = "assistant"
                rest = line[10:].strip()
                current_content = [rest] if rest else []
            elif current_role:
                current_content.append(line)

        if current_role and current_content:
            text = "\n".join(current_content).strip()
            if text:
                messages.append({"role": current_role, "content": text})

        # Ensure we have at least one user message
        if not any(m["role"] == "user" for m in messages):
            messages.append({"role": "user", "content": prompt})

        return messages

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
        messages = self._build_messages(prompt)

        timeout = httpx.Timeout(
            connect=_LLM_CONNECT_TIMEOUT,
            read=_LLM_READ_TIMEOUT,
            write=_LLM_WRITE_TIMEOUT,
            pool=_LLM_CONNECT_TIMEOUT,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                self.model_url,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": True,
                    "max_tokens": self._inference.get("max_tokens", 512),
                    "temperature": self._inference.get("temperature", 0.7),
                    "top_p": self._inference.get("top_p", 0.9),
                    "frequency_penalty": self._inference.get("frequency_penalty", 0.5),
                },
            ) as response:
                async for line in response.aiter_lines():
                    if not line.strip() or line.startswith(":"):
                        continue
                    if line.strip() == "data: [DONE]":
                        break
                    try:
                        payload = json.loads(line.removeprefix("data: "))
                        delta = payload["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            assistant_response += content
                            await send_token_callback(content)
                    except Exception as e:
                        if DEBUG:
                            print(f"[DEBUG] LLM parse error: {e}")

        return assistant_response
