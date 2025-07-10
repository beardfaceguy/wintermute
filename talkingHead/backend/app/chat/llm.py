import json
import os
import httpx
from shared.config_loader import load_vllm_config

VLLM_URL, MODEL_NAME = load_vllm_config()
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

class ChatProcessor:
    def __init__(self, model_url: str = VLLM_URL, model_name: str = MODEL_NAME):
        self.model_url = model_url
        self.model_name = model_name

    async def stream_response(self, prompt: str, send_token_callback) -> str:
        """
        Stream response from the LLM to the provided send_token_callback function.
        Returns the full assembled assistant response string.
        """
        assistant_response = ""

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                self.model_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": True,
                    "max_tokens": 512,
                    "temperature": 0.95,
                    "top_p": 0.95,
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
