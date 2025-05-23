from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import httpx
from urllib.parse import urlunparse
import asyncio

VLLM_SCHEME = "http"
VLLM_HOST = "localhost"
VLLM_PORT = 8001
VLLM_PATH = "/v1/chat/completions"
MODEL_NAME = "mistral-7b-instruct-awq"

VLLM_URL = urlunparse((
    VLLM_SCHEME,
    f"{VLLM_HOST}:{VLLM_PORT}",
    VLLM_PATH,
    '', '', ''
))

router = APIRouter()

class Message(BaseModel):
    text: str

@router.post("/chat/stream")
async def chat_stream_endpoint(message: Message):
    async def token_generator():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", VLLM_URL, json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": message.text}],
                "stream": True
            }) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield line[6:] + "\n"
                        await asyncio.sleep(0.02)  # simulate delay if needed

    return StreamingResponse(token_generator(), media_type="text/event-stream")
