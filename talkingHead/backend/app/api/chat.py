from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import httpx
from urllib.parse import urlunparse
import asyncio
from shared.config_loader import load_vllm_config
VLLM_URL, MODEL_NAME = load_vllm_config()


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
