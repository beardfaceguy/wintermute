from fastapi import APIRouter, Query

from pydantic import BaseModel
from starlette.responses import StreamingResponse
import httpx
import asyncio

from utils.rag_utils import query_rag, get_chat_response
from utils.config_utils import get_vllm_url, get_llm_model_name

router = APIRouter()


class Message(BaseModel):
    text: str


@router.post("/chat/stream")
async def chat_stream_endpoint(message: Message):
    async def token_generator():
        response = await asyncio.to_thread(get_chat_response, message.text)
        for token in response.split():
            yield token + " "
            await asyncio.sleep(0.01)

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@router.get("/rag/query")
async def rag_query(q: str = Query(..., description="The query string")):
    try:
        result = await asyncio.to_thread(query_rag, q)
        return {"response": str(result)}
    except Exception as e:
        print(f"/rag/query failed: {e}")
        return {"error": f"Failed to load or initialize index: {e}"}
