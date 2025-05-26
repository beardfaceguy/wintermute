import os
from pathlib import Path
import sys

from fastapi import APIRouter, Query
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import httpx
from urllib.parse import urlunparse
import asyncio
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.setup_path import extend_path
extend_path()
from shared.config_loader import load_vllm_config
from shared.config_loader import get_rag_config
from shared.vllm_llm import VLLM

VLLM_URL, MODEL_NAME = load_vllm_config()

rag_config = get_rag_config()
STORAGE_DIR = rag_config["storage_dir"]
LIVE_DATA_DIR = rag_config["live_data_dir"]
EMBED_MODEL_NAME = rag_config["embed_model"]
EMBED_DEVICE = rag_config["device"]

router = APIRouter()

def is_index_initialized(storage_dir: str) -> bool:
    required_files = ["docstore.json", "index_store.json", "graph_store.json"]
    return all(os.path.exists(os.path.join(storage_dir, f)) for f in required_files)

def get_index():
    print("Initializing index")
    print(f"storage dir = {STORAGE_DIR}")
    print(f"live data dir = {LIVE_DATA_DIR}")
    os.makedirs(STORAGE_DIR, exist_ok=True) 
    if is_index_initialized(STORAGE_DIR):
        print("Loading from existing storage")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
            return load_index_from_storage(storage_context)
        except Exception as e:
            raise RuntimeError(f"Failed to load index from storage: {e}")
    else:
        print("Creating new vector store index")
        try:
            print("before vector, storage dir = " + STORAGE_DIR)
            vector_store = ChromaVectorStore(persist_dir=STORAGE_DIR)
            print ("after vector")
            if vector_store is None:
                raise RuntimeError("failed to create vector_store")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            print("after storage")
            if storage_context is None:
                raise RuntimeError("failed to create storage_context")
    
            print(f"Loading embed model: {EMBED_MODEL_NAME} on {EMBED_DEVICE}")
            embed_model = HuggingFaceEmbedding(
                model_name=EMBED_MODEL_NAME,
                device=EMBED_DEVICE
            )
            if embed_model is None:
                raise RuntimeError("Embedding model initialization returned None.")
            index = VectorStoreIndex([], storage_context=storage_context, embed_model=embed_model)
            index.storage_context.persist(persist_dir=STORAGE_DIR)
            return index
        except Exception as e:
            raise RuntimeError(f"Failed to create HuggingFaceEmbedding: {e}")


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

@router.get("/rag/query")
async def rag_query(q: str = Query(..., description="The query string")):
    try:
        llm = VLLM(base_url=VLLM_URL, model_name=MODEL_NAME)
        if llm is None:
            raise RuntimeError("failed to create VLLM object")

        index = get_index() 
        query_engine = index.as_query_engine(llm=llm)
        result = query_engine.query(q)

        return {"response": str(result)}
    except Exception as e:
        return {"error": f"Failed to load or initialize index: {e}"}
