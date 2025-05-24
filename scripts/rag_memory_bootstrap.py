import os
import sys
import shutil
from pathlib import Path
import argparse

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

# Add root dir to sys.path to enable shared imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.setup_path import extend_path
extend_path()

from shared.config_loader import load_vllm_config
from shared.vllm_llm import VLLM
# === Configuration ===
DATA_DIR = os.path.expanduser("~/wintermute/memory/live")
CHROMA_DIR = os.path.expanduser("~/wintermute/memory/chroma_store")
EMBED_MODEL = "BAAI/bge-small-en"
VLLM_URL, MODEL_NAME = load_vllm_config()


def build_index(data_dir: str, chroma_dir: str, embed_model):
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"No files found in {data_dir}. Please add documents before initializing.")
        sys.exit(1)

    documents = SimpleDirectoryReader(data_dir).load_data()
    db = chromadb.PersistentClient(path=chroma_dir)
    vector_store = ChromaVectorStore(chroma_collection=db.get_or_create_collection("wintermute_memory"))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )

    # Persist full index
    index.storage_context.persist(persist_dir=chroma_dir)
    return index


def load_existing_index(chroma_dir: str, embed_model):
    try:
        return load_index_from_storage(persist_dir=chroma_dir, embed_model=embed_model)
    except Exception as e:
        print(f"Persisted index load failed: {e}. Falling back to reconstructing vector store.")
        db = chromadb.PersistentClient(path=chroma_dir)
        vector_store = ChromaVectorStore(chroma_collection=db.get_or_create_collection("wintermute_memory"))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model,
        )


def query_index(index, query: str):
    llm = VLLM(base_url=VLLM_URL, model_name=MODEL_NAME)
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(query)
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap or query Wintermute memory.")
    parser.add_argument("--init", action="store_true", help="Initialize memory index from live documents.")
    parser.add_argument("--query", type=str, help="Run a query against the memory index.")
    parser.add_argument("--reset", action="store_true", help="Clear and reset the index before running.")

    args = parser.parse_args()
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    if args.reset and os.path.exists(CHROMA_DIR):
        print(f"Resetting index at {CHROMA_DIR}...")
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    if args.init:
        print("Building new index...")
        build_index(DATA_DIR, CHROMA_DIR, embed_model)
        print("Index initialized and saved.")
    elif args.query:
        index = load_existing_index(CHROMA_DIR, embed_model)
        query_index(index, args.query)
    else:
        print("No action specified. Use --init to initialize or --query to run a query.")
