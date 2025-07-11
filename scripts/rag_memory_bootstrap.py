import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    print("ChromaDB is not installed. Please install chromadb to use this script.")
    exit(1)
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llms import CompletionResponse
from llama_index.core.query_engine import BaseQueryEngine

try:
    from llama_index.core import (
        Document,
        Settings,
        StorageContext,
        VectorStoreIndex,
        load_index_from_storage,
    )
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
except ImportError as e:
    print(f"Failed to import llama-index or related modules:\n{e}")
    exit(1)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from shared.vllm_llm import VLLM

# Load configuration
CONFIG_PATH = "config/shared_api_config.json"
config: dict[str, Any] = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
else:
    print(f"Config file {CONFIG_PATH} not found. Proceeding with default settings.")

# Extract RAG settings
rag_config = config.get("rag", {})
persist_base_dir = rag_config.get("storage_dir", "./storage")
chroma_persist_dir = os.path.join(persist_base_dir, "chroma")
docs_path = rag_config.get("live_data_dir", "./data")
embed_model_name = rag_config.get("embed_model", "BAAI/bge-small-en")
device = rag_config.get("device", "cpu").lower()
collection_name = "rag_collection"

# Extract VLLM LLM settings
vllm_config = config.get("vllm", {})
llm_model = vllm_config.get("model")
llm_scheme = vllm_config.get("scheme")
llm_host = vllm_config.get("host")
llm_port = vllm_config.get("port")
llm_path = vllm_config.get("path")

base_url = f"{llm_scheme}://{llm_host}:{llm_port}{llm_path}"
llm = VLLM(base_url=base_url, model_name=llm_model)

# Set up embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name, device=device)

# Index metadata path
meta_path = os.path.join(persist_base_dir, "index_metadata.json")


def is_index_valid() -> bool:
    if not os.path.isdir(persist_base_dir):
        return False
    if not os.path.isdir(chroma_persist_dir):
        return False
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("embedding_model") != embed_model_name:
                return False
        except Exception:
            return False
    try:
        client = chromadb.PersistentClient(
            path=chroma_persist_dir, settings=ChromaSettings(anonymized_telemetry=False)
        )
        collection = client.get_collection(collection_name)
        if collection.count() == 0:
            return False
    except Exception:
        return False
    return True


def reset_index() -> None:
    if os.path.isdir(persist_base_dir):
        shutil.rmtree(persist_base_dir, ignore_errors=True)
    if os.path.isdir(chroma_persist_dir):
        if not chroma_persist_dir.startswith(persist_base_dir):
            shutil.rmtree(chroma_persist_dir, ignore_errors=True)
    print("Reset: Existing index and vector store have been deleted.")


def init_index() -> None:
    print("Initializing a new vector index...")
    client = chromadb.PersistentClient(
        path=chroma_persist_dir, settings=ChromaSettings(anonymized_telemetry=False)
    )
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    docs = []

    if os.path.isdir(docs_path):
        reader = SimpleDirectoryReader(docs_path)  # type: ignore
        docs = reader.load_data()
    elif os.path.isfile(docs_path):
        with open(docs_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        docs = [Document(text)]  # type: ignore
    else:
        print(
            f"Warning: No valid document source found at '{docs_path}'. The index will be empty."
        )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    if docs:
        _index = VectorStoreIndex.from_documents(
            docs, storage_context=storage_context, llm=llm
        )
    else:
        _index = VectorStoreIndex.from_documents(
            [], storage_context=storage_context, llm=llm
        )
    storage_context.save(persist_dir=persist_base_dir)  # type: ignore
    meta = {"embedding_model": embed_model_name}
    os.makedirs(persist_base_dir, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(
        f"Index initialized and saved to '{persist_base_dir}' (Chroma DB at '{chroma_persist_dir}'). Documents indexed: {len(docs)}."
    )


def query_index(question: str) -> None:
    client = chromadb.PersistentClient(
        path=chroma_persist_dir, settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=persist_base_dir
    )
    try:
        index = load_index_from_storage(storage_context)
    except Exception as e:
        print(f"Error loading index: {e}. Rebuilding index...")
        init_index()
        client = chromadb.PersistentClient(
            path=chroma_persist_dir, settings=ChromaSettings(anonymized_telemetry=False)
        )
        collection = client.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=persist_base_dir
        )
        index = load_index_from_storage(storage_context)

    query_engine: BaseQueryEngine = index.as_query_engine(llm=llm)  # type: ignore
    print(f"Query: {question}")
    response: CompletionResponse = query_engine.query(question)  # type: ignore
    print(f"Response: {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Memory Bootstrap Script")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete any existing index and vector store data.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize a new index from source documents.",
    )
    parser.add_argument(
        "--query",
        type=str,
        metavar="QUESTION",
        help="Query the index with a given question.",
    )
    args = parser.parse_args()

    if args.reset:
        reset_index()
    if args.init:
        init_index()
    if args.query:
        if not is_index_valid():
            print("No valid index found. Building a new index...")
            init_index()
        else:
            print("Existing index is valid. Proceeding to query.")
        query_index(args.query)
