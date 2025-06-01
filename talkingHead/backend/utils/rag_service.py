import os
import json
import shutil
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    load_index_from_storage,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer

from utils.config_utils import (
    get_embed_model_name,
    get_embed_device,
    get_storage_dir,
    get_live_data_dir,
    get_vllm_url,
    get_llm_model_name,
)


from shared.vllm_llm import VLLM


class RAGService:
    def __init__(self):
        self.storage_dir = get_storage_dir()
        self.docs_path = get_live_data_dir()
        self.chroma_dir = os.path.join(self.storage_dir, "chroma")
        self.meta_path = os.path.join(self.storage_dir, "index_metadata.json")
        self.embed_model_name = get_embed_model_name()
        self.device = get_embed_device()
        self.collection_name = "rag_collection"
        self.embedding_model = HuggingFaceEmbedding(
            model_name=self.embed_model_name, device=self.device
        )
        Settings.embed_model = (
            self.embedding_model
        )  # Critical: prevents fallback to OpenAI
        # Set LLM to avoid OpenAI fallback
        try:
            base_url = get_vllm_url()
            model_name = get_llm_model_name()
            Settings.llm = VLLM(base_url=base_url, model_name=model_name)
        except Exception as e:
            print(f"⚠️ Failed to set VLLM instance: {e}")

    def is_valid(self) -> bool:
        if not os.path.isdir(self.storage_dir) or not os.path.isdir(self.chroma_dir):
            return False
        try:
            with open(self.meta_path) as f:
                meta = json.load(f)
            if meta.get("embedding_model") != self.embed_model_name:
                return False
        except Exception:
            return False
        try:
            client = chromadb.PersistentClient(
                path=self.chroma_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            if client.get_collection(self.collection_name).count() == 0:
                return False
        except Exception:
            return False
        return True

    def reset(self):
        shutil.rmtree(self.storage_dir, ignore_errors=True)
        shutil.rmtree(self.chroma_dir, ignore_errors=True)

    def init(self):
        client = chromadb.PersistentClient(
            path=self.chroma_dir, settings=ChromaSettings(anonymized_telemetry=False)
        )
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        collection = client.get_or_create_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)

        if os.path.isdir(self.docs_path):
            docs = SimpleDirectoryReader(self.docs_path).load_data()
        elif os.path.isfile(self.docs_path):
            from llama_index.core import Document

            with open(self.docs_path, "r", encoding="utf-8", errors="ignore") as f:
                docs = [Document(f.read())]
        else:
            docs = []

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = (
            VectorStoreIndex.from_documents(docs, storage_context=storage_context)
            if docs
            else VectorStoreIndex([], storage_context=storage_context)
        )
        storage_context.persist(persist_dir=self.storage_dir)

        with open(self.meta_path, "w") as f:
            json.dump({"embedding_model": self.embed_model_name}, f)

    def query(self, question: str) -> str:
        if not self.is_valid():
            self.init()

        client = chromadb.PersistentClient(
            path=self.chroma_dir, settings=ChromaSettings(anonymized_telemetry=False)
        )
        collection = client.get_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=self.storage_dir
        )
        try:
            index = load_index_from_storage(storage_context)
        except Exception:
            self.init()
            client = chromadb.PersistentClient(
                path=self.chroma_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            collection = client.get_collection(self.collection_name)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=self.storage_dir
            )
            index = load_index_from_storage(storage_context)

        query_engine = index.as_query_engine()
        return str(query_engine.query(question))

    def _load_index(self):
        client = chromadb.PersistentClient(
            path=self.chroma_dir, settings=ChromaSettings(anonymized_telemetry=False)
        )
        collection = client.get_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=self.storage_dir
        )
        return load_index_from_storage(storage_context)

    def get_chat_engine(self, memory=None):
        if not self.is_valid():
            self.init()

        index = self._load_index()  # You already have this method
        if memory is None:
            memory = ChatMemoryBuffer(token_limit=2048)

        return CondenseQuestionChatEngine.from_defaults(
            retriever=index.as_retriever(), memory=memory, llm=Settings.llm
        )
