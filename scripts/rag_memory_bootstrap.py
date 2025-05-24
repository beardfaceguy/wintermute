from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index import HuggingFaceEmbedding, ChromaVectorStore
from llama_index.llms.openai import OpenAI
import chromadb
import os
from shared.config_loader import load_vllm_config
VLLM_URL, MODEL_NAME = load_vllm_config()

# === Configuration ===
DATA_DIR = os.path.expanduser("~/wintermute/memory/live")
CHROMA_DIR = os.path.expanduser("~/wintermute/memory/chroma_store")
VLLM_ENDPOINT = VLLM_URL
EMBED_MODEL = "BAAI/bge-small-en"

# === Load or create index ===
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

if os.path.exists(CHROMA_DIR):
    # Load existing index
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    vector_store = ChromaVectorStore(chroma_collection=db.get_or_create_collection("wintermute_memory"))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = load_index_from_storage(storage_context, embed_model=embed_model)
else:
    # Build new index
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    vector_store = ChromaVectorStore(chroma_collection=db.get_or_create_collection("wintermute_memory"))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )

# === Connect to your local vLLM ===
llm = OpenAI(
    model="mistral-7b-instruct-awq",
    base_url=VLLM_ENDPOINT,
    api_key="not-needed-for-local",  # vLLM doesn't validate it
)

# === Query example ===
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What is Wintermute's cold memory strategy?")
print(response)
