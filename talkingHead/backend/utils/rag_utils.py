from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.llms import ChatResponse
from utils.rag_service import RAGService

_rag = RAGService()


def query_rag(question: str) -> str:
    return _rag.query(question)


def init_rag():
    _rag.init()


def reset_rag():
    _rag.reset()


def is_rag_valid() -> bool:
    return _rag.is_valid()


def get_chat_response(user_input: str) -> str:
    chat_engine: BaseChatEngine = _rag.get_chat_engine()  # type: ignore
    response: ChatResponse = chat_engine.chat(user_input)  # type: ignore
    return str(response.response)  # type: ignore
