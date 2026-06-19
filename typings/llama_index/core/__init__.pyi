# typings/llama_index/core/__init__.pyi

from typing import Any, Protocol

# Declare lightweight stubs for the classes used
class StorageContext(Protocol):
    @classmethod
    def from_defaults(cls, **kwargs: Any) -> StorageContext: ...

class BaseIndex(Protocol): ...

class VectorStoreIndex(BaseIndex, Protocol):
    @classmethod
    def from_documents(cls, documents: list[Any], **kwargs: Any) -> VectorStoreIndex: ...

class SimpleDirectoryReader(Protocol):
    def load_data(self) -> list[Any]: ...

class Document: ...

class _Settings(Protocol):
    embed_model: Any

Settings: _Settings

def load_index_from_storage(
    storage_context: StorageContext, index_id: str | None = ..., **kwargs: Any
) -> BaseIndex: ...

class ChatResponse:
    response: str
