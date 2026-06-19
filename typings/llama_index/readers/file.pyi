# file: typings/llama_index/readers/file.pyi

from collections.abc import Callable
from typing import Any

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

class SimpleDirectoryReader(BaseReader):
    def __init__(
        self,
        input_dir: str | None = ...,
        input_files: list[str] | None = ...,
        exclude: list[str] | None = ...,
        recursive: bool = ...,
        encoding: str = ...,
        num_files_limit: int | None = ...,
        file_metadata: Callable[[str], dict[str, Any]] | None = ...,
    ) -> None: ...
    def load_data(self, **kwargs: Any) -> list[Document]: ...
    def iter_data(self, **kwargs: Any) -> list[Document]: ...
