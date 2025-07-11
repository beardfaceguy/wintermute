# file: typings/llama_index/readers/file.pyi

from typing import Any, Callable, Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

class SimpleDirectoryReader(BaseReader):
    def __init__(
        self,
        input_dir: Optional[str] = ...,
        input_files: Optional[List[str]] = ...,
        exclude: Optional[List[str]] = ...,
        recursive: bool = ...,
        encoding: str = ...,
        num_files_limit: Optional[int] = ...,
        file_metadata: Optional[Callable[[str], Dict[str, Any]]] = ...,
    ) -> None: ...
    def load_data(self, **kwargs: Any) -> List[Document]: ...
    def iter_data(self, **kwargs: Any) -> List[Document]: ...
