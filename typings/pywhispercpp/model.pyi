from collections.abc import Callable
from typing import Any, TextIO

import numpy as np

__author__: str
__copyright__: str
__license__: str
__version__: str
logger: Any

class Segment:
    """
    A small class representing a transcription segment
    """

    def __init__(self, t0: int, t1: int, text: str) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    t0: int
    t1: int
    text: str

class Model:
    _new_segment_callback: Callable[[Segment], None] | None

    def __init__(
        self,
        model: str = ...,
        models_dir: str = ...,
        params_sampling_strategy: int = ...,
        redirect_whispercpp_logs_to: bool | TextIO | str | None = ...,
        **params: Any,
    ) -> None: ...
    def transcribe(
        self,
        media: str | np.ndarray[Any, Any],
        n_processors: int = ...,
        new_segment_callback: Callable[[Segment], None] | None = ...,
        **params: Any,
    ) -> list[Segment]: ...
    def get_params(self) -> dict[str, Any]: ...
    @staticmethod
    def get_params_schema() -> dict[str, Any]: ...
    @staticmethod
    def lang_max_id() -> int: ...
    def print_timings(self) -> None: ...
    @staticmethod
    def system_info() -> None: ...
    @staticmethod
    def available_languages() -> list[str]: ...
    def auto_detect_language(
        self,
        media: str | np.ndarray[Any, Any],
        offset_ms: int = ...,
        n_threads: int = ...,
    ) -> tuple[tuple[str, np.float32], dict[str, np.float32]]: ...
    def __del__(self) -> None: ...
