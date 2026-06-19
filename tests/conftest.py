"""
Root-level conftest.py for the Wintermute platform test suite.

Provides shared fixtures used across all test directories:
  - Temporary config files
  - Mock embedding functions
  - Async event loop configuration
  - Common test data factories
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def isolate_config_cache():
    """Reset the shared config cache between tests so mutations don't leak."""
    import shared.config_loader as cl

    original = cl._config_cache
    cl._config_cache = None
    yield
    cl._config_cache = original


@pytest.fixture()
def tmp_config(tmp_path):
    """Write a minimal shared_api_config.json and patch config_loader to use it."""
    cfg = {
        "vllm": {
            "scheme": "http",
            "host": "localhost",
            "port": 8010,
            "path": "/v1/completions",
            "model": "test-model",
            "aws": {},
        },
        "web_interface": {
            "scheme": "ws",
            "host": "localhost",
            "port": 8000,
            "path": "/ws/chat",
        },
        "rag": {
            "storage_dir": "memory/chroma_store",
            "live_data_dir": "memory/live",
            "embed_model": "BAAI/bge-small-en",
            "device": "cpu",
        },
    }
    cfg_path = tmp_path / "shared_api_config.json"
    cfg_path.write_text(json.dumps(cfg))
    import shared.config_loader as cl

    cl._config_cache = None
    with patch.object(cl, "config_path", cfg_path):
        yield cfg_path, cfg


@pytest.fixture()
def mock_embed():
    """Return a deterministic fake embedding function (384-dim zeros)."""

    def _embed(text: str) -> list[float]:
        return [0.0] * 384

    return _embed


@pytest.fixture()
def mock_embed_unique():
    """Return a fake embedding function that produces unique vectors per input."""
    _cache: dict[str, list[float]] = {}
    _counter = [0]

    def _embed(text: str) -> list[float]:
        if text not in _cache:
            vec = [0.0] * 384
            vec[_counter[0] % 384] = 1.0
            _cache[text] = vec
            _counter[0] += 1
        return _cache[text]

    return _embed
