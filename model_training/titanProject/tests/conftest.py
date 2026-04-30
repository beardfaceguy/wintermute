"""Shared fixtures for titanProject tests."""

import json
import sys
from array import array
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure titanProject root is importable
TITAN_ROOT = Path(__file__).resolve().parent.parent
if str(TITAN_ROOT) not in sys.path:
    sys.path.insert(0, str(TITAN_ROOT))

from model import ModelConfig


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
    config.addinivalue_line("markers", "slow: slow-running test")


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Tiny model config (fits on CPU, runs in milliseconds)
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_gpt_config():
    return ModelConfig(
        variant="gpt",
        vocab_size=256,
        dim=64,
        depth=2,
        heads=4,
        ff_mult=2,
        max_seq_len=128,
    )


@pytest.fixture
def tiny_mac_config():
    return ModelConfig(
        variant="mac",
        vocab_size=256,
        dim=64,
        depth=2,
        heads=4,
        ff_mult=2,
        segment_len=32,
        num_persist_mem_tokens=2,
        num_longterm_mem_tokens=4,
    )


# ---------------------------------------------------------------------------
# Dummy tokenizer (simple char-level, no external files needed)
# ---------------------------------------------------------------------------

class DummyTokenizer:
    """Trivial tokenizer that maps each character to its ordinal."""

    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.tokenizer_fingerprint = f"dummy-char-{vocab_size}"
        self.tokenizer_source_path = "<dummy>"
        self.eos_id = 0
        self.pad_id = 1

    def __call__(self, text: str):
        return self.encode(text)

    def encode(self, text: str):
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


@pytest.fixture
def dummy_tokenizer():
    return DummyTokenizer()


# ---------------------------------------------------------------------------
# Token cache fixtures (small on-disk shards for testing)
# ---------------------------------------------------------------------------

@pytest.fixture
def token_cache_dir(tmp_path):
    """Create a small on-disk token cache with 3 shards for testing."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()

    shard_size = 100
    shards = []
    tokens_written = 0

    for i in range(3):
        shard_data = np.arange(
            tokens_written, tokens_written + shard_size, dtype=np.uint32
        )
        shard_name = f"tokens-{i:05d}.uint32.bin"
        shard_path = cache_dir / shard_name
        shard_data.tofile(shard_path)
        shards.append({
            "filename": shard_name,
            "num_tokens": shard_size,
            "start_token": tokens_written,
        })
        tokens_written += shard_size

    manifest = {
        "cache_version": 3,
        "path": "<test>",
        "tokenizer_fingerprint": "test-fingerprint",
        "max_tokens": None,
        "num_tokens": tokens_written,
        "dtype": "uint32",
        "shard_size_tokens": shard_size,
        "shards": shards,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))

    return cache_dir, manifest


@pytest.fixture
def small_text_file(tmp_path):
    """Create a small text file with known content for dataset tests."""
    text_path = tmp_path / "train.txt"
    lines = [f"line number {i} with some words to tokenize" for i in range(200)]
    text_path.write_text("\n".join(lines) + "\n")
    return text_path
