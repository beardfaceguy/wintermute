"""Tests for data.py: TokenCache, TextWindowDataset, cache utilities."""

import json
import os
from array import array
from pathlib import Path

import numpy as np
import pytest
import torch

from data import (
    TokenCache,
    TextWindowDataset,
    build_cache_key,
    build_dataloader,
    get_cache_root,
    get_shard_size_tokens,
    DEFAULT_CACHE_VERSION,
    DEFAULT_SHARD_SIZE_TOKENS,
)


class TestTokenCache:
    """Token cache shard-based slicing."""

    def test_slice_within_single_shard(self, token_cache_dir):
        cache_dir, manifest = token_cache_dir
        cache = TokenCache(cache_dir, cache_dir / "manifest.json", manifest)

        result = cache.slice(10, 20)
        expected = torch.arange(10, 20, dtype=torch.int64)
        torch.testing.assert_close(result, expected)

    def test_slice_across_shard_boundary(self, token_cache_dir):
        cache_dir, manifest = token_cache_dir
        cache = TokenCache(cache_dir, cache_dir / "manifest.json", manifest)

        # Shards are 100 tokens each, so 90..110 crosses shard 0→1
        result = cache.slice(90, 110)
        expected = torch.arange(90, 110, dtype=torch.int64)
        torch.testing.assert_close(result, expected)

    def test_slice_across_multiple_shards(self, token_cache_dir):
        cache_dir, manifest = token_cache_dir
        cache = TokenCache(cache_dir, cache_dir / "manifest.json", manifest)

        # 50..250 spans all 3 shards
        result = cache.slice(50, 250)
        expected = torch.arange(50, 250, dtype=torch.int64)
        torch.testing.assert_close(result, expected)

    def test_slice_entire_cache(self, token_cache_dir):
        cache_dir, manifest = token_cache_dir
        cache = TokenCache(cache_dir, cache_dir / "manifest.json", manifest)

        result = cache.slice(0, 300)
        expected = torch.arange(0, 300, dtype=torch.int64)
        torch.testing.assert_close(result, expected)

    def test_slice_single_element(self, token_cache_dir):
        cache_dir, manifest = token_cache_dir
        cache = TokenCache(cache_dir, cache_dir / "manifest.json", manifest)

        result = cache.slice(42, 43)
        assert result.shape == (1,)
        assert result.item() == 42

    def test_num_tokens(self, token_cache_dir):
        cache_dir, manifest = token_cache_dir
        cache = TokenCache(cache_dir, cache_dir / "manifest.json", manifest)
        assert cache.num_tokens == 300

    def test_memmap_lazy_loading(self, token_cache_dir):
        cache_dir, manifest = token_cache_dir
        cache = TokenCache(cache_dir, cache_dir / "manifest.json", manifest)

        assert all(m is None for m in cache._memmaps)
        cache.slice(0, 10)
        assert cache._memmaps[0] is not None
        assert cache._memmaps[1] is None
        assert cache._memmaps[2] is None


class TestTextWindowDataset:
    """Dataset indexing and window construction."""

    def test_dataset_length(self, small_text_file, dummy_tokenizer):
        ds = TextWindowDataset(
            str(small_text_file),
            dummy_tokenizer,
            dummy_tokenizer.tokenizer_fingerprint,
            seq_len=32,
        )
        expected_windows = (ds.num_tokens - 1) // 32
        assert len(ds) == expected_windows

    def test_getitem_shapes(self, small_text_file, dummy_tokenizer):
        ds = TextWindowDataset(
            str(small_text_file),
            dummy_tokenizer,
            dummy_tokenizer.tokenizer_fingerprint,
            seq_len=32,
        )
        x, y = ds[0]
        assert x.shape == (32,)
        assert y.shape == (32,)

    def test_getitem_target_is_shifted(self, small_text_file, dummy_tokenizer):
        ds = TextWindowDataset(
            str(small_text_file),
            dummy_tokenizer,
            dummy_tokenizer.tokenizer_fingerprint,
            seq_len=32,
        )
        x, y = ds[0]
        # y should be x shifted by 1 position
        x2, _ = ds[0]
        # Verify by checking that y[:-1] overlaps with x[1:]
        # (y is tokens at positions [1..32], x is tokens at positions [0..31])
        torch.testing.assert_close(y[:-1], x[1:])

    def test_max_tokens_cap(self, small_text_file, dummy_tokenizer):
        ds_full = TextWindowDataset(
            str(small_text_file),
            dummy_tokenizer,
            dummy_tokenizer.tokenizer_fingerprint,
            seq_len=32,
        )
        ds_capped = TextWindowDataset(
            str(small_text_file),
            dummy_tokenizer,
            dummy_tokenizer.tokenizer_fingerprint,
            seq_len=32,
            max_tokens=500,
        )
        assert ds_capped.num_tokens <= 500
        assert ds_capped.num_tokens < ds_full.num_tokens

    def test_cache_reuse(self, small_text_file, dummy_tokenizer):
        """Building the dataset twice with same params should reuse cache."""
        ds1 = TextWindowDataset(
            str(small_text_file),
            dummy_tokenizer,
            dummy_tokenizer.tokenizer_fingerprint,
            seq_len=32,
        )
        ds2 = TextWindowDataset(
            str(small_text_file),
            dummy_tokenizer,
            dummy_tokenizer.tokenizer_fingerprint,
            seq_len=32,
        )
        assert ds1.num_tokens == ds2.num_tokens
        assert ds1.token_cache.cache_dir == ds2.token_cache.cache_dir

    def test_different_fingerprint_builds_new_cache(self, small_text_file, dummy_tokenizer):
        ds1 = TextWindowDataset(
            str(small_text_file),
            dummy_tokenizer,
            "fingerprint-a",
            seq_len=32,
        )
        ds2 = TextWindowDataset(
            str(small_text_file),
            dummy_tokenizer,
            "fingerprint-b",
            seq_len=32,
        )
        assert ds1.token_cache.cache_dir != ds2.token_cache.cache_dir

    def test_too_few_tokens_raises(self, tmp_path, dummy_tokenizer):
        tiny = tmp_path / "tiny.txt"
        tiny.write_text("hi")
        with pytest.raises(ValueError, match="Not enough tokens"):
            TextWindowDataset(
                str(tiny),
                dummy_tokenizer,
                dummy_tokenizer.tokenizer_fingerprint,
                seq_len=1024,
            )


class TestBuildDataloader:
    """Dataloader construction."""

    def test_returns_dataloader(self, small_text_file, dummy_tokenizer):
        loader = build_dataloader(
            str(small_text_file),
            dummy_tokenizer,
            dummy_tokenizer.tokenizer_fingerprint,
            seq_len=32,
            batch_size=4,
        )
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (4, 32)
        assert batch_y.shape == (4, 32)

    def test_drop_last(self, small_text_file, dummy_tokenizer):
        loader = build_dataloader(
            str(small_text_file),
            dummy_tokenizer,
            dummy_tokenizer.tokenizer_fingerprint,
            seq_len=32,
            batch_size=4,
        )
        for x, y in loader:
            assert x.shape[0] == 4


class TestCacheUtilities:
    """Cache key generation and helper functions."""

    def test_build_cache_key_deterministic(self):
        key1 = build_cache_key("/some/path.txt", "fp123", None, 1750000)
        key2 = build_cache_key("/some/path.txt", "fp123", None, 1750000)
        assert key1 == key2

    def test_build_cache_key_varies_with_path(self):
        key1 = build_cache_key("/path/a.txt", "fp", None, 1750000)
        key2 = build_cache_key("/path/b.txt", "fp", None, 1750000)
        assert key1 != key2

    def test_build_cache_key_varies_with_fingerprint(self):
        key1 = build_cache_key("/p.txt", "fp-a", None, 1750000)
        key2 = build_cache_key("/p.txt", "fp-b", None, 1750000)
        assert key1 != key2

    def test_build_cache_key_varies_with_max_tokens(self):
        key1 = build_cache_key("/p.txt", "fp", 1000, 1750000)
        key2 = build_cache_key("/p.txt", "fp", 2000, 1750000)
        assert key1 != key2

    def test_build_cache_key_varies_with_shard_size(self):
        key1 = build_cache_key("/p.txt", "fp", None, 1000000)
        key2 = build_cache_key("/p.txt", "fp", None, 2000000)
        assert key1 != key2

    def test_build_cache_key_length(self):
        key = build_cache_key("/p.txt", "fp", None, 1750000)
        assert len(key) == 24

    def test_get_cache_root_local(self):
        root = get_cache_root("/some/dir/train.txt")
        assert root == Path("/some/dir/.titan_token_cache")

    def test_get_cache_root_s3(self):
        root = get_cache_root("s3://bucket/prefix/train.txt")
        assert ".cache" in str(root)
        assert "wintermute" in str(root)

    def test_get_shard_size_tokens_default(self, monkeypatch):
        monkeypatch.delenv("TITAN_TOKEN_CACHE_SHARD_SIZE_TOKENS", raising=False)
        assert get_shard_size_tokens() == DEFAULT_SHARD_SIZE_TOKENS

    def test_get_shard_size_tokens_override(self, monkeypatch):
        monkeypatch.setenv("TITAN_TOKEN_CACHE_SHARD_SIZE_TOKENS", "500000")
        assert get_shard_size_tokens() == 500000

    def test_get_shard_size_tokens_invalid(self, monkeypatch):
        monkeypatch.setenv("TITAN_TOKEN_CACHE_SHARD_SIZE_TOKENS", "not_a_number")
        with pytest.raises(ValueError, match="must be an integer"):
            get_shard_size_tokens()

    def test_get_shard_size_tokens_zero(self, monkeypatch):
        monkeypatch.setenv("TITAN_TOKEN_CACHE_SHARD_SIZE_TOKENS", "0")
        with pytest.raises(ValueError, match="must be > 0"):
            get_shard_size_tokens()
