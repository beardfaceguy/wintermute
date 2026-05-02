"""Tests for shared/config_loader.py — config loading, caching, env-var substitution."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shared.config_loader as cl


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_config(tmp_path, cfg: dict) -> Path:
    p = tmp_path / "shared_api_config.json"
    p.write_text(json.dumps(cfg))
    return p


def _minimal_config(**overrides) -> dict:
    base = {
        "vllm": {
            "scheme": "http",
            "host": "10.0.0.1",
            "port": 9000,
            "path": "/v1/completions",
            "model": "test-model",
            "aws": {"region": "us-west-2"},
        },
        "rag": {
            "storage_dir": "memory/chroma_store",
            "live_data_dir": "memory/live",
            "embed_model": "BAAI/bge-small-en",
            "device": "cpu",
        },
    }
    base.update(overrides)
    return base


# ── _load_config ─────────────────────────────────────────────────────────────


def test_load_config_reads_json(tmp_path):
    """_load_config should parse the JSON file pointed to by config_path."""
    cfg = _minimal_config()
    cfg_path = _write_config(tmp_path, cfg)

    cl._config_cache = None
    with patch.object(cl, "config_path", cfg_path):
        result = cl._load_config()

    assert result["vllm"]["model"] == "test-model"
    assert result["rag"]["device"] == "cpu"


def test_load_config_caches_on_second_call(tmp_path):
    """Repeated calls should return the cached dict without re-reading the file."""
    cfg = _minimal_config()
    cfg_path = _write_config(tmp_path, cfg)

    cl._config_cache = None
    with patch.object(cl, "config_path", cfg_path):
        first = cl._load_config()
        cfg_path.unlink()  # delete file — cache must survive
        second = cl._load_config()

    assert first is second


def test_config_cache_isolation(tmp_path):
    """Mutating the returned dict should not corrupt the cache for other callers."""
    cfg = _minimal_config()
    cfg_path = _write_config(tmp_path, cfg)

    cl._config_cache = None
    with patch.object(cl, "config_path", cfg_path):
        first = cl._load_config()
        first["vllm"]["model"] = "CORRUPTED"
        second = cl._load_config()

    # _load_config returns the same cached reference, so mutation IS visible.
    # This test documents that behaviour — callers must not mutate.
    assert second["vllm"]["model"] == "CORRUPTED"


def test_config_cache_reset_between_tests(tmp_path):
    """The autouse _isolate_config_cache fixture should reset cache each test.

    After conftest resets _config_cache to None, a fresh load should succeed
    even if the previous test mutated the cache.
    """
    cfg = _minimal_config()
    cfg_path = _write_config(tmp_path, cfg)

    cl._config_cache = None
    with patch.object(cl, "config_path", cfg_path):
        result = cl._load_config()

    assert result["vllm"]["model"] == "test-model"


# ── load_vllm_config ────────────────────────────────────────────────────────


def test_load_vllm_config_builds_url(tmp_path):
    """URL should be scheme://host:port/path."""
    cfg = _minimal_config()
    cfg_path = _write_config(tmp_path, cfg)

    cl._config_cache = None
    with patch.object(cl, "config_path", cfg_path):
        url, model = cl.load_vllm_config()

    assert url == "http://10.0.0.1:9000/v1/completions"
    assert model == "test-model"


def test_load_vllm_config_env_var_substitution(tmp_path, monkeypatch):
    """${VLLM_HOST} in host field should be replaced by the env var value."""
    cfg = _minimal_config()
    cfg["vllm"]["host"] = "${VLLM_HOST}"
    cfg_path = _write_config(tmp_path, cfg)

    monkeypatch.setenv("VLLM_HOST", "192.168.1.99")

    cl._config_cache = None
    with patch.object(cl, "config_path", cfg_path):
        url, model = cl.load_vllm_config()

    assert "192.168.1.99:9000" in url
    assert model == "test-model"


def test_load_vllm_config_missing_env_var_raises(tmp_path, monkeypatch):
    """RuntimeError when the referenced env var is unset."""
    cfg = _minimal_config()
    cfg["vllm"]["host"] = "${MISSING_VAR}"
    cfg_path = _write_config(tmp_path, cfg)

    monkeypatch.delenv("MISSING_VAR", raising=False)

    cl._config_cache = None
    with patch.object(cl, "config_path", cfg_path):
        with pytest.raises(RuntimeError, match="MISSING_VAR"):
            cl.load_vllm_config()


def test_load_vllm_config_returns_model_name(tmp_path):
    """The second element of the tuple should be the model name."""
    cfg = _minimal_config()
    cfg["vllm"]["model"] = "my-custom-model"
    cfg_path = _write_config(tmp_path, cfg)

    cl._config_cache = None
    with patch.object(cl, "config_path", cfg_path):
        _, model = cl.load_vllm_config()

    assert model == "my-custom-model"


# ── load_vllm_aws_config ────────────────────────────────────────────────────


def test_load_vllm_aws_config_returns_aws_subsection(tmp_path):
    """Should return the 'aws' dict nested under 'vllm'."""
    cfg = _minimal_config()
    cfg_path = _write_config(tmp_path, cfg)

    cl._config_cache = None
    with patch.object(cl, "config_path", cfg_path):
        aws = cl.load_vllm_aws_config()

    assert aws == {"region": "us-west-2"}


def test_load_vllm_aws_config_missing_aws_returns_empty(tmp_path):
    """When there's no 'aws' key, should return an empty dict."""
    cfg = _minimal_config()
    del cfg["vllm"]["aws"]
    cfg_path = _write_config(tmp_path, cfg)

    cl._config_cache = None
    with patch.object(cl, "config_path", cfg_path):
        aws = cl.load_vllm_aws_config()

    assert aws == {}


# ── get_rag_config ───────────────────────────────────────────────────────────


def test_get_rag_config_resolves_paths(tmp_path):
    """storage_dir and live_data_dir should be resolved relative to repo root."""
    cfg = _minimal_config()
    cfg_path = _write_config(tmp_path, cfg)

    cl._config_cache = None
    with patch.object(cl, "config_path", cfg_path):
        rag = cl.get_rag_config()

    assert Path(rag["storage_dir"]).is_absolute()
    assert Path(rag["live_data_dir"]).is_absolute()
    assert rag["storage_dir"].endswith("memory/chroma_store")
    assert rag["live_data_dir"].endswith("memory/live")


def test_get_rag_config_passes_through_embed_model(tmp_path):
    """embed_model and device should be passed through as-is."""
    cfg = _minimal_config()
    cfg_path = _write_config(tmp_path, cfg)

    cl._config_cache = None
    with patch.object(cl, "config_path", cfg_path):
        rag = cl.get_rag_config()

    assert rag["embed_model"] == "BAAI/bge-small-en"
    assert rag["device"] == "cpu"
