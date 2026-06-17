# config_loader.py
import json
import os
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlunparse

config_path = (Path(__file__).parent / "../config/shared_api_config.json").resolve()
_config_cache = None


class RagConfig(TypedDict):
    storage_dir: str
    live_data_dir: str
    embed_model: str
    device: str


def _load_config() -> dict[str, dict[str, Any]]:
    global _config_cache
    if _config_cache is None:
        with open(config_path) as f:
            _config_cache = json.load(f)
    return _config_cache


def load_vllm_config() -> tuple[str, str]:
    """Load vLLM endpoint URL and model name.

    The host field supports ${VLLM_HOST} env var substitution so the
    same config works across environments. Set VLLM_HOST to the public
    IP of the running EC2 instance.
    """
    raw = _load_config()["vllm"]
    host = raw["host"]
    if host.startswith("${") and host.endswith("}"):
        env_var = host[2:-1]
        host = os.environ.get(env_var, "")
        if not host:
            raise RuntimeError(
                f"vLLM host requires env var {env_var} — "
                f"set it to the public IP of the running EC2 instance"
            )
    url = urlunparse((raw["scheme"], f"{host}:{raw['port']}", raw["path"], "", "", ""))
    return url, raw["model"]


def load_vllm_aws_config() -> dict[str, Any]:
    """Load AWS infrastructure config for launching/managing the vLLM instance."""
    return _load_config()["vllm"].get("aws", {})


def load_inference_config() -> dict[str, Any]:
    """Load LLM inference parameters (max_tokens, temperature, etc.)."""
    return (
        _load_config()
        .get("vllm", {})
        .get(
            "inference",
            {
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "frequency_penalty": 0.5,
            },
        )
    )


def load_agents_config() -> dict[str, Any]:
    """Load agent runner defaults."""
    return _load_config().get("agents", {})


def load_mcp_servers_config() -> dict[str, Any]:
    """Load MCP server port/URL config."""
    return _load_config().get("mcp_servers", {})


def load_mcp_memory_config() -> dict[str, Any]:
    """Load mcp-memory config."""
    return _load_config().get("mcp_memory", {})


def get_rag_config() -> RagConfig:
    raw = _load_config()["rag"]
    base_path = Path(__file__).resolve().parent.parent  # wintermute root
    return {
        "storage_dir": str((base_path / raw["storage_dir"]).resolve()),
        "live_data_dir": str((base_path / raw["live_data_dir"]).resolve()),
        "embed_model": raw["embed_model"],
        "device": raw["device"],
    }
