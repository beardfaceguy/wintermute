# utils/config_utils.py

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.setup_path import extend_path  # noqa: E402

extend_path()
from shared.config_loader import get_rag_config, load_vllm_config  # noqa: E402

rag_config = get_rag_config()
_vllm_url, _llm_model_name = load_vllm_config()


def get_vllm_url():
    return _vllm_url


def get_llm_model_name():
    return _llm_model_name


def get_storage_dir():
    return rag_config["storage_dir"]


def get_live_data_dir():
    return rag_config["live_data_dir"]


def get_embed_model_name():
    return rag_config["embed_model"]


def get_embed_device():
    return rag_config["device"]


def verify_config():
    assert get_vllm_url().startswith("http"), f"Invalid VLLM URL: {get_vllm_url()}"
    assert get_llm_model_name()
    assert Path(get_storage_dir()).exists(), f"Storage dir missing: {get_storage_dir()}"
    assert get_embed_model_name()
    assert get_embed_device() in ["cpu", "cuda"]
