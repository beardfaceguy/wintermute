# config_loader.py
import json
from pathlib import Path
from urllib.parse import urlunparse
config_path = Path(__file__).parent / "../config/shared_api_config.json"
_config_cache = None

def _load_config():
    global _config_cache
    if _config_cache is None:
        with open(config_path) as f:
            _config_cache = json.load(f)
    return _config_cache

def load_vllm_config():
    raw = _load_config()["vllm"]
    url = urlunparse((
        raw["scheme"],
        f"{raw['host']}:{raw['port']}",
        raw["path"], '', '', ''
    ))
    return url, raw["model"]
    
def get_rag_config():
    raw = _load_config()["rag"]
    base_path = Path(__file__).resolve().parent.parent  # wintermute root
    return {
        "storage_dir": str((base_path / raw["storage_dir"]).resolve()),
        "live_data_dir": str((base_path / raw["live_data_dir"]).resolve()),
        "embed_model": raw["embed_model"],
        "device": raw["device"]
    }