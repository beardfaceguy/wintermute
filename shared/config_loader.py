# config_loader.py
import json
from pathlib import Path
from urllib.parse import urlunparse

def load_vllm_config():
    config_path = Path(__file__).parent / "../config/shared_api_config.json"
    with open(config_path) as f:
        raw = json.load(f)["vllm"]
        url = urlunparse((
            raw["scheme"],
            f"{raw['host']}:{raw['port']}",
            raw["path"], '', '', ''
        ))
        return url, raw["model"]
