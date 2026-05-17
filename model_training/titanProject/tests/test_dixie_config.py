"""Config invariants for Dixie Flatline SFT (no torch import)."""

from pathlib import Path

import yaml


def test_dixie_mistral_full_requests_gradient_checkpointing() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "config_dixie_mistral_full.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["train"]["gradient_checkpointing"] is True
