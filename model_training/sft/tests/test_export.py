"""Tests for export.py — merge a LoRA adapter into its base → standalone HF model.

Wiring-only by default; the real peft merge runs only under SFT_RUN_SMOKE in the
venv (see the integration test at the bottom).
"""

import json
import os

import pytest
from model_training.sft import export as export_mod


def _write_adapter(tmp_path, base="Qwen/Qwen2.5-0.5B-Instruct"):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    (d / "adapter_model.safetensors").write_bytes(b"")  # presence only
    return d


class TestReadBaseModel:
    def test_reads_base_model_field(self, tmp_path):
        d = _write_adapter(tmp_path, base="some/model")
        assert export_mod.read_base_model(d) == "some/model"

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(ValueError, match="adapter_config.json"):
            export_mod.read_base_model(tmp_path)


class _FakeSaveable:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, output_dir):
        self.saved_to = output_dir


class TestMergeAdapter:
    def test_missing_adapter_config_raises(self, tmp_path):
        with pytest.raises(ValueError, match="adapter_config.json"):
            export_mod.merge_adapter(tmp_path, tmp_path / "out")

    def test_orchestration_saves_model_and_tokenizer(self, tmp_path, monkeypatch):
        adapter = _write_adapter(tmp_path, base="base/x")
        out = tmp_path / "merged"
        fake_model, fake_tok = _FakeSaveable(), _FakeSaveable()

        captured = {}

        def fake_load_and_merge(base_model, adapter_dir):
            captured["base_model"] = base_model
            captured["adapter_dir"] = str(adapter_dir)
            return fake_model, fake_tok

        monkeypatch.setattr(export_mod, "_load_and_merge", fake_load_and_merge)

        result = export_mod.merge_adapter(adapter, out)
        assert result == str(out)
        assert fake_model.saved_to == str(out)
        assert fake_tok.saved_to == str(out)
        assert captured["base_model"] == "base/x"  # resolved from adapter_config

    def test_explicit_base_model_overrides_config(self, tmp_path, monkeypatch):
        adapter = _write_adapter(tmp_path, base="base/from-config")
        out = tmp_path / "merged"
        captured = {}
        monkeypatch.setattr(
            export_mod,
            "_load_and_merge",
            lambda base_model, adapter_dir: captured.update(base_model=base_model)
            or (_FakeSaveable(), _FakeSaveable()),
        )
        export_mod.merge_adapter(adapter, out, base_model="base/override")
        assert captured["base_model"] == "base/override"


@pytest.mark.slow
@pytest.mark.integration
def test_real_merge_from_smoke_adapter(tmp_path):
    """Real peft merge — opt-in (SFT_RUN_SMOKE=1, venv with peft/transformers)."""
    if not os.environ.get("SFT_RUN_SMOKE"):
        pytest.skip("set SFT_RUN_SMOKE=1 to run the real adapter merge")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")

    # Train a tiny adapter, then merge it.
    import json as _json

    from model_training.sft.config import SFTConfig
    from model_training.sft.train import train

    data = tmp_path / "smoke.jsonl"
    ex = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    data.write_text("\n".join(_json.dumps(ex) for _ in range(8)) + "\n")
    adapter_dir = tmp_path / "adapter"
    cfg = SFTConfig.from_dict(
        {
            "model": {"base_model": "Qwen/Qwen2.5-0.5B-Instruct", "dtype": "float32"},
            "data": {"train_path": str(data), "max_seq_len": 256},
            "lora": {"r": 8, "alpha": 16},
            "training": {"output_dir": str(adapter_dir), "epochs": 1, "save_steps": 100},
        }
    )
    train(cfg)

    merged = tmp_path / "merged"
    export_mod.merge_adapter(adapter_dir, merged)
    assert (merged / "config.json").exists()
    assert any(merged.glob("*.safetensors"))
