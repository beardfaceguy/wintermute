"""Real end-to-end SFT smoke test (opt-in, heavy).

Unlike the wiring tests, this actually downloads Qwen2.5-0.5B and runs a real
LoRA fine-tune via TRL. It is gated three ways so it never runs in default CI:
  - skips unless SFT_RUN_SMOKE=1 is set,
  - skips if trl/peft/transformers aren't importable (e.g. system Python 3.14),
  - marked slow + integration.

Run it from the pinned venv:
    SFT_RUN_SMOKE=1 model_training/sft/.venv/bin/python -m pytest \
        model_training/sft/tests/test_smoke_integration.py -v
"""

import json
import os

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def test_real_tiny_sft_writes_checkpoint(tmp_path):
    if not os.environ.get("SFT_RUN_SMOKE"):
        pytest.skip("set SFT_RUN_SMOKE=1 to run the real tiny-model fine-tune")
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")

    from model_training.sft.config import SFTConfig
    from model_training.sft.train import train

    data = tmp_path / "smoke.jsonl"
    ex = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    data.write_text("\n".join(json.dumps(ex) for _ in range(8)) + "\n")

    out = tmp_path / "out"
    cfg = SFTConfig.from_dict(
        {
            "model": {"base_model": "Qwen/Qwen2.5-0.5B-Instruct", "dtype": "float32"},
            "data": {"train_path": str(data), "max_seq_len": 256},
            "lora": {"r": 8, "alpha": 16},
            "training": {
                "output_dir": str(out),
                "epochs": 1,
                "per_device_batch_size": 1,
                "grad_accum_steps": 1,
                "logging_steps": 1,
                "save_steps": 100,
            },
        }
    )

    result = train(cfg)
    assert result == str(out)
    assert (out / "adapter_config.json").exists()  # LoRA adapter was saved
