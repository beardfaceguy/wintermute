"""Tests for hf_utils.py — HuggingFace model loading, LoRA, checkpointing, and tokenizer setup.

All heavy dependencies (transformers, peft, bitsandbytes) are mocked so tests
run on CPU without downloading real models.
"""

import types
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

from hf_utils import (
    apply_lora,
    get_hf_tokenizer,
    HFModelWrapper,
    load_hf_checkpoint,
    load_hf_model,
    save_hf_checkpoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(*, has_chat_template=False, has_pad_token=True):
    """Build a MagicMock that behaves like an HF AutoTokenizer."""
    tok = MagicMock()
    tok.pad_token = "▁" if has_pad_token else None
    tok.eos_token = "</s>"
    tok.pad_token_id = 0 if has_pad_token else None
    tok.eos_token_id = 2
    tok.name_or_path = "mock-model"
    tok.chat_template = "{% for m in messages %}...{% endfor %}" if has_chat_template else None
    tok.get_vocab.return_value = {"<s>": 0, "</s>": 2}
    tok.add_special_tokens.return_value = 2
    return tok


def _make_mock_model(*, is_4bit=False):
    """Build a MagicMock that behaves like an HF AutoModelForCausalLM."""
    model = MagicMock()
    model.is_loaded_in_4bit = is_4bit
    params = [torch.randn(4, 4, requires_grad=True) for _ in range(3)]
    model.parameters.return_value = params
    model.save_pretrained = MagicMock()
    return model


# ===========================================================================
# get_hf_tokenizer
# ===========================================================================

class TestGetHFTokenizer:
    @patch("hf_utils.AutoTokenizer", create=True)
    def test_returns_tokenizer_with_pad_token(self, _mock_cls):
        tok = _make_mock_tokenizer(has_pad_token=False)
        _mock_cls.from_pretrained.return_value = tok

        with patch.dict("sys.modules", {"transformers": MagicMock(AutoTokenizer=_mock_cls)}):
            result = get_hf_tokenizer("mock-model")

        assert result.pad_token == result.eos_token

    @patch("hf_utils.AutoTokenizer", create=True)
    def test_assigns_chatml_when_no_chat_template(self, _mock_cls):
        tok = _make_mock_tokenizer(has_chat_template=False)
        _mock_cls.from_pretrained.return_value = tok

        with patch.dict("sys.modules", {"transformers": MagicMock(AutoTokenizer=_mock_cls)}):
            result = get_hf_tokenizer("mock-model")

        assert result.chat_template is not None
        assert "<|im_start|>" in result.chat_template
        assert "<|im_end|>" in result.chat_template

    @patch("hf_utils.AutoTokenizer", create=True)
    def test_chatml_template_has_expected_markers(self, _mock_cls):
        tok = _make_mock_tokenizer(has_chat_template=False)
        _mock_cls.from_pretrained.return_value = tok

        with patch.dict("sys.modules", {"transformers": MagicMock(AutoTokenizer=_mock_cls)}):
            result = get_hf_tokenizer("mock-model")

        tpl = result.chat_template
        assert "system" in tpl
        assert "user" in tpl
        assert "assistant" in tpl
        assert "add_generation_prompt" in tpl

    @patch("hf_utils.AutoTokenizer", create=True)
    def test_adds_special_tokens_when_missing(self, _mock_cls):
        tok = _make_mock_tokenizer(has_chat_template=False)
        _mock_cls.from_pretrained.return_value = tok

        with patch.dict("sys.modules", {"transformers": MagicMock(AutoTokenizer=_mock_cls)}):
            get_hf_tokenizer("mock-model")

        tok.add_special_tokens.assert_called()
        added = [
            call.kwargs.get("additional_special_tokens")
            or call.args[0].get("additional_special_tokens", [])
            for call in tok.add_special_tokens.call_args_list
        ]
        flat = [t for sub in added for t in sub]
        assert "<|im_start|>" in flat
        assert "<|im_end|>" in flat

    @patch("hf_utils.AutoTokenizer", create=True)
    def test_preserves_existing_chat_template(self, _mock_cls):
        original_tpl = "{% for m in messages %}CUSTOM{% endfor %}"
        tok = _make_mock_tokenizer(has_chat_template=True)
        tok.chat_template = original_tpl
        _mock_cls.from_pretrained.return_value = tok

        with patch.dict("sys.modules", {"transformers": MagicMock(AutoTokenizer=_mock_cls)}):
            result = get_hf_tokenizer("mock-model")

        assert result.chat_template == original_tpl

    @patch("hf_utils.AutoTokenizer", create=True)
    def test_skips_special_tokens_when_already_in_vocab(self, _mock_cls):
        tok = _make_mock_tokenizer(has_chat_template=False)
        tok.get_vocab.return_value = {
            "<s>": 0, "</s>": 2,
            "<|im_start|>": 100, "<|im_end|>": 101,
        }
        _mock_cls.from_pretrained.return_value = tok

        with patch.dict("sys.modules", {"transformers": MagicMock(AutoTokenizer=_mock_cls)}):
            get_hf_tokenizer("mock-model")

        tok.add_special_tokens.assert_not_called()

    @patch("hf_utils.AutoTokenizer", create=True)
    def test_pad_token_already_set_not_overwritten(self, _mock_cls):
        tok = _make_mock_tokenizer(has_pad_token=True)
        original_pad = tok.pad_token
        _mock_cls.from_pretrained.return_value = tok

        with patch.dict("sys.modules", {"transformers": MagicMock(AutoTokenizer=_mock_cls)}):
            result = get_hf_tokenizer("mock-model")

        assert result.pad_token == original_pad


# ===========================================================================
# HFModelWrapper
# ===========================================================================

class TestHFModelWrapper:
    def test_wraps_model_and_tokenizer(self):
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        wrapper = HFModelWrapper(model, tokenizer, is_lora=True)

        assert wrapper.model is model
        assert wrapper.tokenizer is tokenizer
        assert wrapper.is_lora is True

    def test_hf_model_property(self):
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        wrapper = HFModelWrapper(model, tokenizer)

        assert wrapper.hf_model is model

    def test_hf_tokenizer_property(self):
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        wrapper = HFModelWrapper(model, tokenizer)

        assert wrapper.hf_tokenizer is tokenizer

    def test_is_lora_defaults_false(self):
        wrapper = HFModelWrapper(_make_mock_model(), _make_mock_tokenizer())
        assert wrapper.is_lora is False


# ===========================================================================
# load_hf_model
# ===========================================================================

class TestLoadHFModel:
    @patch("hf_utils.BitsAndBytesConfig", create=True)
    @patch("hf_utils.AutoModelForCausalLM", create=True)
    def test_loads_model_with_default_dtype(self, mock_auto, _mock_bnb):
        mock_model = _make_mock_model()
        mock_auto.from_pretrained.return_value = mock_model

        transformers_mod = MagicMock()
        transformers_mod.AutoModelForCausalLM = mock_auto
        transformers_mod.BitsAndBytesConfig = _mock_bnb

        with patch.dict("sys.modules", {"transformers": transformers_mod}):
            result = load_hf_model("mock-id")

        assert result is mock_model
        call_kwargs = mock_auto.from_pretrained.call_args[1]
        assert call_kwargs["torch_dtype"] == torch.float16

    @patch("hf_utils.BitsAndBytesConfig", create=True)
    @patch("hf_utils.AutoModelForCausalLM", create=True)
    def test_loads_model_with_explicit_dtype(self, mock_auto, _mock_bnb):
        mock_auto.from_pretrained.return_value = _make_mock_model()

        transformers_mod = MagicMock()
        transformers_mod.AutoModelForCausalLM = mock_auto
        transformers_mod.BitsAndBytesConfig = _mock_bnb

        with patch.dict("sys.modules", {"transformers": transformers_mod}):
            load_hf_model("mock-id", torch_dtype=torch.bfloat16)

        call_kwargs = mock_auto.from_pretrained.call_args[1]
        assert call_kwargs["torch_dtype"] == torch.bfloat16

    @patch("hf_utils.BitsAndBytesConfig", create=True)
    @patch("hf_utils.AutoModelForCausalLM", create=True)
    def test_qlora_mode_creates_quantization_config(self, mock_auto, mock_bnb_config):
        mock_auto.from_pretrained.return_value = _make_mock_model()

        transformers_mod = MagicMock()
        transformers_mod.AutoModelForCausalLM = mock_auto
        transformers_mod.BitsAndBytesConfig = mock_bnb_config

        bnb_mod = MagicMock()

        with patch.dict("sys.modules", {
            "transformers": transformers_mod,
            "bitsandbytes": bnb_mod,
        }):
            load_hf_model("mock-id", load_in_4bit=True)

        mock_bnb_config.assert_called_once()
        bnb_call_kwargs = mock_bnb_config.call_args[1]
        assert bnb_call_kwargs["load_in_4bit"] is True
        assert bnb_call_kwargs["bnb_4bit_use_double_quant"] is True

    @patch("hf_utils.BitsAndBytesConfig", create=True)
    @patch("hf_utils.AutoModelForCausalLM", create=True)
    def test_qlora_bfloat16_compute_dtype(self, mock_auto, mock_bnb_config):
        mock_auto.from_pretrained.return_value = _make_mock_model()

        transformers_mod = MagicMock()
        transformers_mod.AutoModelForCausalLM = mock_auto
        transformers_mod.BitsAndBytesConfig = mock_bnb_config

        with patch.dict("sys.modules", {
            "transformers": transformers_mod,
            "bitsandbytes": MagicMock(),
        }):
            load_hf_model("mock-id", load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")

        bnb_call_kwargs = mock_bnb_config.call_args[1]
        assert bnb_call_kwargs["bnb_4bit_compute_dtype"] == torch.bfloat16

    @patch("hf_utils.BitsAndBytesConfig", create=True)
    @patch("hf_utils.AutoModelForCausalLM", create=True)
    def test_qlora_missing_bitsandbytes_raises(self, mock_auto, mock_bnb_config):
        transformers_mod = MagicMock()
        transformers_mod.AutoModelForCausalLM = mock_auto
        transformers_mod.BitsAndBytesConfig = mock_bnb_config

        with patch.dict("sys.modules", {
            "transformers": transformers_mod,
            "bitsandbytes": None,
        }):
            with pytest.raises(ImportError, match="bitsandbytes"):
                load_hf_model("mock-id", load_in_4bit=True)

    @patch("hf_utils.BitsAndBytesConfig", create=True)
    @patch("hf_utils.AutoModelForCausalLM", create=True)
    def test_device_map_passed_through(self, mock_auto, _mock_bnb):
        mock_auto.from_pretrained.return_value = _make_mock_model()

        transformers_mod = MagicMock()
        transformers_mod.AutoModelForCausalLM = mock_auto
        transformers_mod.BitsAndBytesConfig = _mock_bnb

        with patch.dict("sys.modules", {"transformers": transformers_mod}):
            load_hf_model("mock-id", device_map=None)

        call_kwargs = mock_auto.from_pretrained.call_args[1]
        assert "device_map" not in call_kwargs

    @patch("hf_utils.BitsAndBytesConfig", create=True)
    @patch("hf_utils.AutoModelForCausalLM", create=True)
    def test_attn_implementation_passed_through(self, mock_auto, _mock_bnb):
        mock_auto.from_pretrained.return_value = _make_mock_model()

        transformers_mod = MagicMock()
        transformers_mod.AutoModelForCausalLM = mock_auto
        transformers_mod.BitsAndBytesConfig = _mock_bnb

        with patch.dict("sys.modules", {"transformers": transformers_mod}):
            load_hf_model("mock-id", attn_implementation="flash_attention_2")

        call_kwargs = mock_auto.from_pretrained.call_args[1]
        assert call_kwargs["attn_implementation"] == "flash_attention_2"


# ===========================================================================
# apply_lora
# ===========================================================================

class TestApplyLoRA:
    def test_applies_lora_config(self):
        mock_model = _make_mock_model()
        mock_peft_model = _make_mock_model()

        mock_lora_config = MagicMock()
        mock_get_peft = MagicMock(return_value=mock_peft_model)
        mock_prepare = MagicMock(return_value=mock_model)

        peft_mod = MagicMock()
        peft_mod.LoraConfig = mock_lora_config
        peft_mod.get_peft_model = mock_get_peft
        peft_mod.prepare_model_for_kbit_training = mock_prepare
        peft_mod.TaskType.CAUSAL_LM = "CAUSAL_LM"

        with patch.dict("sys.modules", {"peft": peft_mod}):
            result = apply_lora(mock_model, rank=32, alpha=64, dropout=0.1)

        assert result is mock_peft_model
        mock_lora_config.assert_called_once()
        cfg_kwargs = mock_lora_config.call_args[1]
        assert cfg_kwargs["r"] == 32
        assert cfg_kwargs["lora_alpha"] == 64
        assert cfg_kwargs["lora_dropout"] == 0.1

    def test_target_modules_default_all_linear(self):
        mock_model = _make_mock_model()

        mock_lora_config = MagicMock()
        peft_mod = MagicMock()
        peft_mod.LoraConfig = mock_lora_config
        peft_mod.get_peft_model = MagicMock(return_value=mock_model)
        peft_mod.prepare_model_for_kbit_training = MagicMock()
        peft_mod.TaskType.CAUSAL_LM = "CAUSAL_LM"

        with patch.dict("sys.modules", {"peft": peft_mod}):
            apply_lora(mock_model)

        cfg_kwargs = mock_lora_config.call_args[1]
        assert cfg_kwargs["target_modules"] == "all-linear"

    def test_custom_target_modules(self):
        mock_model = _make_mock_model()
        targets = ["q_proj", "v_proj"]

        mock_lora_config = MagicMock()
        peft_mod = MagicMock()
        peft_mod.LoraConfig = mock_lora_config
        peft_mod.get_peft_model = MagicMock(return_value=mock_model)
        peft_mod.prepare_model_for_kbit_training = MagicMock()
        peft_mod.TaskType.CAUSAL_LM = "CAUSAL_LM"

        with patch.dict("sys.modules", {"peft": peft_mod}):
            apply_lora(mock_model, target_modules=targets)

        cfg_kwargs = mock_lora_config.call_args[1]
        assert cfg_kwargs["target_modules"] == targets

    def test_4bit_model_gets_prepared(self):
        mock_model = _make_mock_model(is_4bit=True)
        prepared_model = _make_mock_model()

        mock_prepare = MagicMock(return_value=prepared_model)
        peft_mod = MagicMock()
        peft_mod.LoraConfig = MagicMock()
        peft_mod.get_peft_model = MagicMock(return_value=prepared_model)
        peft_mod.prepare_model_for_kbit_training = mock_prepare
        peft_mod.TaskType.CAUSAL_LM = "CAUSAL_LM"

        with patch.dict("sys.modules", {"peft": peft_mod}):
            apply_lora(mock_model)

        mock_prepare.assert_called_once_with(mock_model)

    def test_non_4bit_model_skips_prepare(self):
        mock_model = _make_mock_model(is_4bit=False)

        mock_prepare = MagicMock()
        peft_mod = MagicMock()
        peft_mod.LoraConfig = MagicMock()
        peft_mod.get_peft_model = MagicMock(return_value=mock_model)
        peft_mod.prepare_model_for_kbit_training = mock_prepare
        peft_mod.TaskType.CAUSAL_LM = "CAUSAL_LM"

        with patch.dict("sys.modules", {"peft": peft_mod}):
            apply_lora(mock_model)

        mock_prepare.assert_not_called()

    def test_log_fn_called_with_stats(self):
        p1 = torch.randn(10, requires_grad=True)
        p2 = torch.randn(5, requires_grad=False)
        mock_model = MagicMock()
        mock_model.is_loaded_in_4bit = False
        mock_model.parameters.return_value = [p1, p2]

        peft_mod = MagicMock()
        peft_mod.LoraConfig = MagicMock()
        peft_mod.get_peft_model = MagicMock(return_value=mock_model)
        peft_mod.prepare_model_for_kbit_training = MagicMock()
        peft_mod.TaskType.CAUSAL_LM = "CAUSAL_LM"

        log_messages = []

        with patch.dict("sys.modules", {"peft": peft_mod}):
            apply_lora(mock_model, log_fn=log_messages.append)

        assert len(log_messages) == 1
        assert "[lora]" in log_messages[0]
        assert "trainable=" in log_messages[0]

    def test_bias_set_to_none(self):
        mock_model = _make_mock_model()
        mock_lora_config = MagicMock()
        peft_mod = MagicMock()
        peft_mod.LoraConfig = mock_lora_config
        peft_mod.get_peft_model = MagicMock(return_value=mock_model)
        peft_mod.prepare_model_for_kbit_training = MagicMock()
        peft_mod.TaskType.CAUSAL_LM = "CAUSAL_LM"

        with patch.dict("sys.modules", {"peft": peft_mod}):
            apply_lora(mock_model)

        cfg_kwargs = mock_lora_config.call_args[1]
        assert cfg_kwargs["bias"] == "none"


# ===========================================================================
# save_hf_checkpoint / load_hf_checkpoint
# ===========================================================================

class TestSaveHFCheckpoint:
    def test_creates_directory(self, tmp_path):
        ckpt_dir = tmp_path / "new_ckpt" / "step_100"
        model = _make_mock_model()
        save_hf_checkpoint(model, ckpt_dir, step=100)

        assert ckpt_dir.exists()

    def test_saves_training_state(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        model = _make_mock_model()
        save_hf_checkpoint(model, ckpt_dir, step=42)

        state = torch.load(ckpt_dir / "training_state.pt", weights_only=True)
        assert state["step"] == 42

    def test_full_model_save(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        model = _make_mock_model()
        del model.module  # ensure no DDP-style unwrapping
        save_hf_checkpoint(model, ckpt_dir, is_lora=False)

        model.save_pretrained.assert_called_once_with(ckpt_dir)

    def test_lora_adapter_save(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        model = _make_mock_model()
        save_hf_checkpoint(model, ckpt_dir, is_lora=True)

        model.save_pretrained.assert_called_once_with(ckpt_dir)

    def test_saves_tokenizer_when_provided(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        save_hf_checkpoint(model, ckpt_dir, tokenizer=tokenizer)

        tokenizer.save_pretrained.assert_called_once_with(ckpt_dir)

    def test_skips_tokenizer_when_none(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        model = _make_mock_model()
        save_hf_checkpoint(model, ckpt_dir, tokenizer=None)
        # no assertion needed — just verifying no crash

    def test_unwraps_ddp_module(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        inner_model = _make_mock_model()
        ddp_model = MagicMock()
        ddp_model.module = inner_model

        save_hf_checkpoint(ddp_model, ckpt_dir, is_lora=False)

        inner_model.save_pretrained.assert_called_once_with(ckpt_dir)

    def test_log_fn_called_for_lora(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        model = _make_mock_model()
        logs = []
        save_hf_checkpoint(model, ckpt_dir, is_lora=True, log_fn=logs.append)

        assert any("LoRA adapter" in m for m in logs)

    def test_log_fn_called_for_full_model(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        model = _make_mock_model()
        logs = []
        save_hf_checkpoint(model, ckpt_dir, is_lora=False, log_fn=logs.append)

        assert any("full model" in m for m in logs)


class TestLoadHFCheckpoint:
    @patch("hf_utils.load_hf_model")
    def test_loads_base_model_only(self, mock_load):
        mock_model = _make_mock_model()
        mock_load.return_value = mock_model

        result = load_hf_checkpoint("mock-id")

        assert result is mock_model
        mock_load.assert_called_once()

    @patch("hf_utils.load_hf_model")
    def test_loads_with_adapter(self, mock_load):
        mock_base = _make_mock_model()
        mock_load.return_value = mock_base

        mock_peft_model = _make_mock_model()
        peft_mod = MagicMock()
        peft_mod.PeftModel.from_pretrained.return_value = mock_peft_model

        with patch.dict("sys.modules", {"peft": peft_mod}):
            result = load_hf_checkpoint("mock-id", adapter_path="/tmp/adapter")

        peft_mod.PeftModel.from_pretrained.assert_called_once_with(
            mock_base, "/tmp/adapter"
        )
        assert result is mock_peft_model

    @patch("hf_utils.load_hf_model")
    def test_merge_and_unload(self, mock_load):
        mock_base = _make_mock_model()
        mock_load.return_value = mock_base

        mock_merged = _make_mock_model()
        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged

        peft_mod = MagicMock()
        peft_mod.PeftModel.from_pretrained.return_value = mock_peft_model

        with patch.dict("sys.modules", {"peft": peft_mod}):
            result = load_hf_checkpoint(
                "mock-id", adapter_path="/tmp/adapter", merge=True
            )

        mock_peft_model.merge_and_unload.assert_called_once()
        assert result is mock_merged

    @patch("hf_utils.load_hf_model")
    def test_no_merge_without_adapter(self, mock_load):
        mock_model = _make_mock_model()
        mock_load.return_value = mock_model

        result = load_hf_checkpoint("mock-id", merge=True)
        assert result is mock_model

    @patch("hf_utils.load_hf_model")
    def test_passes_dtype_and_device_map(self, mock_load):
        mock_load.return_value = _make_mock_model()

        load_hf_checkpoint(
            "mock-id", torch_dtype=torch.bfloat16, device_map=None
        )

        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["torch_dtype"] == torch.bfloat16
        assert call_kwargs["device_map"] is None

    @patch("hf_utils.load_hf_model")
    def test_passes_trust_remote_code(self, mock_load):
        mock_load.return_value = _make_mock_model()

        load_hf_checkpoint("mock-id", trust_remote_code=True)

        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["trust_remote_code"] is True
