"""Tests for model.py: model factory, forward pass shapes, config validation."""

import pytest
import torch
import torch.testing
from model import GPTLM, HFGPT2LM, ModelConfig, TitansLM, build_model


class TestModelConfig:
    """ModelConfig dataclass defaults and construction."""

    def test_default_variant(self):
        cfg = ModelConfig(vocab_size=100)
        assert cfg.variant == "mac"

    def test_custom_fields(self):
        cfg = ModelConfig(variant="gpt", vocab_size=50000, dim=1024, depth=24, heads=16)
        assert cfg.dim == 1024
        assert cfg.depth == 24
        assert cfg.heads == 16


class TestBuildModel:
    """build_model factory function."""

    def test_gpt_variant(self, tiny_gpt_config):
        model = build_model(tiny_gpt_config)
        assert isinstance(model, GPTLM)

    def test_mac_variant(self, tiny_mac_config):
        model = build_model(tiny_mac_config)
        assert isinstance(model, TitansLM)

    def test_hf_gpt2_variant(self):
        cfg = ModelConfig(variant="hf_gpt2", vocab_size=256, dim=64, depth=2, heads=4)
        model = build_model(cfg)
        assert isinstance(model, HFGPT2LM)

    def test_invalid_variant_raises(self):
        cfg = ModelConfig(variant="nonexistent", vocab_size=256)
        with pytest.raises(ValueError, match="Unsupported variant"):
            build_model(cfg)


class TestGPTLMForward:
    """GPTLM forward pass shape and behavior checks."""

    def test_output_shape(self, tiny_gpt_config):
        model = build_model(tiny_gpt_config)
        x = torch.randint(0, tiny_gpt_config.vocab_size, (2, 32))
        logits = model(x, return_loss=False)
        assert logits.shape == (2, 32, tiny_gpt_config.vocab_size)

    def test_output_dtype_float(self, tiny_gpt_config):
        model = build_model(tiny_gpt_config)
        x = torch.randint(0, tiny_gpt_config.vocab_size, (1, 16))
        logits = model(x, return_loss=False)
        assert logits.dtype == torch.float32

    def test_batch_size_one(self, tiny_gpt_config):
        model = build_model(tiny_gpt_config)
        x = torch.randint(0, tiny_gpt_config.vocab_size, (1, 8))
        logits = model(x, return_loss=False)
        assert logits.shape == (1, 8, tiny_gpt_config.vocab_size)

    def test_max_seq_len(self, tiny_gpt_config):
        model = build_model(tiny_gpt_config)
        x = torch.randint(0, tiny_gpt_config.vocab_size, (1, tiny_gpt_config.max_seq_len))
        logits = model(x, return_loss=False)
        assert logits.shape[1] == tiny_gpt_config.max_seq_len

    def test_exceeds_max_seq_len_raises(self, tiny_gpt_config):
        model = build_model(tiny_gpt_config)
        x = torch.randint(0, tiny_gpt_config.vocab_size, (1, tiny_gpt_config.max_seq_len + 10))
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            model(x, return_loss=False)

    def test_gradient_flows(self, tiny_gpt_config):
        model = build_model(tiny_gpt_config)
        x = torch.randint(0, tiny_gpt_config.vocab_size, (2, 16))
        y = torch.randint(0, tiny_gpt_config.vocab_size, (2, 16))
        logits = model(x, return_loss=False)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, "No gradients flowed through the model"

    def test_deterministic_with_seed(self, tiny_gpt_config):
        x = torch.randint(0, tiny_gpt_config.vocab_size, (1, 16))

        torch.manual_seed(42)
        m1 = build_model(tiny_gpt_config)
        out1 = m1(x, return_loss=False)

        torch.manual_seed(42)
        m2 = build_model(tiny_gpt_config)
        out2 = m2(x, return_loss=False)

        torch.testing.assert_close(out1, out2)


class TestHFGPT2LMForward:
    """HFGPT2LM forward pass shape checks."""

    def test_output_shape(self):
        cfg = ModelConfig(
            variant="hf_gpt2", vocab_size=256, dim=64, depth=2, heads=4, max_seq_len=128
        )
        model = build_model(cfg)
        x = torch.randint(0, 256, (2, 32))
        logits = model(x, return_loss=False)
        assert logits.shape == (2, 32, 256)


class TestModelParameterCount:
    """Sanity check that model parameter counts scale as expected."""

    def test_deeper_model_has_more_params(self):
        small = build_model(
            ModelConfig(variant="gpt", vocab_size=256, dim=64, depth=2, heads=4, max_seq_len=64)
        )
        large = build_model(
            ModelConfig(variant="gpt", vocab_size=256, dim=64, depth=6, heads=4, max_seq_len=64)
        )
        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())
        assert large_params > small_params

    def test_wider_model_has_more_params(self):
        narrow = build_model(
            ModelConfig(variant="gpt", vocab_size=256, dim=64, depth=2, heads=4, max_seq_len=64)
        )
        wide = build_model(
            ModelConfig(variant="gpt", vocab_size=256, dim=128, depth=2, heads=4, max_seq_len=64)
        )
        narrow_params = sum(p.numel() for p in narrow.parameters())
        wide_params = sum(p.numel() for p in wide.parameters())
        assert wide_params > narrow_params
