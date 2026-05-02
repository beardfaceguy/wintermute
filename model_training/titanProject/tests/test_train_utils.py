"""Tests for training utilities: LR schedules, path resolution, hashing, checkpoints."""

import hashlib
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.testing
from torch.optim import AdamW

from model import ModelConfig, build_model
from train_utils import cosine_lr, resolve_path, sha256_file, TokenizerAdapter, get_tokenizer


class TestCosineLR:
    """Cosine learning rate schedule with warmup."""

    def test_zero_at_step_zero(self):
        assert cosine_lr(0, warmup=100, max_steps=1000, base_lr=0.001) == 0.0

    def test_linear_warmup(self):
        lr = cosine_lr(50, warmup=100, max_steps=1000, base_lr=0.001)
        assert lr == pytest.approx(0.0005, abs=1e-8)

    def test_peak_at_warmup(self):
        lr = cosine_lr(100, warmup=100, max_steps=1000, base_lr=0.001)
        assert lr == pytest.approx(0.001, abs=1e-8)

    def test_decay_after_warmup(self):
        lr_peak = cosine_lr(100, warmup=100, max_steps=1000, base_lr=0.001)
        lr_later = cosine_lr(500, warmup=100, max_steps=1000, base_lr=0.001)
        assert lr_later < lr_peak

    def test_min_lr_at_max_steps(self):
        lr = cosine_lr(1000, warmup=100, max_steps=1000, base_lr=0.001, min_lr=0.0001)
        assert lr == pytest.approx(0.0001, abs=1e-8)

    def test_min_lr_zero_default(self):
        lr = cosine_lr(1000, warmup=100, max_steps=1000, base_lr=0.001)
        assert lr == pytest.approx(0.0, abs=1e-8)

    def test_midpoint_value(self):
        # At the midpoint of cosine decay, lr should be (base + min) / 2
        warmup = 100
        max_steps = 1000
        base_lr = 0.001
        min_lr = 0.0001
        midpoint = warmup + (max_steps - warmup) // 2
        lr = cosine_lr(midpoint, warmup, max_steps, base_lr, min_lr)
        expected = (base_lr + min_lr) / 2
        assert lr == pytest.approx(expected, abs=1e-6)

    def test_monotonically_decreasing_after_warmup(self):
        lrs = [cosine_lr(s, 100, 1000, 0.001, 0.0001) for s in range(100, 1001, 50)]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1] + 1e-10

    def test_always_non_negative(self):
        for step in range(0, 2000, 10):
            lr = cosine_lr(step, 100, 1000, 0.001, 0.0)
            assert lr >= 0.0

    def test_zero_warmup(self):
        lr = cosine_lr(0, warmup=0, max_steps=1000, base_lr=0.001)
        assert lr == pytest.approx(0.001, abs=1e-8)


class TestResolvePath:
    """Path resolution logic."""

    def test_absolute_path_returned_as_is(self, tmp_path):
        p = resolve_path(str(tmp_path / "some_file.txt"))
        assert p == tmp_path / "some_file.txt"

    def test_s3_path_returned_as_string(self):
        result = resolve_path("s3://my-bucket/prefix/file.txt")
        assert result == "s3://my-bucket/prefix/file.txt"

    def test_hf_path_returned_as_string(self):
        result = resolve_path("hf://gpt2")
        assert result == "hf://gpt2"


class TestSha256File:
    """File hashing."""

    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        h1 = sha256_file(f)
        h2 = sha256_file(f)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"content a")
        f2.write_bytes(b"content b")
        assert sha256_file(f1) != sha256_file(f2)

    def test_hash_length(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"data")
        assert len(sha256_file(f)) == 64  # SHA-256 hex digest


class TestTokenizerAdapter:
    """TokenizerAdapter encode/decode contract."""

    def test_encode_returns_list(self):
        adapter = TokenizerAdapter(
            encode_fn=lambda t: [ord(c) for c in t],
            decode_fn=lambda ids: "".join(chr(i) for i in ids),
            tokenizer_fingerprint="test",
            tokenizer_source_path="<test>",
            eos_id=0,
            pad_id=1,
        )
        result = adapter.encode("abc")
        assert isinstance(result, list)
        assert result == [97, 98, 99]

    def test_callable(self):
        adapter = TokenizerAdapter(
            encode_fn=lambda t: [ord(c) for c in t],
            decode_fn=lambda ids: "".join(chr(i) for i in ids),
            tokenizer_fingerprint="test",
            tokenizer_source_path="<test>",
            eos_id=0,
            pad_id=1,
        )
        assert adapter("hi") == [104, 105]

    def test_decode(self):
        adapter = TokenizerAdapter(
            encode_fn=lambda t: [ord(c) for c in t],
            decode_fn=lambda ids: "".join(chr(i) for i in ids),
            tokenizer_fingerprint="test",
            tokenizer_source_path="<test>",
            eos_id=0,
            pad_id=1,
        )
        assert adapter.decode([104, 105]) == "hi"

    def test_round_trip(self):
        adapter = TokenizerAdapter(
            encode_fn=lambda t: [ord(c) for c in t],
            decode_fn=lambda ids: "".join(chr(i) for i in ids),
            tokenizer_fingerprint="test",
            tokenizer_source_path="<test>",
            eos_id=0,
            pad_id=1,
        )
        text = "hello"
        assert adapter.decode(adapter.encode(text)) == text


class TestCheckpointRoundTrip:
    """Save and reload model + optimizer state."""

    def test_model_weights_survive_save_load(self, tiny_gpt_config, tmp_path):
        model = build_model(tiny_gpt_config)
        opt = AdamW(model.parameters(), lr=0.001)

        x = torch.randint(0, tiny_gpt_config.vocab_size, (2, 16))
        logits = model(x, return_loss=False)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            torch.randint(0, tiny_gpt_config.vocab_size, (2, 16)).view(-1),
        )
        loss.backward()
        opt.step()

        ckpt_path = tmp_path / "ckpt.pt"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": 1}, ckpt_path)

        model2 = build_model(tiny_gpt_config)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model2.load_state_dict(ckpt["model"])

        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            torch.testing.assert_close(p1, p2, msg=f"Mismatch in {n1}")

    def test_optimizer_state_survives_save_load(self, tiny_gpt_config, tmp_path):
        model = build_model(tiny_gpt_config)
        opt = AdamW(model.parameters(), lr=0.001)

        x = torch.randint(0, tiny_gpt_config.vocab_size, (2, 16))
        logits = model(x, return_loss=False)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            torch.randint(0, tiny_gpt_config.vocab_size, (2, 16)).view(-1),
        )
        loss.backward()
        opt.step()

        ckpt_path = tmp_path / "ckpt.pt"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": 1}, ckpt_path)

        model2 = build_model(tiny_gpt_config)
        opt2 = AdamW(model2.parameters(), lr=0.001)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model2.load_state_dict(ckpt["model"])
        opt2.load_state_dict(ckpt["opt"])

        assert ckpt["step"] == 1
        assert len(opt2.state_dict()["state"]) == len(opt.state_dict()["state"])

    def test_checkpoint_forward_pass_matches(self, tiny_gpt_config, tmp_path):
        """After loading a checkpoint, forward pass should produce identical output."""
        torch.manual_seed(99)
        model = build_model(tiny_gpt_config)

        x = torch.randint(0, tiny_gpt_config.vocab_size, (1, 16))
        model.eval()
        with torch.no_grad():
            out_original = model(x, return_loss=False)

        ckpt_path = tmp_path / "ckpt.pt"
        torch.save({"model": model.state_dict()}, ckpt_path)

        model2 = build_model(tiny_gpt_config)
        model2.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
        model2.eval()
        with torch.no_grad():
            out_loaded = model2(x, return_loss=False)

        torch.testing.assert_close(out_original, out_loaded)


class TestGetTokenizerS3PathCollision:
    """S3 tokenizer downloads must not collide when only the basename matches."""

    def test_different_s3_uris_same_basename_use_different_local_paths(self, tmp_path):
        """Two S3 URIs with the same filename but different prefixes must
        download to different local paths, otherwise the second call silently
        loads the wrong tokenizer bytes."""
        import train_utils

        uri_a = "s3://bucket-a/prefix-a/tokenizer.model"
        uri_b = "s3://bucket-b/prefix-b/tokenizer.model"

        def _local_path_for(uri):
            h = hashlib.sha256(uri.encode()).hexdigest()[:12]
            return Path("/tmp") / f"{h}_tokenizer.model"

        path_a = _local_path_for(uri_a)
        path_b = _local_path_for(uri_b)
        for p in (path_a, path_b):
            p.unlink(missing_ok=True)

        downloaded = {}

        def fake_download(bucket, key, dest):
            downloaded[dest] = (bucket, key)
            Path(dest).write_bytes(b"fake-model-data")

        mock_boto = MagicMock()
        mock_client = MagicMock()
        mock_client.download_file.side_effect = fake_download
        mock_boto.client.return_value = mock_client

        mock_sp = MagicMock()
        mock_sp.load.return_value = True
        mock_sp.eos_id.return_value = 0
        mock_sp.pad_id.return_value = 1
        mock_sp.encode.return_value = [1, 2, 3]
        mock_sp.decode.return_value = "abc"

        try:
            with patch.object(train_utils, "boto3", mock_boto), \
                 patch.object(train_utils, "spm") as mock_spm_mod:
                mock_spm_mod.SentencePieceProcessor.return_value = mock_sp

                get_tokenizer(uri_a)
                get_tokenizer(uri_b)

                assert mock_client.download_file.call_count == 2, (
                    f"Expected 2 downloads but got {mock_client.download_file.call_count}; "
                    "second URI likely hit the cached file from the first"
                )
        finally:
            for p in (path_a, path_b):
                p.unlink(missing_ok=True)
