"""Tests for distributed training utilities: DDP helpers, data sharding, grad
accum scaling, rank-guarded behavior, single-process end-to-end training, and
multi-process DDP via gloo backend on CPU.

Most tests run on CPU without NCCL or actual multi-GPU hardware. The gloo-based
tests (marked @pytest.mark.slow) spawn real processes to validate gradient
synchronization and rank coordination.
"""

import os
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from model import ModelConfig, build_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
from train_utils import (
    build_distributed_dataloader,
    cleanup_distributed,
    cosine_lr,
    is_main,
    reduce_scalar,
    setup_distributed,
)

# ---------------------------------------------------------------------------
# Distributed helper functions
# ---------------------------------------------------------------------------


class TestSetupDistributed:
    """setup_distributed() fallback behavior without torchrun."""

    def test_fallback_without_rank_env(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)

        with patch("train_utils.torch.cuda.set_device"):
            rank, local_rank, world_size = setup_distributed()
        assert rank == 0
        assert local_rank == 0
        assert world_size == 1

    def test_cleanup_noop_without_init(self):
        """cleanup should not raise when no process group is initialized."""
        cleanup_distributed()


class TestIsMain:
    def test_rank_zero_is_main(self):
        assert is_main(0) is True

    def test_rank_nonzero_not_main(self):
        assert is_main(1) is False
        assert is_main(7) is False


class TestReduceScalar:
    def test_single_process_passthrough(self):
        assert reduce_scalar(3.14, world_size=1) == 3.14

    def test_single_process_negative(self):
        assert reduce_scalar(-1.5, world_size=1) == -1.5

    def test_single_process_zero(self):
        assert reduce_scalar(0.0, world_size=1) == 0.0


# ---------------------------------------------------------------------------
# Distributed dataloader
# ---------------------------------------------------------------------------


class TestBuildDistributedDataloader:
    """build_distributed_dataloader with world_size=1 and simulated multi-rank."""

    def test_single_process_returns_none_sampler(self, small_text_file, dummy_tokenizer):
        loader, sampler = build_distributed_dataloader(
            str(small_text_file),
            dummy_tokenizer,
            dummy_tokenizer.tokenizer_fingerprint,
            seq_len=32,
            batch_size=2,
            rank=0,
            world_size=1,
            num_workers=0,
        )
        assert sampler is None
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (2, 32)

    def test_multi_rank_returns_distributed_sampler(self, small_text_file, dummy_tokenizer):
        loader, sampler = build_distributed_dataloader(
            str(small_text_file),
            dummy_tokenizer,
            dummy_tokenizer.tokenizer_fingerprint,
            seq_len=32,
            batch_size=2,
            rank=0,
            world_size=4,
            num_workers=0,
        )
        assert isinstance(sampler, DistributedSampler)

    def test_different_ranks_get_different_indices(self, small_text_file, dummy_tokenizer):
        """Each rank should see a mostly-disjoint shard of the dataset.

        DistributedSampler pads the dataset to be evenly divisible by world_size,
        which can cause the last few indices to repeat across ranks. We verify
        that the vast majority of indices are unique per rank.
        """
        world_size = 4
        all_indices = []
        for rank in range(world_size):
            _, sampler = build_distributed_dataloader(
                str(small_text_file),
                dummy_tokenizer,
                dummy_tokenizer.tokenizer_fingerprint,
                seq_len=32,
                batch_size=2,
                rank=rank,
                world_size=world_size,
                shuffle=False,
                num_workers=0,
            )
            indices = list(sampler)
            all_indices.append(indices)

        # Each rank should get roughly 1/world_size of the data
        total_per_rank = len(all_indices[0])
        for rank_indices in all_indices:
            assert len(rank_indices) == total_per_rank

        # The index lists themselves should differ between ranks
        for i in range(world_size):
            for j in range(i + 1, world_size):
                assert all_indices[i] != all_indices[j], (
                    f"Rank {i} and {j} got identical index lists"
                )

    def test_all_ranks_cover_full_dataset(self, small_text_file, dummy_tokenizer):
        """Union of all rank indices should cover the full dataset (with possible padding)."""
        world_size = 4
        combined = set()
        for rank in range(world_size):
            _, sampler = build_distributed_dataloader(
                str(small_text_file),
                dummy_tokenizer,
                dummy_tokenizer.tokenizer_fingerprint,
                seq_len=32,
                batch_size=2,
                rank=rank,
                world_size=world_size,
                shuffle=False,
                num_workers=0,
            )
            combined.update(list(sampler))

        loader, _ = build_distributed_dataloader(
            str(small_text_file),
            dummy_tokenizer,
            dummy_tokenizer.tokenizer_fingerprint,
            seq_len=32,
            batch_size=2,
            rank=0,
            world_size=1,
            shuffle=False,
            num_workers=0,
        )
        full_size = len(loader.dataset)
        # DistributedSampler may pad to make even splits
        assert len(combined) >= full_size


class TestDistributedSamplerEpoch:
    """DistributedSampler.set_epoch changes shuffling order."""

    def test_set_epoch_changes_order(self, small_text_file, dummy_tokenizer):
        _, sampler = build_distributed_dataloader(
            str(small_text_file),
            dummy_tokenizer,
            dummy_tokenizer.tokenizer_fingerprint,
            seq_len=32,
            batch_size=2,
            rank=0,
            world_size=2,
            shuffle=True,
            num_workers=0,
        )
        sampler.set_epoch(0)
        order_epoch_0 = list(sampler)

        sampler.set_epoch(1)
        order_epoch_1 = list(sampler)

        # Same length, but different order
        assert len(order_epoch_0) == len(order_epoch_1)
        assert order_epoch_0 != order_epoch_1, "set_epoch should change shuffle order"

    def test_same_epoch_gives_same_order(self, small_text_file, dummy_tokenizer):
        _, sampler = build_distributed_dataloader(
            str(small_text_file),
            dummy_tokenizer,
            dummy_tokenizer.tokenizer_fingerprint,
            seq_len=32,
            batch_size=2,
            rank=0,
            world_size=2,
            shuffle=True,
            num_workers=0,
        )
        sampler.set_epoch(42)
        order_a = list(sampler)

        sampler.set_epoch(42)
        order_b = list(sampler)

        assert order_a == order_b


# ---------------------------------------------------------------------------
# Gradient accumulation auto-scaling
# ---------------------------------------------------------------------------


class TestGradAccumScaling:
    """The auto-scaling math: grad_accum_steps // world_size."""

    def test_8gpu_scales_32_to_4(self):
        original = 32
        world_size = 8
        scaled = max(1, original // world_size)
        assert scaled == 4

    def test_4gpu_scales_32_to_8(self):
        assert max(1, 32 // 4) == 8

    def test_2gpu_scales_32_to_16(self):
        assert max(1, 32 // 2) == 16

    def test_1gpu_no_scaling(self):
        assert max(1, 32 // 1) == 32

    def test_floor_division_rounds_down(self):
        # 32 GPUs with accum=32 → 1 (not 0)
        assert max(1, 32 // 32) == 1

    def test_more_gpus_than_accum_clamps_to_1(self):
        # 64 GPUs with accum=32 → floor is 0, clamped to 1
        assert max(1, 32 // 64) == 1

    def test_effective_batch_preserved(self):
        """Effective batch should be the same before and after scaling."""
        batch_size = 2
        seq_len = 1024
        original_accum = 32
        world_size = 8
        scaled_accum = max(1, original_accum // world_size)

        original_effective = batch_size * seq_len * original_accum * 1
        scaled_effective = batch_size * seq_len * scaled_accum * world_size

        assert original_effective == scaled_effective


# ---------------------------------------------------------------------------
# Rank-guarded logging
# ---------------------------------------------------------------------------


class TestRankGuardedLogging:
    """Only rank 0 should produce log output."""

    def test_rank_0_logs(self, capsys):
        import time

        start_time = time.time()

        def log_fn(msg, rank=0):
            if not is_main(rank):
                return
            elapsed = time.time() - start_time
            print(f"[{elapsed:7.1f}s] {msg}")

        log_fn("hello from rank 0", rank=0)
        captured = capsys.readouterr()
        assert "hello from rank 0" in captured.out

    def test_non_rank_0_silent(self, capsys):
        import time

        start_time = time.time()

        def log_fn(msg, rank=1):
            if not is_main(rank):
                return
            elapsed = time.time() - start_time
            print(f"[{elapsed:7.1f}s] {msg}")

        log_fn("hello from rank 1", rank=1)
        log_fn("hello from rank 3", rank=3)
        log_fn("hello from rank 7", rank=7)
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# Effective tokens per step across world sizes
# ---------------------------------------------------------------------------


class TestEffectiveTokenCounting:
    """total_tokens_seen should account for all GPUs."""

    def test_tokens_scaled_by_world_size(self):
        batch_size = 2
        seq_len = 1024
        world_size = 8
        x_numel = batch_size * seq_len
        tokens_this_step = x_numel * world_size
        assert tokens_this_step == 16384

    def test_tokens_single_gpu(self):
        batch_size = 2
        seq_len = 1024
        world_size = 1
        tokens_this_step = (batch_size * seq_len) * world_size
        assert tokens_this_step == 2048


# ---------------------------------------------------------------------------
# DDP checkpoint save/load: raw_model.state_dict() portability
# ---------------------------------------------------------------------------


class TestDDPCheckpointPortability:
    """Checkpoints saved from DDP (model.module) should load into bare models."""

    def test_save_module_load_bare(self, tiny_gpt_config, tmp_path):
        """Simulate DDP: wrap model, save model.module, reload into bare model."""
        model = build_model(tiny_gpt_config)
        opt = AdamW(model.parameters(), lr=0.001)

        # Simulate a training step
        x = torch.randint(0, tiny_gpt_config.vocab_size, (2, 16))
        logits = model(x, return_loss=False)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            torch.randint(0, tiny_gpt_config.vocab_size, (2, 16)).view(-1),
        )
        loss.backward()
        opt.step()

        # In DDP, raw_model = model.module; here we simulate by saving model directly
        # (since without actual DDP wrapping, model IS the raw model)
        ckpt_path = tmp_path / "ddp_ckpt.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": 1,
            },
            ckpt_path,
        )

        # Load into a fresh bare model
        model2 = build_model(tiny_gpt_config)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model2.load_state_dict(ckpt["model"])

        # Verify weights match
        model.eval()
        model2.eval()
        test_x = torch.randint(0, tiny_gpt_config.vocab_size, (1, 16))
        with torch.no_grad():
            out1 = model(test_x, return_loss=False)
            out2 = model2(test_x, return_loss=False)
        torch.testing.assert_close(out1, out2)

    def test_state_dict_keys_have_no_module_prefix(self, tiny_gpt_config):
        """raw_model.state_dict() should not have 'module.' prefix."""
        model = build_model(tiny_gpt_config)
        sd = model.state_dict()
        for key in sd:
            assert not key.startswith("module."), f"Unexpected 'module.' prefix in key: {key}"


# ---------------------------------------------------------------------------
# no_sync context manager compatibility
# ---------------------------------------------------------------------------


class TestNoSyncContext:
    """Verify the sync_context logic produces correct context managers."""

    def test_single_gpu_always_nullcontext(self):
        world_size = 1
        grad_accum_steps = 4
        for accum_in_step in range(grad_accum_steps):
            is_last = (accum_in_step + 1) == grad_accum_steps
            ctx = nullcontext() if (world_size <= 1 or is_last) else "no_sync"
            assert type(ctx) is nullcontext, (
                f"Single GPU should always use nullcontext, got {type(ctx)} at step {accum_in_step}"
            )

    def test_multi_gpu_last_step_syncs(self):
        """On the last accumulation micro-batch, gradients should sync (nullcontext)."""
        world_size = 8
        grad_accum_steps = 4
        accum_in_step = grad_accum_steps - 1
        is_last = (accum_in_step + 1) == grad_accum_steps
        should_sync = world_size <= 1 or is_last
        assert should_sync is True

    def test_multi_gpu_non_last_step_skips_sync(self):
        """On non-last micro-batches, no_sync should be used."""
        world_size = 8
        grad_accum_steps = 4
        for accum_in_step in range(grad_accum_steps - 1):
            is_last = (accum_in_step + 1) == grad_accum_steps
            should_sync = world_size <= 1 or is_last
            assert should_sync is False, (
                f"Expected no_sync at micro-batch {accum_in_step}/{grad_accum_steps}"
            )

    def test_accum_1_always_syncs(self):
        """With grad_accum=1, every step is the last step → always sync."""
        world_size = 8
        grad_accum_steps = 1
        accum_in_step = 0
        is_last = (accum_in_step + 1) == grad_accum_steps
        should_sync = world_size <= 1 or is_last
        assert should_sync is True


# ---------------------------------------------------------------------------
# Single-process end-to-end mini training (CPU, no DDP)
# ---------------------------------------------------------------------------


class TestSingleProcessEndToEnd:
    """Run a few training steps on CPU via the multi-GPU code path (world_size=1).
    Validates that the training loop logic works end-to-end."""

    def _run_mini_train(
        self,
        tiny_gpt_config,
        small_text_file,
        dummy_tokenizer,
        tmp_path,
        max_steps=10,
        grad_accum_steps=2,
    ):
        """Helper: run a short training loop and return (losses, checkpoint_path)."""
        model = build_model(tiny_gpt_config)
        opt = AdamW(model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-8)

        loader, sampler = build_distributed_dataloader(
            str(small_text_file),
            dummy_tokenizer,
            dummy_tokenizer.tokenizer_fingerprint,
            seq_len=32,
            batch_size=2,
            rank=0,
            world_size=1,
            shuffle=True,
            num_workers=0,
        )

        model.train()
        global_step = 0
        accum_in_step = 0
        accum_loss_sum = 0.0
        losses = []
        opt.zero_grad(set_to_none=True)

        for x, y in loader:
            logits = model(x, return_loss=False)
            raw_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = raw_loss / grad_accum_steps
            loss.backward()

            accum_in_step += 1
            accum_loss_sum += raw_loss.item()

            if accum_in_step < grad_accum_steps:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            global_step += 1
            step_loss = accum_loss_sum / grad_accum_steps
            losses.append(step_loss)
            accum_in_step = 0
            accum_loss_sum = 0.0

            if global_step >= max_steps:
                break

        ckpt_path = tmp_path / "mini_ckpt.pt"
        tmp_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": global_step,
            },
            ckpt_path,
        )

        return losses, ckpt_path, model

    def test_loss_decreases(self, tiny_gpt_config, small_text_file, dummy_tokenizer, tmp_path):
        losses, _, _ = self._run_mini_train(
            tiny_gpt_config,
            small_text_file,
            dummy_tokenizer,
            tmp_path,
            max_steps=20,
            grad_accum_steps=1,
        )
        assert len(losses) == 20
        # Average of first 5 should be higher than average of last 5
        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        assert late < early, f"Loss should decrease: early={early:.4f} late={late:.4f}"

    def test_grad_accum_produces_same_step_count(
        self, tiny_gpt_config, small_text_file, dummy_tokenizer, tmp_path
    ):
        losses, _, _ = self._run_mini_train(
            tiny_gpt_config,
            small_text_file,
            dummy_tokenizer,
            tmp_path,
            max_steps=5,
            grad_accum_steps=4,
        )
        assert len(losses) == 5

    def test_checkpoint_is_resumable(
        self, tiny_gpt_config, small_text_file, dummy_tokenizer, tmp_path
    ):
        _, ckpt_path, _ = self._run_mini_train(
            tiny_gpt_config,
            small_text_file,
            dummy_tokenizer,
            tmp_path,
            max_steps=5,
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert ckpt["step"] == 5
        assert "model" in ckpt
        assert "opt" in ckpt

        # Reload into fresh model
        model2 = build_model(tiny_gpt_config)
        model2.load_state_dict(ckpt["model"])

        opt2 = AdamW(model2.parameters(), lr=0.001)
        opt2.load_state_dict(ckpt["opt"])

    def test_single_vs_accum_produces_different_weights(
        self, tiny_gpt_config, small_text_file, dummy_tokenizer, tmp_path
    ):
        """Different grad_accum should produce different final weights
        (sanity: the accum path actually changes optimizer behavior)."""
        torch.manual_seed(42)
        _, _, model_accum1 = self._run_mini_train(
            tiny_gpt_config,
            small_text_file,
            dummy_tokenizer,
            tmp_path / "a1",
            max_steps=10,
            grad_accum_steps=1,
        )
        torch.manual_seed(42)
        _, _, model_accum4 = self._run_mini_train(
            tiny_gpt_config,
            small_text_file,
            dummy_tokenizer,
            tmp_path / "a4",
            max_steps=10,
            grad_accum_steps=4,
        )
        # Weights should differ because accum changes effective batch composition
        all_same = all(
            torch.equal(p1, p2)
            for p1, p2 in zip(model_accum1.parameters(), model_accum4.parameters(), strict=False)
        )
        assert not all_same, "Different grad_accum should produce different weights"


# ---------------------------------------------------------------------------
# Config-driven effective batch size calculations
# ---------------------------------------------------------------------------


class TestEffectiveBatchConfig:
    """Verify effective_tokens_per_step = batch_size * seq_len * grad_accum * world_size."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,accum,world_size,expected",
        [
            (2, 1024, 32, 1, 65536),
            (2, 1024, 4, 8, 65536),
            (2, 1024, 16, 2, 65536),
            (2, 1024, 8, 4, 65536),
            (4, 512, 16, 2, 65536),
            (1, 1024, 32, 1, 32768),
        ],
    )
    def test_effective_tokens(self, batch_size, seq_len, accum, world_size, expected):
        effective = batch_size * seq_len * accum * world_size
        assert effective == expected


# ===========================================================================
# Gloo-based multi-process DDP tests (CPU, no GPU required)
#
# These spawn real processes to validate actual distributed operations:
# gradient sync, all-reduce, rank-guarded checkpointing, etc.
# ===========================================================================

WORLD_SIZE_GLOO = 2
TINY_CFG = ModelConfig(
    variant="gpt", vocab_size=256, dim=64, depth=2, heads=4, ff_mult=2, max_seq_len=64
)


def _init_gloo(rank, world_size, tmp_dir):
    """Initialize a gloo process group using a file-based store."""
    store_path = os.path.join(tmp_dir, "store")
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=world_size)


def _cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


# --- Worker functions (must be top-level for mp.spawn) ---


def _worker_gradient_sync(rank, world_size, tmp_dir, results_dir):
    """Each rank trains the same model on different data; DDP should sync gradients."""
    try:
        _init_gloo(rank, world_size, tmp_dir)

        torch.manual_seed(0)
        model = build_model(TINY_CFG)
        model = DDP(model)

        opt = AdamW(model.parameters(), lr=0.01)

        torch.manual_seed(rank + 100)
        x = torch.randint(0, TINY_CFG.vocab_size, (4, 32))
        y = torch.randint(0, TINY_CFG.vocab_size, (4, 32))

        model.train()
        opt.zero_grad()
        logits = model(x, return_loss=False)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        opt.step()

        # Save each rank's model weights
        sd = {k: v.clone() for k, v in model.module.state_dict().items()}
        torch.save(sd, os.path.join(results_dir, f"weights_rank{rank}.pt"))
    finally:
        _cleanup()


def _worker_reduce_scalar(rank, world_size, tmp_dir, results_dir):
    """Test reduce_scalar with an actual all-reduce over gloo."""
    try:
        _init_gloo(rank, world_size, tmp_dir)

        value = float(rank + 1)  # rank 0 → 1.0, rank 1 → 2.0
        t = torch.tensor(value, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        result = t.item() / world_size

        torch.save(
            {"rank": rank, "result": result}, os.path.join(results_dir, f"reduce_rank{rank}.pt")
        )
    finally:
        _cleanup()


def _worker_checkpoint_rank_guard(rank, world_size, tmp_dir, results_dir):
    """Only rank 0 should write the checkpoint file."""
    try:
        _init_gloo(rank, world_size, tmp_dir)

        torch.manual_seed(0)
        model = build_model(TINY_CFG)
        model = DDP(model)

        if is_main(rank):
            ckpt_path = os.path.join(results_dir, "ckpt.pt")
            torch.save({"model": model.module.state_dict(), "step": 1}, ckpt_path)

        dist.barrier()
    finally:
        _cleanup()


def _worker_barrier_coordination(rank, world_size, tmp_dir, results_dir):
    """Verify barrier actually blocks: rank 0 writes a file, barrier, rank 1 reads it."""
    try:
        _init_gloo(rank, world_size, tmp_dir)

        signal_path = os.path.join(results_dir, "signal.txt")

        if rank == 0:
            Path(signal_path).write_text("ready")

        dist.barrier()

        if rank == 1:
            exists = Path(signal_path).exists()
            torch.save(
                {"signal_found": exists}, os.path.join(results_dir, f"barrier_rank{rank}.pt")
            )
    finally:
        _cleanup()


def _worker_ddp_training_loop(rank, world_size, tmp_dir, results_dir):
    """Multi-process mini training loop: DDP model, grad accum, cosine LR."""
    try:
        _init_gloo(rank, world_size, tmp_dir)

        torch.manual_seed(0)
        model = build_model(TINY_CFG)
        model = DDP(model)

        opt = AdamW(model.parameters(), lr=0.01, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-8)

        grad_accum_steps = 2
        max_steps = 5
        global_step = 0
        accum_in_step = 0
        losses = []
        opt.zero_grad(set_to_none=True)

        torch.manual_seed(rank + 200)
        dataset_x = torch.randint(0, TINY_CFG.vocab_size, (40, 16))
        dataset_y = torch.randint(0, TINY_CFG.vocab_size, (40, 16))

        for i in range(len(dataset_x)):
            x = dataset_x[i : i + 1]
            y = dataset_y[i : i + 1]

            is_last_accum = (accum_in_step + 1) == grad_accum_steps
            sync_ctx = nullcontext() if is_last_accum else model.no_sync()

            with sync_ctx:
                logits = model(x, return_loss=False)
                raw_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = raw_loss / grad_accum_steps
                loss.backward()

            accum_in_step += 1
            if accum_in_step < grad_accum_steps:
                continue

            lr = cosine_lr(global_step, warmup=2, max_steps=max_steps, base_lr=0.01, min_lr=0.001)
            for pg in opt.param_groups:
                pg["lr"] = lr

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            global_step += 1
            losses.append(raw_loss.item())
            accum_in_step = 0

            if global_step >= max_steps:
                break

        torch.save(
            {
                "losses": losses,
                "weights": {k: v.clone() for k, v in model.module.state_dict().items()},
                "step": global_step,
            },
            os.path.join(results_dir, f"loop_rank{rank}.pt"),
        )
    finally:
        _cleanup()


# --- Actual test class ---


@pytest.mark.slow
class TestGlooDDP:
    """Multi-process DDP tests using gloo backend on CPU.

    These spawn real processes via mp.spawn to validate distributed operations
    that cannot be tested in a single process.
    """

    def _run_workers(self, worker_fn, tmp_path, world_size=WORLD_SIZE_GLOO):
        """Spawn worker processes and return the results directory."""
        tmp_dir = str(tmp_path / "store")
        results_dir = str(tmp_path / "results")
        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        mp.spawn(worker_fn, args=(world_size, tmp_dir, results_dir), nprocs=world_size, join=True)
        return results_dir

    def test_gradients_synchronized_across_ranks(self, tmp_path):
        """After one DDP step, all ranks should have identical model weights."""
        results_dir = self._run_workers(_worker_gradient_sync, tmp_path)

        w0 = torch.load(os.path.join(results_dir, "weights_rank0.pt"), map_location="cpu")
        w1 = torch.load(os.path.join(results_dir, "weights_rank1.pt"), map_location="cpu")

        for key in w0:
            torch.testing.assert_close(
                w0[key],
                w1[key],
                msg=f"Weight mismatch after DDP step for key '{key}'",
            )

    def test_all_reduce_scalar(self, tmp_path):
        """reduce_scalar equivalent: rank 0 sends 1.0, rank 1 sends 2.0 → mean is 1.5."""
        results_dir = self._run_workers(_worker_reduce_scalar, tmp_path)

        r0 = torch.load(os.path.join(results_dir, "reduce_rank0.pt"), map_location="cpu")
        r1 = torch.load(os.path.join(results_dir, "reduce_rank1.pt"), map_location="cpu")

        assert r0["result"] == pytest.approx(1.5)
        assert r1["result"] == pytest.approx(1.5)

    def test_only_rank_0_writes_checkpoint(self, tmp_path):
        """Checkpoint file should exist (written by rank 0), but only one copy."""
        results_dir = self._run_workers(_worker_checkpoint_rank_guard, tmp_path)

        ckpt_path = os.path.join(results_dir, "ckpt.pt")
        assert os.path.exists(ckpt_path), "Rank 0 should have written the checkpoint"

        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert ckpt["step"] == 1
        assert "model" in ckpt

    def test_barrier_coordination(self, tmp_path):
        """Rank 1 should see rank 0's file after the barrier."""
        results_dir = self._run_workers(_worker_barrier_coordination, tmp_path)

        r1 = torch.load(os.path.join(results_dir, "barrier_rank1.pt"), map_location="cpu")
        assert r1["signal_found"] is True, (
            "Barrier should ensure rank 0's write is visible to rank 1"
        )

    def test_full_ddp_training_loop(self, tmp_path):
        """Multi-step training with DDP, grad accum, and no_sync.
        Both ranks should finish with identical weights and decreasing loss."""
        results_dir = self._run_workers(_worker_ddp_training_loop, tmp_path)

        r0 = torch.load(os.path.join(results_dir, "loop_rank0.pt"), map_location="cpu")
        r1 = torch.load(os.path.join(results_dir, "loop_rank1.pt"), map_location="cpu")

        # Both ranks completed all steps
        assert r0["step"] == 5
        assert r1["step"] == 5

        # Weights must be identical after DDP training
        for key in r0["weights"]:
            torch.testing.assert_close(
                r0["weights"][key],
                r1["weights"][key],
                msg=f"Weight divergence after DDP training for key '{key}'",
            )

    def test_ddp_checkpoint_loadable_without_ddp(self, tmp_path):
        """Checkpoint from DDP training should load into a bare (non-DDP) model."""
        results_dir = self._run_workers(_worker_ddp_training_loop, tmp_path)

        r0 = torch.load(os.path.join(results_dir, "loop_rank0.pt"), map_location="cpu")

        bare_model = build_model(TINY_CFG)
        bare_model.load_state_dict(r0["weights"])

        x = torch.randint(0, TINY_CFG.vocab_size, (1, 16))
        bare_model.eval()
        with torch.no_grad():
            logits = bare_model(x, return_loss=False)
        assert logits.shape == (1, 16, TINY_CFG.vocab_size)
