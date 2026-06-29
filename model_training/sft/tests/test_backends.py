"""Tests for training-compute backends (base / local / sagemaker).

Wiring-only, mock-and-defer: no real training job runs here. The local backend's
call into train() is monkeypatched; the SageMaker estimator construction and role
resolution are isolated behind seam functions that tests replace.
"""

import pytest
from model_training.sft.backends import local as local_mod
from model_training.sft.backends import sagemaker as sm_mod
from model_training.sft.backends.base import TrainBackend
from model_training.sft.config import SFTConfig


def _cfg(**over):
    d = {
        "model": {"base_model": "Qwen/Qwen3-8B"},
        "data": {"train_path": "s3://bucket/train.jsonl"},
        "training": {"output_dir": "/opt/ml/model"},
    }
    for k, v in over.items():
        d.setdefault(k, {}).update(v)
    return SFTConfig.from_dict(d)


# ── base ──────────────────────────────────────────────────────────────────────


class TestBase:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            TrainBackend()

    def test_subclass_must_implement_run(self):
        class Incomplete(TrainBackend):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_valid_subclass_works(self):
        class Ok(TrainBackend):
            def run(self, cfg):
                return "ckpt"

        assert Ok().run(_cfg()) == "ckpt"


# ── local ─────────────────────────────────────────────────────────────────────


class TestLocalBackend:
    def test_run_delegates_to_train(self, monkeypatch):
        seen = {}

        def fake_train(cfg):
            seen["cfg"] = cfg
            return "/opt/ml/model"

        monkeypatch.setattr("model_training.sft.train.train", fake_train)
        cfg = _cfg()
        out = local_mod.LocalBackend().run(cfg)
        assert out == "/opt/ml/model"
        assert seen["cfg"] is cfg

    def test_is_a_train_backend(self):
        assert isinstance(local_mod.LocalBackend(), TrainBackend)


# ── sagemaker ─────────────────────────────────────────────────────────────────


class TestBuildEstimatorKwargs:
    def test_core_fields(self):
        kw = sm_mod.build_estimator_kwargs(
            _cfg(),
            role_arn="arn:aws:iam::123:role/SageMakerExecutionRole",
            source_dir="/repo",
            config_path="model_training/sft/configs/qwen3_8b_lora.yaml",
        )
        assert kw["entry_point"] == "model_training/sft/train.py"
        assert kw["source_dir"] == "/repo"
        assert kw["role"] == "arn:aws:iam::123:role/SageMakerExecutionRole"
        assert kw["instance_type"] == sm_mod.DEFAULT_INSTANCE
        assert kw["instance_count"] == 1
        assert kw["hyperparameters"]["config"] == (
            "model_training/sft/configs/qwen3_8b_lora.yaml"
        )

    def test_framework_versions_when_no_image(self):
        kw = sm_mod.build_estimator_kwargs(
            _cfg(), role_arn="r", source_dir="/repo", config_path="c.yaml"
        )
        assert "transformers_version" in kw
        assert "pytorch_version" in kw
        assert "py_version" in kw
        assert "image_uri" not in kw

    def test_image_uri_overrides_framework_versions(self):
        kw = sm_mod.build_estimator_kwargs(
            _cfg(),
            role_arn="r",
            source_dir="/repo",
            config_path="c.yaml",
            image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface:x",
        )
        assert kw["image_uri"].endswith("huggingface:x")
        assert "transformers_version" not in kw

    def test_instance_type_override(self):
        kw = sm_mod.build_estimator_kwargs(
            _cfg(), role_arn="r", source_dir="/repo", config_path="c.yaml",
            instance_type="ml.p4d.24xlarge",
        )
        assert kw["instance_type"] == "ml.p4d.24xlarge"

    def test_job_name_is_sanitized(self):
        kw = sm_mod.build_estimator_kwargs(
            _cfg(), role_arn="r", source_dir="/repo", config_path="c.yaml"
        )
        # base_job_name must be DNS-safe: no slashes, lowercased
        assert "/" not in kw["base_job_name"]
        assert kw["base_job_name"] == kw["base_job_name"].lower()


class TestSagemakerRun:
    def test_run_builds_estimator_and_fits(self, monkeypatch):
        captured = {}

        class FakeEstimator:
            def __init__(self):
                self.model_data = "s3://bucket/output/model.tar.gz"

            def fit(self, inputs, wait=True):
                captured["inputs"] = inputs
                captured["wait"] = wait

        monkeypatch.setattr(sm_mod, "_resolve_role", lambda profile: "arn:role/x")
        monkeypatch.setattr(
            sm_mod, "_make_estimator", lambda kwargs, profile: captured.update(kwargs=kwargs) or FakeEstimator()
        )

        out = sm_mod.SageMakerBackend(profile="experimental").run(
            _cfg(), source_dir="/repo", config_path="c.yaml"
        )
        assert out == "s3://bucket/output/model.tar.gz"
        assert captured["inputs"] == {"train": "s3://bucket/train.jsonl"}
        assert captured["kwargs"]["role"] == "arn:role/x"

    def test_is_a_train_backend(self):
        assert isinstance(sm_mod.SageMakerBackend(), TrainBackend)
