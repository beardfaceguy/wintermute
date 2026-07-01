"""Tests for serving.sagemaker — provision an LMI endpoint for a model."""

import pytest
from serving import sagemaker as sm_mod
from serving.base import ServeBackend, ServingHandle


class TestBuildServingEnv:
    def test_hf_model_id_for_hub_ref(self):
        env = sm_mod.build_serving_env("Qwen/Qwen3-8B")
        assert env["HF_MODEL_ID"] == "Qwen/Qwen3-8B"
        assert env["OPTION_DTYPE"] == "bf16"
        assert env["TENSOR_PARALLEL_DEGREE"] == "max"

    def test_s3_ref_uses_model_dir_not_hub_id(self):
        env = sm_mod.build_serving_env("s3://bucket/model.tar.gz")
        # An S3 artifact is mounted by the container; no HF_MODEL_ID download.
        assert "HF_MODEL_ID" not in env
        assert env["OPTION_DTYPE"] == "bf16"

    def test_env_overrides_merge(self):
        env = sm_mod.build_serving_env("Qwen/Qwen3-8B", env_extra={"OPTION_MAX_MODEL_LEN": "16384"})
        assert env["OPTION_MAX_MODEL_LEN"] == "16384"

    def test_hf_token_injected_when_given(self):
        env = sm_mod.build_serving_env("Qwen/Qwen3-8B", hf_token="hf_abc")
        assert env["HF_TOKEN"] == "hf_abc"

    def test_no_hf_token_key_when_absent(self):
        assert "HF_TOKEN" not in sm_mod.build_serving_env("Qwen/Qwen3-8B")


class TestGetHfToken:
    def test_explicit_wins(self):
        assert sm_mod.get_hf_token("hf_explicit") == "hf_explicit"


class _FakePredictor:
    def __init__(self):
        self.endpoint_name = "wm-sft-ep"


class TestDeploy:
    def test_deploy_resolves_role_builds_and_returns_handle(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(sm_mod, "get_hf_token", lambda explicit: None)  # hermetic
        monkeypatch.setattr(sm_mod, "_resolve_role", lambda profile: "arn:role/x")

        def fake_deploy(model_ref, env, role_arn, instance_type, endpoint_name, profile):
            captured.update(
                model_ref=model_ref,
                env=env,
                role_arn=role_arn,
                instance_type=instance_type,
                endpoint_name=endpoint_name,
            )
            return "wm-sft-ep"

        monkeypatch.setattr(sm_mod, "_deploy_model", fake_deploy)

        handle = sm_mod.SageMakerServeBackend(profile="experimental").deploy(
            "Qwen/Qwen3-8B", instance_type="ml.g5.2xlarge", endpoint_name="wm-sft-ep"
        )
        assert isinstance(handle, ServingHandle)
        assert handle.backend == "sagemaker"
        assert handle.endpoint_name == "wm-sft-ep"
        assert handle.base_url is None  # invoked via boto3, not a URL
        assert captured["role_arn"] == "arn:role/x"
        assert captured["env"]["HF_MODEL_ID"] == "Qwen/Qwen3-8B"
        assert captured["instance_type"] == "ml.g5.2xlarge"

    def test_is_a_serve_backend(self):
        assert isinstance(sm_mod.SageMakerServeBackend(), ServeBackend)


class TestDelete:
    def test_delete_calls_delete_endpoint(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            sm_mod, "_delete_endpoint", lambda profile, name: captured.update(profile=profile, name=name)
        )
        handle = ServingHandle(backend="sagemaker", model_name="m", endpoint_name="wm-sft-ep")
        sm_mod.SageMakerServeBackend(profile="experimental").delete(handle)
        assert captured == {"profile": "experimental", "name": "wm-sft-ep"}

    def test_delete_without_endpoint_name_raises(self):
        handle = ServingHandle(backend="sagemaker", model_name="m")
        with pytest.raises(ValueError, match="endpoint_name"):
            sm_mod.SageMakerServeBackend().delete(handle)


class TestCli:
    def test_delete_flag_calls_delete_endpoint(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            sm_mod, "_delete_endpoint", lambda profile, name: captured.update(profile=profile, name=name)
        )
        sm_mod.main(["--profile", "experimental", "--delete", "wm-ep"])
        assert captured == {"profile": "experimental", "name": "wm-ep"}

    def test_deploy_flag_invokes_backend(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(sm_mod, "get_hf_token", lambda explicit: None)
        monkeypatch.setattr(sm_mod, "_resolve_role", lambda profile: "arn:role/x")
        monkeypatch.setattr(
            sm_mod,
            "_deploy_model",
            lambda *a: captured.update(model_ref=a[0], endpoint_name=a[4]) or a[4],
        )
        sm_mod.main(["--model", "Qwen/Qwen3-8B", "--endpoint-name", "wm-ep"])
        assert captured["model_ref"] == "Qwen/Qwen3-8B"
        assert captured["endpoint_name"] == "wm-ep"

    def test_deploy_requires_model(self):
        with pytest.raises(SystemExit):
            sm_mod.main(["--profile", "experimental"])  # no --model, no --delete
