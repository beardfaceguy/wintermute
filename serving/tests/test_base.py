"""Tests for serving.base — ServeBackend interface and ServingHandle."""

import pytest

from serving.base import ServeBackend, ServingHandle


class TestServingHandle:
    def test_openai_client_built_from_base_url(self, monkeypatch):
        # Reuse eval.model.OpenAICompatBackend rather than a new client.
        import eval.model as em

        captured = {}

        class FakeBackend:
            def __init__(self, base_url, model, api_key="none"):
                captured.update(base_url=base_url, model=model)

        monkeypatch.setattr(em, "OpenAICompatBackend", FakeBackend)

        handle = ServingHandle(backend="vllm", model_name="m", base_url="http://x:8000")
        handle.openai_client()
        assert captured == {"base_url": "http://x:8000", "model": "m"}

    def test_openai_client_requires_base_url(self):
        handle = ServingHandle(backend="sagemaker", model_name="m", endpoint_name="ep")
        with pytest.raises(ValueError, match="base_url"):
            handle.openai_client()


class TestServeBackend:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            ServeBackend()

    def test_subclass_must_implement_deploy_and_delete(self):
        class OnlyDeploy(ServeBackend):
            def deploy(self, model_ref, **kwargs):
                return ServingHandle(backend="x", model_name="m")

        with pytest.raises(TypeError):
            OnlyDeploy()

    def test_valid_subclass(self):
        class Ok(ServeBackend):
            def deploy(self, model_ref, **kwargs):
                return ServingHandle(backend="x", model_name=model_ref)

            def delete(self, handle):
                pass

        h = Ok().deploy("m")
        assert h.model_name == "m"
