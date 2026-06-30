"""Tests for serving.vllm — provision a vLLM OpenAI-compatible server.

Wiring-only: the server spawn and health check are seams that tests replace.
"""

from serving import vllm as vllm_mod
from serving.base import ServeBackend, ServingHandle


class TestBuildVllmArgs:
    def test_core_args(self):
        args = vllm_mod.build_vllm_args(
            "merged/smoke", host="0.0.0.0", port=8001, served_model_name="wm-sft"
        )
        assert "--model" in args and args[args.index("--model") + 1] == "merged/smoke"
        assert args[args.index("--host") + 1] == "0.0.0.0"
        assert args[args.index("--port") + 1] == "8001"
        assert args[args.index("--served-model-name") + 1] == "wm-sft"

    def test_served_model_name_optional(self):
        args = vllm_mod.build_vllm_args("m", served_model_name=None)
        assert "--served-model-name" not in args

    def test_extra_args_appended(self):
        extra = ["--enable-lora", "--max-model-len", "4096"]
        args = vllm_mod.build_vllm_args("m", extra_args=extra)
        assert "--enable-lora" in args
        assert args[args.index("--max-model-len") + 1] == "4096"


class _FakeProc:
    def __init__(self):
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


class TestDeploy:
    def test_deploy_spawns_waits_and_returns_handle(self, monkeypatch):
        captured = {}
        proc = _FakeProc()
        monkeypatch.setattr(
            vllm_mod, "_spawn_server", lambda args: captured.update(args=args) or proc
        )
        monkeypatch.setattr(
            vllm_mod, "_wait_healthy", lambda url, timeout=300: captured.update(url=url)
        )

        handle = vllm_mod.VLLMBackend(host="127.0.0.1", port=8000).deploy(
            "merged/smoke", served_model_name="wm-sft"
        )
        assert isinstance(handle, ServingHandle)
        assert handle.backend == "vllm"
        assert handle.model_name == "wm-sft"
        assert handle.base_url == "http://127.0.0.1:8000"
        assert handle.extra["process"] is proc
        assert "merged/smoke" in captured["args"]
        assert captured["url"] == "http://127.0.0.1:8000"

    def test_is_a_serve_backend(self):
        assert isinstance(vllm_mod.VLLMBackend(), ServeBackend)


class TestDelete:
    def test_delete_terminates_process(self):
        proc = _FakeProc()
        handle = ServingHandle(
            backend="vllm", model_name="m", base_url="http://x", extra={"process": proc}
        )
        vllm_mod.VLLMBackend().delete(handle)
        assert proc.terminated is True

    def test_delete_without_process_is_noop(self):
        handle = ServingHandle(backend="vllm", model_name="m", base_url="http://x")
        vllm_mod.VLLMBackend().delete(handle)  # must not raise
