"""Tests for serving.ollama — register a model with Ollama via a Modelfile."""

from serving import ollama as ollama_mod
from serving.base import ServeBackend, ServingHandle


class TestBuildModelfile:
    def test_from_line(self):
        mf = ollama_mod.build_modelfile("outputs/smoke-merged")
        assert mf.splitlines()[0] == "FROM outputs/smoke-merged"

    def test_system_prompt_included(self):
        mf = ollama_mod.build_modelfile("m", system="be terse")
        assert 'SYSTEM """be terse"""' in mf

    def test_no_system_line_when_absent(self):
        assert "SYSTEM" not in ollama_mod.build_modelfile("m")


class TestDeploy:
    def test_deploy_creates_and_returns_handle(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            ollama_mod,
            "_ollama_create",
            lambda name, modelfile: captured.update(name=name, modelfile=modelfile),
        )
        handle = ollama_mod.OllamaBackend(host="127.0.0.1", port=11434).deploy(
            "outputs/smoke-merged", model_name="wm-sft", system="be terse"
        )
        assert isinstance(handle, ServingHandle)
        assert handle.backend == "ollama"
        assert handle.model_name == "wm-sft"
        assert handle.base_url == "http://127.0.0.1:11434"
        assert captured["name"] == "wm-sft"
        assert "FROM outputs/smoke-merged" in captured["modelfile"]

    def test_existing_local_path_is_made_absolute(self, tmp_path, monkeypatch):
        (tmp_path / "merged").mkdir()
        monkeypatch.chdir(tmp_path)
        captured = {}
        monkeypatch.setattr(
            ollama_mod, "_ollama_create", lambda name, mf: captured.update(mf=mf)
        )
        ollama_mod.OllamaBackend().deploy("merged", model_name="wm")
        from_line = captured["mf"].splitlines()[0]
        assert from_line == f"FROM {tmp_path / 'merged'}"  # absolute, not "FROM merged"

    def test_is_a_serve_backend(self):
        assert isinstance(ollama_mod.OllamaBackend(), ServeBackend)


class TestDelete:
    def test_delete_removes_model(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(ollama_mod, "_ollama_rm", lambda name: captured.update(name=name))
        handle = ServingHandle(backend="ollama", model_name="wm-sft", base_url="http://x")
        ollama_mod.OllamaBackend().delete(handle)
        assert captured["name"] == "wm-sft"
