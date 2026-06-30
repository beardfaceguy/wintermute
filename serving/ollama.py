"""Ollama serving backend — register a model with a local Ollama daemon.

deploy() writes a Modelfile (`FROM <model dir or GGUF>`) and runs `ollama create`,
then returns a ServingHandle whose openai_client() targets Ollama's OpenAI-
compatible endpoint (reusing eval.model.OpenAICompatBackend).

The `ollama` CLI calls are seams (_ollama_create, _ollama_rm) so the wiring is
unit-tested without a running daemon. GGUF conversion of a merged HF model
(via llama.cpp) is upstream of this backend; pass either a GGUF file or a
model directory Ollama can import.
"""

from __future__ import annotations

import subprocess

from serving.base import ServeBackend, ServingHandle


def build_modelfile(model_path: str, system: str | None = None) -> str:
    """Build Modelfile text pointing at a GGUF file or importable model dir."""
    lines = [f"FROM {model_path}"]
    if system:
        lines.append(f'SYSTEM """{system}"""')
    return "\n".join(lines) + "\n"


def _ollama_create(name: str, modelfile: str) -> None:
    # `ollama create <name> -f -` reads the Modelfile from stdin.
    subprocess.run(
        ["ollama", "create", name, "-f", "-"],
        input=modelfile,
        text=True,
        check=True,
    )


def _ollama_rm(name: str) -> None:
    subprocess.run(["ollama", "rm", name], check=True)


class OllamaBackend(ServeBackend):
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.host = host
        self.port = port

    def deploy(
        self,
        model_ref: str,
        *,
        model_name: str | None = None,
        system: str | None = None,
        **kwargs,
    ) -> ServingHandle:
        name = model_name or "wintermute-sft"
        _ollama_create(name, build_modelfile(model_ref, system=system))
        return ServingHandle(
            backend="ollama",
            model_name=name,
            base_url=f"http://{self.host}:{self.port}",
        )

    def delete(self, handle: ServingHandle) -> None:
        _ollama_rm(handle.model_name)
