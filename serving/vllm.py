"""vLLM serving backend — launch an OpenAI-compatible vLLM server for a model.

deploy() spawns `python -m vllm.entrypoints.openai.api_server`, waits for the
/health endpoint, and returns a ServingHandle whose openai_client() reuses
eval.model.OpenAICompatBackend. vLLM itself is never imported here — it runs as
a subprocess — so this module loads without the vllm package installed.

The spawn and health-check are seams (_spawn_server, _wait_healthy) so the
wiring is unit-tested without launching a real server.
"""

from __future__ import annotations

import http.client
import subprocess
import sys
import time
from urllib.parse import urlparse

from serving.base import ServeBackend, ServingHandle


def build_vllm_args(
    model_ref: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    served_model_name: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the vLLM api_server CLI args for a model."""
    args = ["--model", model_ref, "--host", host, "--port", str(port)]
    if served_model_name:
        args += ["--served-model-name", served_model_name]
    if extra_args:
        args += list(extra_args)
    return args


def _spawn_server(args: list[str]) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *args]
    return subprocess.Popen(cmd)


def _wait_healthy(base_url: str, timeout: int = 300, interval: float = 2.0) -> None:
    # http.client (vs urllib) keeps this to plain HTTP against a host/port we
    # control — no file:// scheme surface to worry about.
    parsed = urlparse(base_url)
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        conn = None
        try:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
            conn.request("GET", "/health")
            if conn.getresponse().status == 200:
                return
        except Exception as e:  # noqa: BLE001 — server not up yet; keep polling
            last_err = e
        finally:
            if conn is not None:
                conn.close()
        time.sleep(interval)
    raise TimeoutError(f"vLLM server at {base_url} not healthy within {timeout}s ({last_err})")


class VLLMBackend(ServeBackend):
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.host = host
        self.port = port

    def deploy(
        self,
        model_ref: str,
        *,
        served_model_name: str | None = None,
        extra_args: list[str] | None = None,
        health_timeout: int = 300,
        **kwargs,
    ) -> ServingHandle:
        name = served_model_name or model_ref
        args = build_vllm_args(
            model_ref,
            host=self.host,
            port=self.port,
            served_model_name=name,
            extra_args=extra_args,
        )
        proc = _spawn_server(args)
        base_url = f"http://{self.host}:{self.port}"
        _wait_healthy(base_url, timeout=health_timeout)
        return ServingHandle(
            backend="vllm",
            model_name=name,
            base_url=base_url,
            extra={"process": proc},
        )

    def delete(self, handle: ServingHandle) -> None:
        proc = handle.extra.get("process")
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001 — fall back to a hard kill
            proc.kill()
