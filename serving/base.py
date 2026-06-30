"""ServeBackend — the interface every serving backend implements.

A backend provisions a model (HF id, local dir, or S3 URI) on its target and
returns a ServingHandle describing how to reach it. The handle hands back an
eval.model client rather than defining a new inference client here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ServingHandle:
    """Describes a provisioned model endpoint.

    base_url      — set for OpenAI-compatible servers (vLLM, Ollama).
    endpoint_name — set for SageMaker endpoints (invoked via boto3, not a URL).
    extra         — backend-specific bookkeeping (process handle, container id…).
    """

    backend: str
    model_name: str
    base_url: str | None = None
    endpoint_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def openai_client(self):
        """Return an eval.model.OpenAICompatBackend for OpenAI-compatible backends."""
        if not self.base_url:
            raise ValueError(
                f"{self.backend} handle has no base_url; openai_client() is only for "
                "OpenAI-compatible servers (vLLM/Ollama)"
            )
        from eval.model import OpenAICompatBackend

        return OpenAICompatBackend(base_url=self.base_url, model=self.model_name)


class ServeBackend(ABC):
    @abstractmethod
    def deploy(self, model_ref: str, **kwargs) -> ServingHandle:
        """Provision model_ref and return a handle to reach it."""

    @abstractmethod
    def delete(self, handle: ServingHandle) -> None:
        """Tear down a previously deployed endpoint."""
