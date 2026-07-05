"""
Model backend abstraction for the eval harness.

Backends:
  - OpenAICompatBackend: any OpenAI-compatible endpoint — vLLM, Ollama,
    llama.cpp, OpenAI, Gemini, Groq, Together, Mistral AI, etc.
  - AnthropicBackend: Claude models via the Anthropic SDK.
  - HFLocalBackend: local HuggingFace checkpoint via transformers pipeline.

Usage:
  from eval.model import make_backend

  # Self-hosted (Ollama / vLLM)
  model = make_backend("http://192.168.8.157:11434", "mistral:7b")

  # OpenAI
  model = make_backend("openai:gpt-4o")
  model = make_backend("openai:gpt-4o-mini")

  # Anthropic / Claude
  model = make_backend("anthropic:claude-sonnet-4-6")
  model = make_backend("anthropic:claude-opus-4-8")

  # Google Gemini (OpenAI-compat endpoint)
  model = make_backend("gemini:gemini-2.5-pro")

  # Groq (fast inference, OpenAI-compat)
  model = make_backend("groq:llama-3.3-70b-versatile")

  # Together AI (OpenAI-compat)
  model = make_backend("together:meta-llama/Llama-3-70b-chat-hf")

  # Mistral AI (OpenAI-compat)
  model = make_backend("mistral:mistral-large-latest")

  # Local HF checkpoint
  model = make_backend("hf:/mnt/checkpoints/my_sft_step3000")

API keys are loaded from env vars (or wintermute/.env):
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY,
  GROQ_API_KEY, TOGETHER_API_KEY, MISTRAL_API_KEY
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

_THINK_RE = re.compile(r"<think>.*?</think>|<think>.*$", re.DOTALL)


@dataclass
class GenerateConfig:
    max_tokens: int = 512
    temperature: float = 0.0  # deterministic by default for benchmarks
    top_p: float = 1.0
    system_prompt: str | None = None


class ModelBackend(ABC):
    """Minimal interface every backend must implement."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Human-readable identifier stored in results."""

    @abstractmethod
    def chat(self, messages: list[dict], cfg: GenerateConfig) -> str:
        """Send a messages list and return the assistant reply text."""

    def complete(self, prompt: str, cfg: GenerateConfig) -> str:
        """Convenience wrapper: single user turn."""
        return self.chat([{"role": "user", "content": prompt}], cfg)


# ---------------------------------------------------------------------------
# OpenAI-compatible backend
# ---------------------------------------------------------------------------


class OpenAICompatBackend(ModelBackend):
    """
    Talks to any /v1/chat/completions endpoint.

    base_url examples:
      "http://localhost:8010"          — local vLLM
      "http://192.168.1.5:11434"       — Ollama
      "https://api.openai.com"         — OpenAI
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "none",
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai") from None

        self._model = model
        self._client = OpenAI(
            base_url=base_url.rstrip("/") + "/v1",
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "none"),
            timeout=600.0,
        )

    @property
    def model_id(self) -> str:
        return self._model

    def chat(self, messages: list[dict], cfg: GenerateConfig) -> str:
        if cfg.system_prompt:
            messages = [{"role": "system", "content": cfg.system_prompt}] + messages
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        content = resp.choices[0].message.content or ""
        return _THINK_RE.sub("", content).strip()


# ---------------------------------------------------------------------------
# Ollama native backend (thinking control)
# ---------------------------------------------------------------------------


class OllamaBackend(ModelBackend):
    """
    Ollama via its NATIVE /api/chat endpoint (not the OpenAI-compat /v1 path).

    Why this exists: Ollama's /v1 layer ignores the `think` parameter, so there
    is no way to disable a reasoning model's <think> phase through it. On hard
    prompts qwen3/deepseek-r1 then spend the entire token budget thinking and
    return empty `content` (finish_reason=length) — scoring a false 0. The native
    /api/chat endpoint honours `think`, so `think=False` makes the model emit an
    answer directly. See Vikunja #918.

    Ollama also splits reasoning into a separate `thinking` field; we return only
    `message.content` (and strip any residual <think> for safety).
    """

    def __init__(self, base_url: str, model: str, think: bool = False):
        try:
            import httpx
        except ImportError:
            raise ImportError("pip install httpx") from None

        self._httpx = httpx
        # /api/chat is the native Ollama root path; tolerate a base_url that was
        # given with a trailing /v1 (the OpenAI-compat suffix) so we don't build
        # a bogus /v1/api/chat.
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        self._url = root + "/api/chat"
        self._model = model
        self._think = think

    @property
    def model_id(self) -> str:
        return self._model

    def chat(self, messages: list[dict], cfg: GenerateConfig) -> str:
        if cfg.system_prompt:
            messages = [{"role": "system", "content": cfg.system_prompt}] + messages
        payload = {
            "model": self._model,
            "messages": messages,
            "think": self._think,
            "stream": False,
            "options": {
                "num_predict": cfg.max_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
            },
        }
        resp = self._httpx.post(self._url, json=payload, timeout=600.0)
        resp.raise_for_status()
        content = (resp.json().get("message") or {}).get("content") or ""
        return _THINK_RE.sub("", content).strip()


# ---------------------------------------------------------------------------
# Anthropic / Claude backend
# ---------------------------------------------------------------------------


class AnthropicBackend(ModelBackend):
    """
    Claude models via the Anthropic SDK.
    pip install anthropic

    model examples: claude-sonnet-4-6, claude-opus-4-8, claude-haiku-4-5-20251001
    """

    def __init__(self, model: str, api_key: str = ""):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("pip install anthropic") from None

        self._model = model
        self._client = _anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )

    @property
    def model_id(self) -> str:
        return self._model

    def chat(self, messages: list[dict], cfg: GenerateConfig) -> str:
        kwargs: dict = dict(
            model=self._model,
            max_tokens=cfg.max_tokens,
            messages=messages,
        )
        if cfg.system_prompt:
            kwargs["system"] = cfg.system_prompt
        resp = self._client.messages.create(**kwargs)
        if not resp.content:
            return ""
        return resp.content[0].text


# ---------------------------------------------------------------------------
# Local HuggingFace backend
# ---------------------------------------------------------------------------


class HFLocalBackend(ModelBackend):
    """
    Loads a HuggingFace model locally via transformers.

    model_path: HF model ID ("mistralai/Mistral-7B-Instruct-v0.3")
                or local path ("/mnt/checkpoints/my_sft_step3000")
    device:     "auto" (default), "cpu", "cuda", "mps"
    """

    def __init__(self, model_path: str, device: str = "auto"):
        try:
            import torch
            from transformers import AutoTokenizer, pipeline
        except ImportError:
            raise ImportError("pip install transformers torch") from None

        self._model_path = model_path
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._pipeline = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=self._tokenizer,
            device_map=device,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )

    @property
    def model_id(self) -> str:
        return self._model_path

    def chat(self, messages: list[dict], cfg: GenerateConfig) -> str:
        if cfg.system_prompt:
            messages = [{"role": "system", "content": cfg.system_prompt}] + messages

        # Use apply_chat_template if available, fall back to naive join
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = (
                "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"
            )

        out = self._pipeline(
            prompt,
            max_new_tokens=cfg.max_tokens,
            temperature=cfg.temperature if cfg.temperature > 0 else None,
            do_sample=cfg.temperature > 0,
            top_p=cfg.top_p,
            return_full_text=False,
        )
        return out[0]["generated_text"].strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_backend(
    target: str,
    model: str = "",
    api_key: str = "",
    device: str = "auto",
    think: bool | None = None,
) -> ModelBackend:
    """
    target shortcuts for frontier providers (format: <provider>:<model>):
      openai:<model>      — OpenAI (OPENAI_API_KEY)
      anthropic:<model>   — Anthropic/Claude (ANTHROPIC_API_KEY)
      gemini:<model>      — Google Gemini via OpenAI-compat (GEMINI_API_KEY)
      groq:<model>        — Groq (GROQ_API_KEY)
      together:<model>    — Together AI (TOGETHER_API_KEY)
      mistral:<model>     — Mistral AI (MISTRAL_API_KEY)
      hf:<id_or_path>     — local HuggingFace model / checkpoint
      http(s)://...       — any OpenAI-compatible endpoint (--model required)

    think: if not None, an http(s) target is treated as Ollama and served via its
      native /api/chat endpoint with thinking set to this value (False disables
      the <think> phase for reasoning models — see OllamaBackend / #918). Left as
      None, http(s) targets use the OpenAI-compat /v1 path unchanged.
    """
    if target.startswith("hf:"):
        return HFLocalBackend(target[3:], device=device)

    # Named provider shortcuts
    _OPENAI_COMPAT = {
        "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
        "groq": ("https://api.groq.com/openai", "GROQ_API_KEY"),
        "together": ("https://api.together.xyz", "TOGETHER_API_KEY"),
        "mistral": ("https://api.mistral.ai", "MISTRAL_API_KEY"),
        "openai": ("https://api.openai.com", "OPENAI_API_KEY"),
    }
    if ":" in target and not target.startswith("http"):
        provider, _, model_name = target.partition(":")
        if provider == "anthropic":
            return AnthropicBackend(model_name, api_key=api_key)
        if provider in _OPENAI_COMPAT:
            base_url, env_var = _OPENAI_COMPAT[provider]
            resolved_key = api_key or os.environ.get(env_var, "")
            return OpenAICompatBackend(base_url, model=model_name, api_key=resolved_key)

    if think is not None:
        return OllamaBackend(target, model=model, think=think)
    return OpenAICompatBackend(target, model=model, api_key=api_key)
