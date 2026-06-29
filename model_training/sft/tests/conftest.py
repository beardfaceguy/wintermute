"""Shared fixtures for model_training/sft tests.

sft modules are imported package-qualified (`from model_training.sft.config
import ...`) so they never collide with titanProject's flat `data`/`model`
modules when both suites run in one pytest session. No sys.path hacking needed —
the repo root is already importable under pytest.
"""

import pytest


class FakeTokenizer:
    """Minimal stand-in for a HF tokenizer's chat-template API.

    Renders messages as `<|role|>content` lines so tests can assert on
    structure without pulling in transformers.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<|{m['role']}|>{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        text = "\n".join(parts)
        return [ord(c) for c in text] if tokenize else text


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()
