"""
LoCoMo — Long-Context Conversation Memory
Dataset: snap-research/LoCoMo
Metric:  QA accuracy over long conversational histories

Tests whether the model can recall facts from earlier in a multi-turn
conversation. Key for validating that mcp-memory surfaces context correctly.
"""

from __future__ import annotations

from eval.benchmarks.base import DEFAULT_CFG, BaseBenchmark
from eval.model import GenerateConfig, ModelBackend
from eval.results import BenchmarkResult

SYSTEM_PROMPT = (
    "You are answering questions about a conversation. "
    "Answer concisely and directly based on the conversation provided."
)

MAX_CONTEXT_CHARS = 8000  # truncate very long histories to fit context window


class LoCoMoBenchmark(BaseBenchmark):
    name = "locomo"
    suite = "memory"
    metric = "accuracy"

    def __init__(self, max_samples: int = 0):
        self.max_samples = max_samples

    def run(self, model: ModelBackend, cfg: GenerateConfig = DEFAULT_CFG) -> BenchmarkResult:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets") from None

        cfg = GenerateConfig(max_tokens=128, temperature=0.0, system_prompt=SYSTEM_PROMPT)
        ds = load_dataset("snap-research/LoCoMo", split="test")
        rows = list(ds)
        if self.max_samples:
            rows = rows[: self.max_samples]

        correct = 0
        total = 0

        for row in rows:
            # Build conversation context
            conversation = row.get("conversation", "")
            if isinstance(conversation, list):
                conversation = "\n".join(
                    f"{turn.get('speaker', 'Speaker')}: {turn.get('text', '')}"
                    for turn in conversation
                )
            # Truncate if too long
            if len(conversation) > MAX_CONTEXT_CHARS:
                conversation = "...[truncated]...\n" + conversation[-MAX_CONTEXT_CHARS:]

            qa_pairs = row.get("qa_pairs", row.get("questions", []))
            for qa in qa_pairs:
                question = qa.get("question", "")
                answer = qa.get("answer", qa.get("ground_truth", ""))
                if not question or not answer:
                    continue

                prompt = f"Conversation:\n{conversation}\n\nQuestion: {question}\n\nAnswer:"
                response = model.complete(prompt, cfg)

                if _answer_matches(response, answer):
                    correct += 1
                total += 1

        return self._result(
            score=round(correct / total, 4) if total else 0.0,
            details={"correct": correct, "total": total},
        )


def _answer_matches(response: str, expected: str) -> bool:
    """Loose match: expected answer appears in response (case-insensitive)."""
    return expected.lower().strip() in response.lower()
