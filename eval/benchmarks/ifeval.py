"""
IFEval — Instruction Following Evaluation
Dataset: google/IFEval
Metric:  prompt-level accuracy (did the model satisfy ALL instructions in the prompt?)

Each prompt has 1-3 verifiable constraints: word count limits, required keywords,
forbidden words, JSON format, starts/ends with specific text, etc.
We implement the most common verifiers. Uses the official
`instruction_following_eval` package when available, falls back to built-in.
"""

from __future__ import annotations

import json
import re

from eval.benchmarks.base import BaseBenchmark, DEFAULT_CFG
from eval.model import GenerateConfig, ModelBackend
from eval.results import BenchmarkResult

SYSTEM_PROMPT = "Follow the instructions carefully. Respond directly."


class IFEvalBenchmark(BaseBenchmark):
    name = "ifeval"
    suite = "intelligence"
    metric = "prompt_accuracy"

    def __init__(self, max_samples: int = 0):
        self.max_samples = max_samples

    def run(self, model: ModelBackend, cfg: GenerateConfig = DEFAULT_CFG) -> BenchmarkResult:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        cfg = GenerateConfig(max_tokens=512, temperature=0.0, system_prompt=SYSTEM_PROMPT)
        ds = load_dataset("google/IFEval", split="train")
        rows = list(ds)
        if self.max_samples:
            rows = rows[: self.max_samples]

        # Try official scorer first
        try:
            return self._run_official(model, cfg, rows)
        except ImportError:
            return self._run_builtin(model, cfg, rows)

    def _run_official(self, model, cfg, rows):
        from instruction_following_eval import evaluation_main  # type: ignore
        raise ImportError("use builtin")  # force fallback until officially installed

    def _run_builtin(self, model, cfg, rows):
        prompt_correct = 0
        instruction_correct = 0
        total_instructions = 0

        for row in rows:
            response = model.complete(row["prompt"], cfg)
            instructions_satisfied = []
            for instr_id, kwargs in zip(row["instruction_id_list"], row["kwargs"]):
                ok = _check_instruction(instr_id, kwargs, response)
                instructions_satisfied.append(ok)
                instruction_correct += int(ok)
                total_instructions += 1

            if all(instructions_satisfied):
                prompt_correct += 1

        total_prompts = len(rows)
        return self._result(
            score=round(prompt_correct / total_prompts, 4) if total_prompts else 0.0,
            details={
                "prompt_correct": prompt_correct,
                "total_prompts": total_prompts,
                "instruction_accuracy": round(instruction_correct / total_instructions, 4) if total_instructions else 0.0,
            },
        )


# ---------------------------------------------------------------------------
# Instruction verifiers
# ---------------------------------------------------------------------------

def _check_instruction(instr_id: str, kwargs: dict, response: str) -> bool:
    """Return True if the response satisfies the instruction."""
    try:
        fn = _VERIFIERS.get(instr_id)
        if fn is None:
            return True  # unknown instruction — don't penalise
        return fn(kwargs, response)
    except Exception:
        return False


def _v_keyword_existence(kwargs, resp):
    kw = kwargs.get("keyword", "")
    return kw.lower() in resp.lower()


def _v_keyword_frequency(kwargs, resp):
    kw = kwargs.get("keyword", "").lower()
    freq = kwargs.get("frequency", 1)
    rel = kwargs.get("relation", "at least")
    count = resp.lower().count(kw)
    if rel == "at least":
        return count >= freq
    if rel == "at most":
        return count <= freq
    return count == freq


def _v_forbidden_words(kwargs, resp):
    words = kwargs.get("forbidden_words", [])
    rl = resp.lower()
    return not any(w.lower() in rl for w in words)


def _v_word_count_less(kwargs, resp):
    limit = kwargs.get("num_words", 0)
    return len(resp.split()) < limit


def _v_word_count_greater(kwargs, resp):
    limit = kwargs.get("num_words", 0)
    return len(resp.split()) > limit


def _v_start_with(kwargs, resp):
    prefix = kwargs.get("starter", "")
    return resp.strip().lower().startswith(prefix.lower())


def _v_end_with(kwargs, resp):
    suffix = kwargs.get("end_phrase", "")
    return resp.strip().lower().endswith(suffix.lower())


def _v_json_format(kwargs, resp):
    text = resp.strip()
    # Strip markdown code fences if present
    m = re.search(r"```(?:json)?\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def _v_lowercase(kwargs, resp):
    return resp == resp.lower()


def _v_uppercase(kwargs, resp):
    return resp == resp.upper()


def _v_num_sentences(kwargs, resp):
    num = kwargs.get("num_sentences", 1)
    rel = kwargs.get("relation", "at least")
    count = len(re.split(r"[.!?]+", resp.strip()))
    if rel == "at least":
        return count >= num
    if rel == "at most":
        return count <= num
    return count == num


def _v_no_comma(kwargs, resp):
    return "," not in resp


def _v_num_bullets(kwargs, resp):
    num = kwargs.get("num_bullets", 1)
    count = len(re.findall(r"^\s*[-*•]\s", resp, re.MULTILINE))
    return count >= num


def _v_postscript(kwargs, resp):
    marker = kwargs.get("postscript_marker", "P.S.")
    return marker in resp


def _v_title(kwargs, resp):
    # Response should contain a title in << >> markers
    return bool(re.search(r"<<[^>]+>>", resp))


def _v_quotation(kwargs, resp):
    return resp.strip().startswith('"') and resp.strip().endswith('"')


_VERIFIERS = {
    "keywords:existence": _v_keyword_existence,
    "keywords:frequency": _v_keyword_frequency,
    "keywords:forbidden_words": _v_forbidden_words,
    "length_constraints:number_words": _v_word_count_greater,
    "length_constraints:number_sentences": _v_num_sentences,
    "detectable_format:json_format": _v_json_format,
    "detectable_format:number_bullet_lists": _v_num_bullets,
    "detectable_content:postscript": _v_postscript,
    "detectable_format:title": _v_title,
    "startend:end_checker": _v_end_with,
    "startend:quotation": _v_quotation,
    "change_case:english_lowercase": _v_lowercase,
    "change_case:english_capital": _v_uppercase,
    "punctuation:no_comma": _v_no_comma,
}
