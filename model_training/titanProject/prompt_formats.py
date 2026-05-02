from typing import Dict, List, Optional


CHAT_PROMPTS: List[str] = [
    "User: Hello there. Can you introduce yourself in one sentence? Assistant:",
    "User: Tell me a short story about a blue cat and a red kite. Assistant:",
    "User: What is 2 plus 2? Assistant:",
]


INSTRUCTION_CASES: List[Dict[str, str]] = [
    {"instruction": "Introduce yourself in one sentence.", "input": ""},
    {"instruction": "Tell me a short story about a blue cat and a red kite.", "input": ""},
    {"instruction": "What is 2 plus 2?", "input": ""},
]


def render_instruction_prompt(instruction_text: str, input_text: str = "") -> str:
    instruction = str(instruction_text or "").strip()
    input_block = str(input_text or "").strip()
    if not instruction:
        raise ValueError("Instruction text must be non-empty")
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}"
    )
    if input_block:
        prompt += f"\n\n### Input:\n{input_block}"
    prompt += "\n\n### Response:\n"
    return prompt


def infer_prompt_family(cfg: dict) -> str:
    data_cfg = cfg.get("data", {})
    train_path = str(data_cfg.get("train_path", ""))
    val_path = str(data_cfg.get("val_path", ""))
    combined = f"{train_path} {val_path}".lower()
    if ".jsonl" in combined or "instruction" in combined:
        return "instruction"
    return "chat"


def default_prompts(prompt_family: str) -> List[str]:
    if prompt_family == "chat":
        return list(CHAT_PROMPTS)
    if prompt_family == "instruction":
        return [
            render_instruction_prompt(case["instruction"], case.get("input", ""))
            for case in INSTRUCTION_CASES
        ]
    raise ValueError(f"Unsupported prompt family: {prompt_family}")


def default_stop_strings(prompt_family: str) -> List[str]:
    if prompt_family == "chat":
        return ["\nUser:", "\nAssistant:"]
    if prompt_family == "instruction":
        return ["\n### Instruction:", "\n### Input:", "\n### Response:"]
    raise ValueError(f"Unsupported prompt family: {prompt_family}")


def _strip_once(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :].lstrip()
    return text


def _extract_instruction_parts(prompt: str) -> Dict[str, str]:
    marker = "### Instruction:\n"
    if marker not in prompt:
        return {"instruction": "", "input": ""}
    body = prompt.split(marker, 1)[1]
    if "\n\n### Input:\n" in body:
        instruction, rest = body.split("\n\n### Input:\n", 1)
        if "\n\n### Response:\n" in rest:
            input_text, _ = rest.split("\n\n### Response:\n", 1)
        else:
            input_text = rest
        return {"instruction": instruction.strip(), "input": input_text.strip()}
    if "\n\n### Response:\n" in body:
        instruction, _ = body.split("\n\n### Response:\n", 1)
    else:
        instruction = body
    return {"instruction": instruction.strip(), "input": ""}


def extract_completion(
    decoded_text: str,
    *,
    prompt: str,
    prompt_family: str,
    user_prefix: str = "User:",
    assistant_prefix: str = "Assistant:",
) -> str:
    completion = decoded_text[len(prompt) :] if decoded_text.startswith(prompt) else decoded_text
    completion = completion.strip()

    if prompt_family == "chat":
        completion = _strip_once(completion, assistant_prefix)
        for marker in (f"\n{user_prefix}", f"\n{assistant_prefix}"):
            if marker in completion:
                completion = completion.split(marker, 1)[0].strip()
        return completion

    if prompt_family == "instruction":
        parts = _extract_instruction_parts(prompt)
        repeated_segments = [
            "### Response:",
            f"{parts['instruction']}\n\n### Response:" if parts["instruction"] else "",
            render_instruction_prompt(parts["instruction"], parts["input"]).strip()
            if parts["instruction"]
            else "",
        ]
        changed = True
        while changed and completion:
            changed = False
            stripped = completion.lstrip()
            for segment in repeated_segments:
                if segment and stripped.startswith(segment):
                    stripped = stripped[len(segment) :].lstrip()
                    completion = stripped
                    changed = True
            if parts["instruction"] and stripped.startswith(parts["instruction"]):
                stripped = stripped[len(parts["instruction"]) :].lstrip()
                completion = stripped
                changed = True

        for marker in ("\n### Instruction:", "\n### Input:", "\n### Response:"):
            if marker in completion:
                completion = completion.split(marker, 1)[0].strip()
        return completion.strip()

    raise ValueError(f"Unsupported prompt family: {prompt_family}")
