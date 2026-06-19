"""
Generate a small synthetic memory-test dataset for Titans vs baseline.

Each sample encodes a simple fact, adds filler to push the query beyond
local context, then asks a question whose answer is in the fact.
"""

import argparse
import random
from pathlib import Path

ENTITIES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"]
ATTRS = ["red", "blue", "green", "yellow", "purple", "orange", "silver", "gold"]
OBJECTS = ["cat", "dog", "car", "bike", "book", "hat", "ball", "house"]


def make_sample(rng: random.Random, filler_len: int) -> tuple[str, str]:
    name = rng.choice(ENTITIES)
    color = rng.choice(ATTRS)
    obj = rng.choice(OBJECTS)
    fact = f"{name}'s {obj} is {color}."
    filler_tokens = rng.choices(ATTRS + OBJECTS + ENTITIES, k=filler_len)
    filler = " ".join(filler_tokens) + "."
    question = f"Question: What color is {name}'s {obj}? Answer:"
    answer = color
    text = fact + " " + filler + " " + question
    return text, answer


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic memory-test samples.")
    parser.add_argument(
        "--out", type=str, default="memory_test.txt", help="Output text file (one sample per line)"
    )
    parser.add_argument(
        "--answers", type=str, default="memory_test_answers.txt", help="Answers file (one per line)"
    )
    parser.add_argument("--n", type=int, default=200, help="Number of samples")
    parser.add_argument(
        "--filler-len",
        type=int,
        default=120,
        help="Number of filler tokens between fact and question",
    )
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    ans_path = Path(args.answers)

    texts = []
    answers = []
    for _ in range(args.n):
        t, a = make_sample(rng, args.filler_len)
        texts.append(t)
        answers.append(a)

    out_path.write_text("\n".join(texts), encoding="utf-8")
    ans_path.write_text("\n".join(answers), encoding="utf-8")
    print(f"Wrote {len(texts)} samples to {out_path}")
    print(f"Wrote answers to {ans_path}")


if __name__ == "__main__":
    main()
