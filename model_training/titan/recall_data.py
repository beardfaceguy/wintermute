"""
Synthetic recall TRAINING data generator for Stage A.

Emits self-contained needle-in-haystack examples (one per line) that TEACH a
Titan to retrieve a value bound to a key earlier in the context — the exact
skill recall_probe.py measures. Each example:

    "The secret code for <NAME_0> is <CODE_0>. ... [K facts] ...
     The secret code for <TARGET> is <CODE_target>."

The TARGET's fact is placed EARLY (within the first `target_frac` of the K
facts), then many filler facts push it well beyond the model's ~512-token local
attention window, and the example ends by re-stating the target's fact. The LM
loss on that final <CODE> is the retrieval training signal: to predict it, the
model must have stored the binding when it first saw it — i.e. use its memory.

Keys/values are procedurally generated from a TRAIN-only syllable pool that is
DISJOINT from recall_probe.py's EVAL pool, so the probe measures generalization
of the retrieval *skill*, never memorization of specific facts.
"""
import argparse
import random

# TRAIN-only syllables. MUST stay disjoint from recall_probe.py's EVAL pool.
_TRAIN_A = ["ka", "ve", "zo", "mu", "ri", "del", "fen", "gor", "hix", "jun",
            "nab", "pyr", "quo", "sen", "tor", "umi", "vex", "wob", "xan", "yur", "zeb", "lo"]
_TRAIN_B = ["dor", "lix", "mun", "tas", "vel", "run", "pha", "gix", "som",
            "nix", "bel", "dan", "fyx", "lom", "pud", "ryn"]


def _make_names(n, rng):
    names = set()
    while len(names) < n:
        names.add((rng.choice(_TRAIN_A) + rng.choice(_TRAIN_B)).capitalize())
    return list(names)


def make_example(k, target_frac, rng):
    names = _make_names(k, rng)
    codes, used = {}, set()
    for nm in names:
        c = f"{rng.randint(1000, 9999)}"
        while c in used:
            c = f"{rng.randint(1000, 9999)}"
        used.add(c)
        codes[nm] = c
    facts = [f"The secret code for {nm} is {codes[nm]}." for nm in names]
    ti = rng.randint(0, max(0, int(k * target_frac)))  # target sits early -> far from the query
    target = names[ti]
    return " ".join(facts) + f" The secret code for {target} is {codes[target]}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="recall_corpus.txt")
    ap.add_argument("--n-examples", type=int, default=20000)
    ap.add_argument("--facts", type=int, default=150, help="facts/example; ~10 tok each → sets context length")
    ap.add_argument("--target-frac", type=float, default=0.2, help="target within first frac of facts (kept far from query)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    total_words = 0
    with open(args.out, "w") as f:
        for _ in range(args.n_examples):
            ex = make_example(args.facts, args.target_frac, rng)
            f.write(ex + "\n")
            total_words += len(ex.split())
    avg_w = total_words // max(1, args.n_examples)
    print(f"wrote {args.n_examples} examples → {args.out}")
    print(f"avg ~{avg_w} words/example (~{int(avg_w * 1.3)} tokens est.); "
          f"target in first {args.target_frac:.0%} of {args.facts} facts (far from query, beyond a 512-token window)")


if __name__ == "__main__":
    main()
