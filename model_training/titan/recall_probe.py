"""
Stage-A memory recall probe for a trained Titan (the go/no-go on the memory claim).

Buries K `key -> value` facts in one context, then asks for ONE fact's value,
and measures whether the model retrieves it. Sweeps the target fact's DEPTH (how
far back it sits) into buckets. Working long-term memory shows up as recall that
stays high in the FAR buckets — i.e. for facts placed beyond the local attention
window, reachable only via the test-time neural memory. Near-only recall = the
model is just using attention, not memory.

Long context matters: pick --n-facts so the prompt exceeds the model's
window_size (~512) — otherwise every fact is in-window and the probe can't
distinguish memory from attention. At ~10 tokens/fact, 150 facts ≈ 1500 tokens.
Keep the full prompt <= --seq-len or early facts (incl. the target) get truncated.

Names are procedurally generated from an EVAL-only syllable pool DISJOINT from
recall_data.py's TRAIN pool → we measure generalization of the retrieval skill,
never memorization. Chance ≈ 0 (random 4-digit codes), so sustained far-recall
is real signal.

Usage:
  python recall_probe.py <ckpt.pt> --tokenizer-dir DIR [--d-model .. --n-layers ..] \
      [--n-facts 150 --samples-per-bucket 6 --seq-len 2048]
  TITAN_CPU=1 python recall_probe.py ...   # force CPU (e.g. GPU busy)
"""
import argparse
import os
import random

import torch
from tokenizers import ByteLevelBPETokenizer
from titans_trainer import TitansConfig, TitansModel

HERE = os.path.dirname(os.path.abspath(__file__))

# EVAL-only syllables. MUST stay disjoint from recall_data.py's TRAIN pool.
_EVAL_A = ["bra", "cly", "dwe", "emo", "fro", "gna", "hul", "ith", "joa", "kra",
           "mlo", "nue", "osh", "pli", "rho", "sko", "tyr", "urn", "vla", "wyn"]
_EVAL_B = ["thor", "wik", "zal", "cor", "mek", "tuv", "lys", "ban", "dox", "fer",
           "gam", "hop", "jyl", "kesh", "mor", "pax"]


def _make_names(n, rng):
    names = set()
    while len(names) < n:
        names.add((rng.choice(_EVAL_A) + rng.choice(_EVAL_B)).capitalize())
    return list(names)


def build_sample(n_facts, target_idx, rng):
    """Fresh example each call. Returns (prompt, gold_code). Prompt ends at
    'The secret code for <target> is' — the model must complete the code."""
    names = _make_names(n_facts, rng)
    codes, used = {}, set()
    for nm in names:
        c = f"{rng.randint(1000, 9999)}"
        while c in used:
            c = f"{rng.randint(1000, 9999)}"
        used.add(c)
        codes[nm] = c
    facts = [f"The secret code for {nm} is {codes[nm]}." for nm in names]
    target = names[target_idx]
    prompt = " ".join(facts) + f" The secret code for {target} is"
    return prompt, codes[target]


def load_model(ckpt, tok_dir, d_model, n_layers, n_heads, seq_len, vocab, device):
    tok = ByteLevelBPETokenizer(
        os.path.join(tok_dir, "vocab.json"), os.path.join(tok_dir, "merges.txt")
    )
    cfg = TitansConfig(vocab_size=vocab, d_model=d_model, n_layers=n_layers,
                       n_heads=n_heads, max_seq_len=seq_len, causal=True, chunk_size=128)
    model = TitansModel.from_config(cfg).to(device).eval()
    # trusted local checkpoint we produced; weights_only=False needed for trainer-format ckpts
    ck = torch.load(ckpt, map_location=device, weights_only=False)  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    model.load_state_dict(sd)
    return model, tok


@torch.no_grad()
def answer(model, tok, prompt, seq_len, device, max_new=8):
    ids = tok.encode(prompt).ids
    if len(ids) > seq_len - max_new:
        # would truncate early facts (incl. the target) — caller should lower --n-facts
        ids = ids[-(seq_len - max_new):]
    x = torch.tensor([ids], device=device)
    out = []
    for _ in range(max_new):
        logits = model(x[:, -seq_len:])["logits"][0, -1]
        nxt = int(logits.argmax())  # greedy — we want the model's best recall
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tokenizer-dir", default=os.path.join(HERE, "tokenizer"))
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--n-layers", type=int, default=14)
    ap.add_argument("--n-heads", type=int, default=12)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--n-facts", type=int, default=150, help="facts/context; size so prompt > window (~512) and <= seq-len")
    ap.add_argument("--samples-per-bucket", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cpu" if os.environ.get("TITAN_CPU") == "1" else ("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    model, tok = load_model(args.ckpt, args.tokenizer_dir, args.d_model, args.n_layers,
                            args.n_heads, args.seq_len, args.vocab_size, device)
    approx_tok = args.n_facts * 10
    print(f"loaded {args.ckpt} on {device} | n_facts={args.n_facts} (~{approx_tok} tok ctx, "
          f"window≈512, seq_len={args.seq_len})\n")

    labels = ["0-20% (far)", "20-40%", "40-60%", "60-80%", "80-100% (near)"]
    buckets = {b: [] for b in labels}
    n = args.n_facts
    for bi, label in enumerate(labels):
        for s in range(args.samples_per_bucket):
            frac = (bi + (s + 0.5) / args.samples_per_bucket) / 5.0   # depth inside this bucket
            target_idx = min(n - 1, int(frac * n))
            prompt, gold = build_sample(n, target_idx, rng)
            buckets[label].append(gold in answer(model, tok, prompt, args.seq_len, device))

    print(f"{'depth of target fact':<20} {'recall':>8}   (gold value survives at this distance)")
    print("-" * 52)
    for b in labels:
        hits = buckets[b]
        print(f"{b:<20} {sum(hits)/len(hits):>7.0%}   ({sum(hits)}/{len(hits)})")
    overall = [h for hs in buckets.values() for h in hs]
    print("-" * 52)
    print(f"{'OVERALL':<20} {sum(overall)/max(1,len(overall)):>7.0%}   ({sum(overall)}/{len(overall)})")
    print("\nRead: high recall in the FAR buckets = the test-time memory is carrying facts "
          "beyond the ~512 attention window. Near-only recall = attention, not memory.")


if __name__ == "__main__":
    main()
