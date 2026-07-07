"""
Stage-A memory recall probe for a trained Titan (the go/no-go on the memory claim).

Injects many `key -> value` facts into one context, then asks for ONE fact's
value, and measures whether the model retrieves it. Sweeps the target fact's
DEPTH (how far back it sits) to produce a recall-vs-distance curve. Working
long-term memory shows up as recall that stays high when the fact is far from
the query — i.e. beyond what recent-token attention alone could carry.

Chance level is ~0 (values are random 4-digit codes), so any sustained recall at
depth is signal.

--------------------------------------------------------------------------------
HONEST CAVEAT ON SUBSTRATE
--------------------------------------------------------------------------------
This only yields a *meaningful* signal on a model trained (a) at long context and
(b) ideally with some recall-structured data in the mix. Our first models
(TinyStories, seq 512, never asked to recall anything) are a WEAK substrate:
short context + no learned reason to use the memory. The harness is reusable and
correct; expect low numbers until we train a longer-context, recall-aware Titan.
Treat early results as a wiring check, not a verdict on the architecture.
--------------------------------------------------------------------------------

Usage:
  python recall_probe.py <ckpt.pt> [--tokenizer-dir DIR] [--d-model 768 --n-layers 14 --n-heads 12]
"""
import argparse
import os

import torch
from tokenizers import ByteLevelBPETokenizer
from titans_trainer import TitansConfig, TitansModel

HERE = os.path.dirname(os.path.abspath(__file__))

# deterministic pools (no RNG import needed; index arithmetic keeps runs reproducible)
NAMES = [
    "Zorp", "Blivet", "Quill", "Marlo", "Fenn", "Ograh", "Vesper", "Tato",
    "Nix", "Wren", "Doon", "Jasper", "Kite", "Lumo", "Pell", "Brack",
    "Sable", "Thorne", "Vim", "Yolk", "Cobb", "Dune", "Ferro", "Gale",
]


def _code(i):
    # 4-digit code deterministic per index; distinct across facts in a sample
    return f"{1000 + (i * 3607) % 9000}"


def build_sample(n_facts, target_idx):
    """Return (prompt, gold_code). One fact per name; target placed at target_idx."""
    facts = [f"The secret code for {NAMES[i]} is {_code(i)}." for i in range(n_facts)]
    context = " ".join(facts)
    target_name = NAMES[target_idx]
    prompt = f"{context} The secret code for {target_name} is"
    return prompt, _code(target_idx)


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
    ids = tok.encode(prompt).ids[-(seq_len - max_new):]
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
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--n-facts", type=int, default=20, help="facts per context (context length knob)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.ckpt, args.tokenizer_dir, args.d_model, args.n_layers,
                            args.n_heads, args.seq_len, args.vocab_size, device)
    print(f"loaded {args.ckpt} on {device} | n_facts={args.n_facts}\n")

    # sweep target depth across the context; bucket recall by how far back the fact sits
    buckets = {"0-20% (far)": [], "20-40%": [], "40-60%": [], "60-80%": [], "80-100% (near)": []}
    labels = list(buckets)
    n = min(args.n_facts, len(NAMES))
    for target_idx in range(n):
        depth = target_idx / max(1, n - 1)          # 0 = earliest/farthest from query
        bucket = labels[min(4, int(depth * 5))]
        prompt, gold = build_sample(n, target_idx)
        got = answer(model, tok, prompt, args.seq_len, device)
        hit = gold in got
        buckets[bucket].append(hit)

    print(f"{'depth of target fact':<20} {'recall':>8}   (gold value survives at this distance)")
    print("-" * 52)
    for b in labels:
        hits = buckets[b]
        if hits:
            print(f"{b:<20} {sum(hits)/len(hits):>7.0%}   ({sum(hits)}/{len(hits)})")
    overall = [h for hs in buckets.values() for h in hs]
    print("-" * 52)
    print(f"{'OVERALL':<20} {sum(overall)/max(1,len(overall)):>7.0%}   ({sum(overall)}/{len(overall)})")
    print("\nRead: high recall in the 'far' buckets = the test-time memory is "
          "carrying facts beyond recent attention. Near-only recall = attention, not memory.")


if __name__ == "__main__":
    main()
