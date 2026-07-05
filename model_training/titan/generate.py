"""
Load a Titan checkpoint and generate from TinyStories-style prompts.

Auto-selects CUDA if free, else CPU (a 46.6M model generates fine on CPU — useful
when the GPU is busy training). Rebuilds the model from the known architecture and
loads only the weights, so it works with trainer-format checkpoints (which store
training hyperparams that TitansModel.__init__ would reject).
"""
import os
import sys
import torch
from tokenizers import ByteLevelBPETokenizer
from titans_trainer import TitansConfig, TitansModel

WORK = os.environ.get("TITAN_WORK", "titan_work")
HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = 512
VOCAB = 8000

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORK, "titan_run", "final.pt")
device = "cuda" if (torch.cuda.is_available() and os.environ.get("TITAN_CPU") != "1") else "cpu"

# tokenizer: alongside the checkpoint > work dir > repo-shipped
_cands = [os.path.join(os.path.dirname(ckpt_path), "tokenizer"),
          os.path.join(WORK, f"tok_{VOCAB}"), os.path.join(HERE, "tokenizer")]
td = next((c for c in _cands if os.path.exists(os.path.join(c, "vocab.json"))), _cands[-1])
tok = ByteLevelBPETokenizer(os.path.join(td, "vocab.json"), os.path.join(td, "merges.txt"))
eos = tok.token_to_id("</s>")

cfg = TitansConfig(vocab_size=VOCAB, d_model=512, n_layers=8, n_heads=8,
                   max_seq_len=SEQ, causal=True, chunk_size=128)
model = TitansModel.from_config(cfg).to(device).eval()
# We only ever load our own locally-produced checkpoints. weights_only=False is
# required because trainer-format checkpoints carry config/optimizer objects that
# the safe loader rejects; the pickle-RCE risk does not apply to trusted local files.
ck = torch.load(ckpt_path, map_location=device, weights_only=False)  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
model.load_state_dict(sd)
print(f"loaded {ckpt_path} on {device} (tokenizer: {td})\n")


@torch.no_grad()
def generate(prompt, max_new=100, temp=0.8, top_k=40):
    x = torch.tensor([tok.encode(prompt).ids], device=device)
    for _ in range(max_new):
        logits = model(x[:, -SEQ:])["logits"][0, -1] / temp
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = -float("inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        x = torch.cat([x, nxt.view(1, 1)], dim=1)
        if nxt.item() == eos:
            break
    return tok.decode(x[0].tolist())


if __name__ == "__main__":
    prompts = sys.argv[2:] or [
        "Once upon a time, there was a little",
        "Tom and Sara went to the park. They saw a",
        "The cat sat on the",
        "Lily had a red ball. She wanted to",
    ]
    for p in prompts:
        print("PROMPT:", p)
        print("OUT   :", generate(p).replace("\n", " "))
        print("-" * 70)
