"""
Data stage: download a TinyStories subset and train the BPE tokenizer.

Also sweeps several vocab sizes and prints compression (chars/token) — on
TinyStories this plateaus immediately, so we keep the smallest (8k). Outputs
land in $TITAN_WORK (default ./titan_work): tinystories_corpus.txt and tok_<V>/.
"""
import os
import time
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

WORK = os.environ.get("TITAN_WORK", "titan_work")
os.makedirs(WORK, exist_ok=True)
CORPUS = os.path.join(WORK, "tinystories_corpus.txt")
N_STORIES = int(os.environ.get("TITAN_N_STORIES", "300000"))
VOCAB_SWEEP = [8000, 16000, 32000, 49152]
CHOSEN_VOCAB = 8000  # the sweep below shows compression plateaus here

SPECIAL = ["<pad>", "<s>", "</s>", "<unk>"]


def build_corpus():
    if os.path.exists(CORPUS) and os.path.getsize(CORPUS) > 1000:
        print(f"corpus exists: {os.path.getsize(CORPUS)/1e6:.0f} MB")
        return
    print(f"downloading {N_STORIES} TinyStories (streaming)...", flush=True)
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    t0 = time.time()
    with open(CORPUS, "w") as f:
        for i, ex in enumerate(ds):
            if i >= N_STORIES:
                break
            f.write(" ".join(ex["text"].split()) + "\n")
            if i % 50000 == 0:
                print(f"  {i} stories, {time.time()-t0:.0f}s", flush=True)
    print(f"corpus: {os.path.getsize(CORPUS)/1e6:.0f} MB", flush=True)


def train_tokenizer(vocab_size, sample):
    outdir = os.path.join(WORK, f"tok_{vocab_size}")
    os.makedirs(outdir, exist_ok=True)
    tok = ByteLevelBPETokenizer()
    t0 = time.time()
    tok.train(files=[CORPUS], vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL)
    tt = time.time() - t0
    tok.save_model(outdir)
    cpt = len(sample) / len(tok.encode(sample).ids)
    print(f"  vocab={vocab_size:>6}  train={tt:>4.0f}s  chars/token={cpt:.2f}  -> {outdir}", flush=True)
    return outdir


if __name__ == "__main__":
    build_corpus()
    with open(CORPUS) as f:
        f.seek(max(0, os.path.getsize(CORPUS) - 1_000_000))
        sample = f.read()
    print("\n=== BPE vocab sweep (compression) ===")
    for v in VOCAB_SWEEP:
        train_tokenizer(v, sample)
    print(f"\nUsing vocab={CHOSEN_VOCAB} for pretraining (compression plateaus here).")
