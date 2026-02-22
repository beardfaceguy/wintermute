# Titans run log

## 2025-12-28: Run 001 (1500-step long smoke)
- command: `python train.py --device mps --max-steps 1500 --log-every 50 --max-tokens 2000000`
- config: `config_small.yaml` (bpe_50k_bf, dim 320, depth 5, heads 6, seq_len 256, batch 12, warmup 500, cosine)
- device: `mps` (autocast/GradScaler enabled)
- data cap: 2,000,000 tokens per split (train/val windows: 7812 each)
- timing: train ~1658s to step 1500; eval finished ~1791s
- train losses (samples): step 50: 10.28; 250: 6.93; 500: 5.61; 750: 4.42; 1000: 4.08; 1250: 4.09; 1500: 4.10
- eval @1500: loss 4.1036, ppl 60.56
- notes: MPS layer-norm dtype warning (half vs float), non-blocking. LR peaked near warmup end (~0.000299) and decayed to ~0 by step 1500.

## 2025-12-28: Run 002 (3000-step full-sampled)
- command: `python train.py --device mps --max-steps 3000 --log-every 50`
- config: `config_small.yaml` updated (lr 5e-4, warmup 300, cosine; dim 320, depth 5, heads 6, seq_len 256, batch 12)
- device: `mps` (autocast/GradScaler enabled)
- data: full sampled TinyStories (train ~48.5M tokens, 189,481 windows; val ~2.4M tokens, 9,436 windows); no token cap
- timing: train ~3355s to step 3000; eval finished ~3515s (~58 minutes total)
- train losses (samples): 50: 9.82; 250: 6.22; 500: 4.18; 750: 3.98; 1000: 3.64; 1500: 3.50; 2000: 3.02; 2500: 3.18; 3000: 3.01
- eval @3000: loss 3.0895, ppl 21.97
- notes: Significant improvement vs Run 001 (eval ppl 60.6 → 22.0). LR decays to ~0 near the end; slight wobble/plateau after ~2200 steps likely due to LR floor/cosine tail. MPS layer-norm dtype warning persists (benign).

## 2025-12-28: Run 003 (3000-step, cosine min LR)
- command: `python train.py --device mps --config configs/config_lrmin.yaml --max-steps 3000 --log-every 50`
- config: `config_lrmin.yaml` (lr 5e-4, lr_min 5e-5, warmup 300, cosine; dim 320, depth 5, heads 6, seq_len 256, batch 12)
- device: `mps` (AMP default off on MPS after warning fix; autocast dtype float32 if forced on)
- data: full sampled TinyStories (train ~48.5M tokens; val ~2.4M tokens)
- timing: train ~3343s to step 3000; eval finished ~3503s (~58 minutes)
- train losses (samples): 50: 9.78; 250: 6.26; 500: 4.25; 750: 3.90; 1000: 3.67; 1500: 3.43; 2000: 3.27; 2500: 3.08; 3000: 3.07
- eval @3000: loss 3.0335, ppl 20.77
- notes: Small gain vs Run 002 (ppl 21.97 → 20.77). LR floor keeps decay from hitting zero; warning about layer-norm dtype addressed by disabling AMP on MPS by default.

## 2025-12-28: Run 004 (3000-step, dim 384)
- command: `python train.py --device mps --config configs/config_dim384.yaml --max-steps 3000 --log-every 50 --no-amp`
- config: `config_dim384.yaml` (lr 5e-4, warmup 300, cosine; dim 384, depth 5, heads 6, seq_len 256, batch 12)
- device: `mps` (AMP disabled)
- data: full sampled TinyStories (train ~48.5M tokens; val ~2.4M tokens)
- timing: train ~3341s to step 3000; eval finished ~3528s (~58 minutes)
- train losses (samples): 50: 8.72; 250: 4.49; 500: 3.42; 750: 3.00; 1000: 2.99; 1500: 2.74; 2000: 2.66; 2500: 2.65; 3000: 2.60
- eval @3000: loss 2.5838, ppl 13.25
- notes: Big gain vs Run 003 (ppl 20.77 → 13.25) from increasing dim to 384. No AMP warnings. Some late-step noise as LR → 0.

## 2025-12-28: Run 005 (3000-step, seq_len 512)
- command: `python train.py --device mps --config configs/config_seq512.yaml --max-steps 3000 --log-every 50 --no-amp`
- config: `config_seq512.yaml` (lr 5e-4, warmup 300, cosine; dim 320, depth 5, heads 6, seq_len 512, batch 8)
- device: `mps` (AMP disabled)
- data: full sampled TinyStories (train ~48.5M tokens; val ~2.4M tokens)
- timing: train ~5032s to step 3000; eval finished ~5233s (~84 minutes)
- train losses (samples): 50: 9.05; 250: 4.70; 500: 3.42; 750: 3.04; 1000: 2.89; 1500: 2.76; 2000: 2.66; 2500: 2.60; 3000: 2.62
- eval @3000: loss 2.5771, ppl 13.16
- notes: Similar perplexity to dim=384 run (13.16 vs 13.25) but slower wall-clock due to seq_len 512 and smaller batch (8). Longer context doesn’t hurt; throughput cost is notable.

## 2025-12-28: Run 006 (3000-step, slower decay with lr_min=1e-4)
- command: `python train.py --device mps --config configs/config_decay_slow.yaml --max-steps 3000 --log-every 50 --no-amp`
- config: `config_decay_slow.yaml` (lr 5e-4, lr_min 1e-4, warmup 300, cosine; dim 320, depth 5, heads 6, seq_len 256, batch 12)
- device: `mps` (AMP disabled)
- data: full sampled TinyStories (train ~48.5M tokens; val ~2.4M tokens)
- timing: train ~2982s to step 3000; eval finished ~3143s (~52 minutes)
- train losses (samples): 50: 9.24; 250: 4.83; 500: 3.39; 750: 3.30; 1000: 2.90; 1500: 2.80; 2000: 2.70; 2500: 2.71; 3000: 2.54
- eval @3000: loss 2.6019, ppl 13.49
- notes: Slower decay (lr_min 1e-4) held LR up; eval perplexity landed slightly worse than the seq512/dim384 runs (~13.5 vs 13.1–13.2). Time was faster (~52 min). Maintaining a floor helped stability but didn’t beat the best runs.

## 2025-12-28: Run 007 (4000-step, baseline with lr_min 5e-5)
- command: `python train.py --device mps --config configs/config_lrmin.yaml --max-steps 4000 --log-every 50 --no-amp`
- config: `config_lrmin.yaml` (lr 5e-4, lr_min 5e-5, warmup 300, cosine; dim 320, depth 5, heads 6, seq_len 256, batch 12)
- device: `mps` (AMP disabled)
- data: full sampled TinyStories (train ~48.5M tokens; val ~2.4M tokens)
- timing: train ~3950s to step 4000; eval finished ~4110s (~68 minutes)
- train losses (samples): 50: 9.03; 250: 4.76; 500: 3.39; 750: 3.15; 1000: 2.99; 1500: 2.68; 2000: 2.85; 2500: 2.63; 3000: 2.52; 3500: 2.66; 4000: 2.50
- eval @4000: loss 2.5236, ppl 12.47
- notes: Extending to 4k steps with a small LR floor beats prior best (prev best ppl ~13.1–13.2). LR floor prevented decay to zero, sustaining improvements past 3k. Late-step losses stayed low without instability.

## 2025-12-28: Run 008 (3000-step, dim 384 + lr_min 5e-5)
- command: `python train.py --device mps --config configs/config_dim384_lrmin.yaml --max-steps 3000 --log-every 50 --no-amp`
- config: `config_dim384_lrmin.yaml` (lr 5e-4, lr_min 5e-5, warmup 300, cosine; dim 384, depth 5, heads 6, seq_len 256, batch 12)
- device: `mps` (AMP disabled)
- data: full sampled TinyStories (train ~48.5M tokens; val ~2.4M tokens)
- timing: train ~3322s to step 3000; eval finished ~3507s (~58 minutes)
- train losses (samples): 50: 8.84; 250: 4.37; 500: 3.40; 750: 3.12; 1000: 2.93; 1500: 2.88; 2000: 2.68; 2500: 2.64; 3000: 2.56
- eval @3000: loss 2.5718, ppl 13.09
- notes: Combining dim 384 with lr_min=5e-5 did not beat the 4k baseline (ppl 12.47) and is on par with seq512 (~13.1). Dim+floor helped stability but best perplexity still comes from longer training (Run 007).

## 2025-12-29: Run 009 (4000-step, full combo: dim 384 + seq_len 512 + lr_min 5e-5)
- command: `python train.py --device mps --config configs/config_combo_all.yaml --log-every 100 --no-amp`
- config: `config_combo_all.yaml` (lr 5e-4, lr_min 5e-5, warmup 300, cosine; dim 384, depth 5, heads 6, seq_len 512, batch 6)
- device: `mps` (AMP disabled)
- data: full sampled TinyStories (train ~48.5M tokens; val ~2.4M tokens)
- timing: train ~25,780s to step 4000; eval finished ~26,015s (~7.2 hours)
- train losses (samples): 100: 6.23; 200: 4.96; 300: 3.92; 500: 3.46; 700: 3.21; 1000: 2.87; 1500: 2.76; 2000: 2.76; 2500: 2.52; 3000: 2.71; 3500: 2.45; 4000: 2.54
- eval @4000: loss 2.4904, ppl 12.07
- notes: Best perplexity to date (previous best 12.47). Runtime ~7.2h (longer due to seq512 + dim384 + smaller batch). LR floor sustained improvements through 4k.

## 2025-12-30: Run 010 (baseline no-memory, 3000-step)
- command: `python train.py --device mps --config configs/config_baseline_nomem.yaml --max-steps 3000 --log-every 100 --no-amp`
- config: `config_baseline_nomem.yaml` (no mem tokens; dim 384, depth 5, heads 6, seq_len 512, batch 6; lr 5e-4, lr_min 5e-5, warmup 300, cosine)
- device: `mps` (AMP disabled)
- data: full sampled TinyStories (train ~48.5M tokens; val ~2.4M tokens)
- timing: train ~4759s to step 3000; eval finished ~5019s (~84 minutes)
- train losses (samples): 100: 6.21; 300: 4.33; 500: 3.53; 1000: 3.06; 1500: 2.87; 2000: 2.63; 2500: 2.61; 3000: 2.48
- eval @3000: loss 2.5767, ppl 13.15
- memory test (prompt QA, span decode 3 tokens): Accuracy 0/200 (0%)

## 2025-12-30: QA finetune on Titans checkpoint (ckpt_step_4000.pt)
- command: `python finetune_memory_qa.py --device mps --config configs/config_combo_all.yaml --ckpt ckpt_step_4000.pt --steps 300 --lr 1e-4 --batch-size 6 --save ckpt_finetune_qa.pt`
- data: synthetic QA (memory_test.txt with "Answer:" prompts; 200 samples, filler 120)
- result: memory test accuracy 36/200 (18%) using `ckpt_finetune_qa.pt`
- note: baseline no-memory remains 0%; Titans improves on QA format after short finetune

## 2026-01-02: Memory test on ckpt_step_4000 (zero-shot, no finetune)
- command: `python memory_test_eval.py --device mps --config configs/config_combo_all.yaml --ckpt ckpt_step_4000.pt --data memory_test.txt --answers memory_test_answers.txt --max-answer-tokens 3`
- result: Accuracy 0/200 (0%) on the synthetic QA memory probe without additional finetuning

## 2026-01-02: QA finetune (300 steps) on ckpt_step_4000 and eval
- command: `python finetune_memory_qa.py --device mps --config configs/config_combo_all.yaml --ckpt ckpt_step_4000.pt --steps 300 --lr 1e-4 --batch-size 6 --save ckpt_finetune_qa_step4000_s300.pt`
- finetune loss samples: step 50 2.99; 100 2.84; 150 2.85; 200 2.78; 250 2.73; 300 2.60
- eval command: `python memory_test_eval.py --device mps --config configs/config_combo_all.yaml --ckpt ckpt_finetune_qa_step4000_s300.pt --data memory_test.txt --answers memory_test_answers.txt --max-answer-tokens 3`
- result: Accuracy 42/200 (21.0%) on the synthetic QA memory probe

### Decode span sensitivity (same finetuned checkpoint)
- command: `python memory_test_eval.py --device mps --config configs/config_combo_all.yaml --ckpt ckpt_finetune_qa_step4000_s300.pt --data memory_test.txt --answers memory_test_answers.txt --max-answer-tokens 6`
- result: Accuracy 51/200 (25.5%)

## 2026-01-20: Longer QA finetune on Titans checkpoint (ckpt_step_4000.pt)
- command: `python finetune_memory_qa.py --device mps --config configs/config_combo_all.yaml --ckpt ckpt_step_4000.pt --steps 1000 --lr 5e-5 --batch-size 6 --save ckpt_finetune_qa_long.pt`
- finetune loss samples: 50 3.39; 100 2.96; 200 2.84; 400 2.67; 600 2.10; 800 1.44; 1000 0.86
- eval command: `python memory_test_eval.py --device mps --config configs/config_combo_all.yaml --ckpt ckpt_finetune_qa_long.pt --data memory_test.txt --answers memory_test_answers.txt --max-answer-tokens 6`
- result: Accuracy 85/200 (42.50%) on the synthetic QA memory probe

### Baseline (no-memory config) eval with tolerant load
- command: `python memory_test_eval.py --device mps --config configs/config_baseline_nomem.yaml --ckpt ckpt_step_4000.pt --data memory_test.txt --answers memory_test_answers.txt --max-answer-tokens 6`
- note: baseline config disables longterm mem tokens; checkpoint has mem params, loaded with mem keys dropped (strict=False)
- result: Accuracy 21/200 (10.50%) on the synthetic QA memory probe

## Memory test plan (Titans vs baseline)
- Goal: show Titans test-time memory benefit on a short-context factual recall task.
- Task: synthetic samples with a “fact” sentence, long filler, then a question (e.g., “Alice’s cat is blue. ... What color is Alice’s cat?”) so the query sits beyond local context.
- Metrics: exact-match accuracy on the answer token(s); optionally loss/perplexity on the answer span.
- Models: Titans checkpoint (`ckpt_step_4000.pt`, dim384/seq512); baseline ideally with memory disabled (e.g., same config but `num_longterm_mem_tokens=0`) or a separate baseline checkpoint.
- Eval harness: script to load tokenizer + checkpoint, run forward on the synthetic set, extract predicted answer tokens, report accuracy. Use seq_len 512, small batch (8–12).
- Next steps: (1) generate synthetic dataset (`memory_test_gen.py`), (2) add eval script (`memory_test_eval.py`), (3) run Titans vs baseline and compare accuracy.

### Next options
- Longer finetune on memory QA (more steps or lower LR)
- Longer decode span during memory eval
- No-memory baseline comparison under same decode settings

### Next test 1: Longer finetune on memory QA (planned)
- intent: extend QA-format finetune on `ckpt_step_4000.pt` to see if accuracy improves beyond 25.5%
- planned train cmd: `python finetune_memory_qa.py --device mps --config configs/config_combo_all.yaml --ckpt ckpt_step_4000.pt --steps 1000 --lr 5e-5 --batch-size 6 --save ckpt_finetune_qa_long.pt`
- planned eval cmd: `python memory_test_eval.py --device mps --config configs/config_combo_all.yaml --ckpt ckpt_finetune_qa_long.pt --data memory_test.txt --answers memory_test_answers.txt --max-answer-tokens 6`
- status: not started

### Checklist (Raschka build-from-scratch) — status
- [x] BPE tokenizer with stable vocab; consistent train/eval paths
- [x] Sliding-window sampling, train/val split fixed, token caps for smoke
- [x] GPT decoder stack with causal mask, MH attention, GELU MLPs, residuals, layer norm
- [x] Positional encodings applied for target seq_len; LM head tied? (Titans ties embeddings internally)
- [x] AdamW, warmup + cosine with LR floor; grad clip; AMP off on MPS for stability
- [x] Regular checkpoints with config and tokenizer reference
- [x] Train/val loss + perplexity logged each eval
- [x] Basic generation sanity checks (greedy); decode span toggles for QA eval
- [x] Top-k/temperature decoding option exposed in generate script
- [x] Supervised finetune aligned with tokenizer/context; QA prompts “Answer:”
- [ ] Optional PEFT/LoRA (not needed for current tiny model)
- [ ] HF/transformers export script (not yet needed)