## LLM Training Side-Quest – Project Plan

### Objectives
- Understand the end-to-end workflow for training a compact, general-purpose LLM.
- Stand up an experimental pipeline that can be iterated locally or on rented GPUs.
- Capture lessons learned to feed back into Wintermute’s governance memory.

### Scope & Assumptions
- Target model size: 110M–350M parameters for on-device experiments (fits 16 GB unified memory); scale to 1B+ only when cloud GPUs are available.
- Focus on supervised pretraining + light instruction tuning; RLHF out of scope for first pass.
- Primary tooling: Apple’s MLX framework for local experimentation; optionally mirror configs in PyTorch/Hugging Face when scaling to cloud.
- Compute assumed: Apple M3 Pro (16 GB unified) for prototyping; schedule cloud (A100/H100) sessions for large-batch or long trainings.

### High-Level Phases
1. **Project Setup**
   - Confirm hardware access (local + cloud) and storage needs.
   - Define success metrics (perplexity targets, evaluation tasks).
2. **Data Strategy**
   - Identify base corpora (OpenWebText2, SlimPajama, code, Wintermute docs).
   - Establish filtering, dedup, and chunking process.
   - Train or adopt tokenizer; document vocabulary decisions.
3. **Model Configuration**
   - Select architecture template compatible with MLX (e.g., GPT-NeoX-style blocks) and adapt configs.
   - Decide precision (float16/bfloat16) and optimizer (AdamW with weight decay) supported by MLX.
4. **Training Pipeline**
   - Build MLX training scripts (Dataloader, optimizer step, checkpointing).
   - Integrate logging (MLX metrics, optional Weights & Biases via Python hooks).
   - Establish checkpoint cadence and validation loop.
5. **Evaluation & Alignment**
   - Set up perplexity measurement on held-out set.
   - Integrate basic instruction tuning dataset (e.g., Alpaca-style).
   - Draft plan for human eval or automated benchmarks (MT-Bench-lite).
6. **Deployment Experiment**
   - Export MLX checkpoints to Hugging Face/PyTorch format if needed.
   - Run smoke tests via simple CLI or TalkingHead integration (Metal inference or converted model).
7. **Governance & Documentation**
   - Log decisions in `Wintermute_dev_log.md` as DECs or CPs if needed.
   - Record infra steps, costs, and issues in this project doc.

### Risk & Mitigation Notes
- **Compute Limits:** Constrain local runs to ≤350M params, use sequence length ≤1K, gradient accumulation, and MLX optimizations; schedule cloud bursts for larger models or longer horizons.
- **Data Quality:** Run toxicity checks and enforce dedup using MinHash.
- **Training Instability:** Use gradient clipping, LR warmup, and monitor loss.
- **Cost Overruns:** Track GPU hours; prototype configs with tiny model before scaling.

### Working TODO List
- [ ] Confirm available GPU resources (local M3 Pro + optional cloud) and budget.
- [ ] Install and validate MLX environment (Python version, Metal support).
- [ ] Choose base dataset mix and document licensing constraints.
- [ ] Decide tokenizer approach (reuse vs train new) and build tokenization script.
- [ ] Draft model config (hidden size, layers, heads) and memory estimates.
- [ ] Scaffold MLX training script (data pipeline, optimizer loop, checkpointing).
- [ ] Define validation/eval datasets and baseline metrics.
- [ ] Plan lightweight instruction tuning phase (dataset + procedure).
- [ ] Outline deployment experiment (vLLM/TGI) and testing checklist.
- [ ] Capture governance updates (DEC/CP) once scope solidifies.
- [ ] Review lessons learned; feed cross-cutting insights into `memory.md`.

### Next Conversation Steps
1. Align on compute availability and acceptable model size.
2. Prioritize dataset acquisition tasks.
3. Walk through each TODO item with concrete actions and owners.


