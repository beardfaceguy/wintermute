# Wintermute Design Decisions
_Immutable – newest appended at bottom_

| DEC‑ID | Date | Decision | Rationale | Linked CP |
|--------|------|----------|-----------|-----------|
| DEC‑02 | 2025‑04‑21 | Keep CUDA stack; evaluate WebGPU only for client-side visualizations | WebGPU lacks memory headroom and kernel maturity for back-end LLM workloads; CUDA is required for vLLM, FlashAttention, and bitsandbytes. WebGPU may be used later for browser-native visualizations (e.g., embedding plots, real-time model metrics) | —
