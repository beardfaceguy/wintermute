# Wintermute — Cursor Agent Onboarding

**All Cursor Agents working on this project MUST read this file first.**

## What Is Wintermute?

A self-improving AI system combining a locally-hosted LLM (vLLM), strategic memory (pgvector), MCP tool ecosystem, and a browser-based chat interface (talkingHead). The long-term goal is an autonomous agent that learns from its own interactions.

## Project Map

```
wintermute/
├── talkingHead/              # Browser chat interface (React + FastAPI + WebSocket)
│   ├── frontend/             # React UI
│   └── backend/              # FastAPI + vLLM streaming chat
│       ├── app/
│       │   ├── main.py       # CORS, routes, WebSocket mount
│       │   ├── chat/llm.py   # ChatProcessor (vLLM completions)
│       │   └── websocket/    # WebSocket chat handlers
│       └── memory/
│           └── strategic.py  # Async bridge to mcp-memory (CLA-140)
├── mcp_memory/               # MCP Memory Server (FastMCP + pgvector)
│   ├── server.py             # 6 tools + 1 resource (memory_add, memory_search, etc.)
│   └── app/
│       ├── models/memory_entry.py  # SQLAlchemy MemoryEntry model
│       └── db/session.py     # PostgreSQL engine + session factory
├── mcp_servers/
│   └── mcp_postgres/
│       └── server.py         # Read-only SQL MCP server (5 tools + 1 resource)
├── agents/
│   ├── runner.py             # AgentRunner (LLM ↔ MCP tool-calling loop)
│   ├── sql_agent.py          # Test-driven SQL generation agent
│   ├── freud.py              # Sanity auditor: contradiction, duplicate, quality checks (CLA-142)
│   └── test_cases/
│       └── memory_queries.yaml  # 6 SQL test cases
├── infra/
│   └── docker-compose.yml    # PostgreSQL + pgvector container
├── config/
│   └── shared_api_config.json # Ports, endpoints, service config
├── model_training/titanProject/  # LLM training pipeline
│   ├── train.py             # Unified pretraining loop: single-GPU, multi-GPU DDP, CPU/MPS
│   ├── finetune_sft.py      # SFT fine-tuning loop
│   ├── model.py             # Model factory (GPTLM, TitansLM, HFGPT2LM)
│   ├── data.py              # Disk-backed dataset + TokenCache
│   ├── prepare_sft_mix.py   # Multi-source SFT data preparation
│   ├── export_to_hf.py      # Export checkpoint to HF format
│   ├── tests/               # pytest suite (210 tests)
│   │   ├── conftest.py      # Shared fixtures (tiny configs, dummy tokenizer)
│   │   ├── test_data.py     # Data pipeline + TokenCache tests
│   │   ├── test_model.py    # Model construction + forward pass tests
│   │   ├── test_train_utils.py  # LR schedule, checkpointing, S3 tokenizer, utilities
│   │   ├── test_multi_gpu.py    # DDP tests (unit + Gloo multi-process)
│   │   └── test_sft_formats.py  # SFT data format parsing tests
│   └── configs/
│       ├── config_gpt_medium.yaml              # 407M pretrain config
│       ├── config_gpt_medium_sanity_overfit.yaml # 407M sanity gate
│       ├── config_sft_gpt_medium_instruction.yaml # 407M SFT config
│       ├── config_sft_hf_qlora.yaml            # QLoRA template for HF models (7B+)
│       └── config_gpt_small.yaml               # 117M pretrain (completed)
├── .cursor/
│   ├── mcp.json              # Cursor MCP server registrations
│   ├── docs/
│   │   ├── Wintermute_Master_Spec.md                     # Architecture vision & roadmap
│   │   ├── SFT_PIPELINE_GUIDE.md                         # SFT data format, config, and pipeline reference
│   │   ├── IMPLEMENTATION_MCP_CORE.md                    # MCP infra implementation record
│   │   ├── IMPLEMENTATION_AWS_titan_llm_model_training.md     # AWS training infra
│   │   ├── TRAINING_RUN_GPT_MEDIUM_20260419.md           # GPT-Medium 407M run log
│   │   ├── llm_training_project.md                       # LLM training project summary
│   │   ├── AGENT_LINEAR_HANDOFF.md                       # Linear-first agent working order
│   │   ├── IMPLEMENTATION_DOC_LINEAR.md                  # Linear adoption technical companion
│   │   ├── ssm_timeout_fixes.md                          # SSM timeout root cause + fixes
│   │   ├── ALIX_coding_practices.md                      # ALIX team coding standards (PR reviews)
│   │   └── archive/                                      # Superseded historical docs
│   └── rules/                # Agent behavior rules
└── requirements.txt          # Python dependencies
```

## Current Status (2026-05-01)

### COMPLETED
- **mcp-memory** (CLA-137): FastMCP server with semantic search, trust scoring, zone management
- **mcp-postgres** (CLA-138): Read-only SQL access with safety guards
- **Agent Runner** (agents/runner.py): OpenAI function-calling bridge to MCP servers
- **SQL Agent PoC** (CLA-139): 6/6 test cases passing, stores strategies in memory
- **talkingHead Memory Integration** (CLA-140): mcp-memory wired into WebSocket chat — pre-response semantic search + post-response conversation storage
- **Live vLLM SQL Agent** (CLA-141): Shared config, preflight check, robust SQL extraction, timing, structured results
- **Freud Sanity Auditor** (CLA-142): Batch auditor with 4 checks (low-quality, near-duplicate, contradiction, stale) + trust calibration + auto-promote
- **GPT-small Training**: 117M param LLM trained on FineWeb-Edu (see docs/llm_training_project.md)

### RECENTLY COMPLETED
- **HF 7B QLoRA SFT — End-to-End Validated** (CLA-259, 2026-05-01): Mistral-7B-v0.3 fine-tuned using QLoRA on single L4 GPU (g6.2xlarge). Full pipeline proven: dataset prep → QLoRA fine-tune → LoRA merge → custom FastAPI inference server (`simple_serve.py`) → talkingHead chat UI. Model produces coherent conversational responses. Pipeline ready for specialized domain fine-tuning. LoRA adapter in S3. See `.cursor/docs/SFT_PIPELINE_GUIDE.md`.
- **talkingHead Cloud Deployment** (2026-05-01): Backend hardened for headless AWS deployment — `pywhispercpp`/Whisper model optional (graceful degradation), `pgvector` optional (SQLite fallback), CORS opened, WebSocket URL auto-detects remote hostname. Deploy via `scripts/aws_commands/deploy_talkinghead.sh`.
- **GPT-Medium General SFT** (2026-04-30): 407M model fine-tuned for 5,000 steps on OASST1+OpenHermes+SlimOrca+GSM8K mix. Final eval ppl 6.79. Model passes conversation tests (factual Q&A, coding, explanations). Checkpoint in S3 at `gpt_medium_sft_20260430052503/`. Weights-only local copy in `saved_models/`. SFT pipeline documented in `.cursor/docs/SFT_PIPELINE_GUIDE.md`.
- **Training Pipeline Unification** (2026-04-29): Merged `train.py` and `train_multi_gpu.py` into a single unified `train.py` supporting single-GPU, multi-GPU DDP, and CPU/MPS. Extracted shared utilities into `train_utils.py`. Updated `finetune_sft.py` with DDP support using the same shared utilities.
- **Training Pipeline Hardening** (2026-04-28): Comprehensive robustness overhaul of all launch scripts. Dynamic data root detection (LVM/EBS/root), `mkdir -p CODE_DIR` for fresh instances, portable token cache keys, self-stop with auto-tagging, quieter pip output. New `gpt_medium_pretrain_multigpu_cloudwatch.sh` script. Full documentation in `scripts/aws_commands/README.md`.
- **Titan GPT-Medium Pretraining** (2026-04-28, CLA-143): 407M param model trained to step 125,000 on g6.2xlarge. Final val ppl **19.38**, 8.19B tokens, ~214 hours. Last checkpoint at step 124,000 in S3. SFT gate PASSED. See `.cursor/docs/TRAINING_RUN_GPT_MEDIUM_20260419.md`.
- **Training Pipeline Fixes** (2026-04-28): Fixed missing final checkpoint save in `train.py`. Fixed silent self-stop failure (IAM tag mismatch) in runner scripts.
- **SFT Format Support** (2026-04-30): `finetune_sft.py` auto-detects 4 formats per line (HF messages JSONL, ShareGPT JSONL, Alpaca JSONL, chat text). Compatible with most Hugging Face Hub datasets without conversion. 29 new tests added.
- **Multi-GPU DDP Training** (2026-04-22): DDP support validated on g5.12xlarge (4x A10G). 32k tok/s throughput, perfect weight sync across ranks. Ready for 1B+ scale-up.
- **Automated Test Suite**: pytest framework with 477 tests (267 root/talkingHead + 210 titanProject) covering data pipeline, model construction, training utilities, checkpointing, DDP, SFT format parsing, MCP servers, agents, resource lifecycle, and memory operations.

### NEXT UP (in priority order)
1. **Domain-Specific SFT Forks**: Fine-tune specialized models using HF 7B base + QLoRA (pentest, code review, tool fluency). See CLA-245, CLA-248, CLA-177.
2. **Jung/Adler auditors**: Secondary analysis passes for pattern drift and motivational consistency

### Linear Project
- **Project**: "Wintermute Platform" on Linear (Clawcorp workspace)
- **Program Tracker**: CLA-136 — [Program] Wintermute Core MCP Infrastructure

## Prerequisites

```bash
# Database
cd infra && docker compose up -d

# Python packages
pip3 install --break-system-packages --user fastmcp psycopg2-binary sentence-transformers sqlalchemy pgvector pyyaml

# Database URL
export DATABASE_URL="postgresql://wintermute:wintermute@localhost:5432/wintermute"

# vLLM (launch AWS instance, then set the IP)
./scripts/aws_commands/vllm_serve.sh launch
export VLLM_HOST=<public_ip_from_output>
```

## Key Technical Details

- **MCP Framework**: FastMCP v3.2.4 (real Model Context Protocol, not HTTP shims)
- **Embedding Model**: BAAI/bge-small-en-v1.5 (384-dim, loaded lazily)
- **Database**: PostgreSQL 16 + pgvector extension
- **LLM Serving**: vLLM on AWS EC2 (g5.xlarge), port 8010 — set `VLLM_HOST` env var to instance IP
- **mcp-memory port**: 8002 (changed from 8001 to avoid collision)
- **Transport**: stdio (for Cursor integration) or HTTP (for standalone use)

## Critical Rules

1. Read `.cursor/docs/Wintermute_Master_Spec.md` for the full architecture vision
2. Read `.cursor/docs/IMPLEMENTATION_MCP_CORE.md` for what was already built and how
3. Check Linear issues (CLA-136 through CLA-142) for current task status
4. The old REST API files in `mcp_memory/app/main.py` and `mcp_memory/app/api/` are **superseded** — use `mcp_memory/server.py` instead
5. The MCP servers are registered in `.cursor/mcp.json` — both `wintermute-memory` and `wintermute-postgres` should appear in Cursor's MCP panel
6. Table creation is deferred (lazy `_ensure_tables()`) — the DB must be running before invoking tools, not before importing
