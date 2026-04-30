# Wintermute MCP Core Infrastructure — Implementation Record

**Status: Phase 1 COMPLETE (2026-04-19)**
**Linear Program Issue: CLA-136**

## What Was Built

### 1. wintermute-memory MCP Server

**File:** `mcp_memory/server.py`
**Framework:** FastMCP v3.2.4 (real MCP protocol, stdio + HTTP transport)
**Database:** PostgreSQL + pgvector (via `infra/docker-compose.yml`)
**Embedding model:** BAAI/bge-small-en-v1.5 (384-dim, loaded lazily via sentence-transformers)

#### Tools

| Tool | Description |
|------|-------------|
| `memory_add` | Store entries with auto-generated embeddings, tags, zone (live/cold) |
| `memory_search` | Semantic cosine similarity search via pgvector, with zone/trust filters |
| `memory_recall_recent` | Chronological recall with zone and JSONB tag filtering |
| `memory_promote` | Move entry from live to cold zone (requires trust_score >= 0.7) |
| `memory_flag` | Flag entry for audit review (Freud hook point) |
| `memory_update_trust` | Update trust score (0.0–1.0) |

#### Resources

| URI | Description |
|-----|-------------|
| `memory://stats` | Counts by zone, flagged entries, avg trust score |

#### Schema (existing, unchanged)

`mcp_memory/app/models/memory_entry.py`: `MemoryEntry` table with UUID pk, text, embedding (Vector(384)), tags (JSONB), zone (live/cold), trust_score (float), audit_flagged (bool), created_at.

#### Key Design Decisions

- Table creation is deferred (`_ensure_tables()`) rather than at import time, so the server can be imported without a running DB.
- Imports use try/except to work both as `python mcp_memory/server.py` (from the mcp_memory dir) and `from mcp_memory.server import mcp` (from project root).
- The old FastAPI REST API in `mcp_memory/app/main.py` and `mcp_memory/app/api/memory.py` is preserved but superseded by the MCP server.

### 2. wintermute-postgres MCP Server

**File:** `mcp_servers/mcp_postgres/server.py`
**Framework:** FastMCP v3.2.4
**Database:** Direct psycopg2 connections (read-only enforced)

#### Tools

| Tool | Description |
|------|-------------|
| `sql_list_tables` | Enumerate all user-created tables |
| `sql_describe_table` | Full schema: columns, types, PKs, FKs, row count |
| `sql_query` | Execute read-only SQL (mutations rejected by keyword blocklist + `SET TRANSACTION READ ONLY`) |
| `sql_explain` | EXPLAIN ANALYZE on queries |
| `sql_sample_rows` | Quick data inspection (up to 20 rows) |

#### Resources

| URI | Description |
|-----|-------------|
| `postgres://schema-summary` | All tables with column counts |

#### Safety

- `_is_read_only()` rejects INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, CREATE, GRANT, REVOKE, COPY.
- Connection uses `conn.set_session(readonly=True, autocommit=True)`.
- Results capped at max 1000 rows.

### 3. Agent Runner

**File:** `agents/runner.py`
**Purpose:** Lightweight tool-calling loop connecting MCP servers to any OpenAI-compatible LLM.

- Connects to multiple MCP servers via FastMCP Client
- Discovers tools and converts MCP schemas to OpenAI function-calling format
- Dispatches tool_calls from LLM responses back to appropriate MCP server
- Loops until LLM produces final text response or max iterations reached
- Supports vLLM, OpenAI, or any compatible endpoint

### 4. Test-Driven SQL Agent

**File:** `agents/sql_agent.py`
**Test cases:** `agents/test_cases/memory_queries.yaml`

#### Flow

1. Load YAML test cases (natural-language question + expected validation)
2. Explore schema via mcp-postgres tools
3. Generate SQL (manual mode via hint map, or LLM mode via vLLM)
4. Execute via `sql_query` and validate results
5. Store successful/failed strategies in mcp-memory with semantic embeddings

#### Test Results (2026-04-19)

6/6 tests passing in manual mode:
- count_all, count_by_zone, find_flagged, high_trust, tag_filter, recent_entries

LLM mode available via `--llm http://localhost:8001/v1 --model MODEL_NAME`.

## Cursor MCP Configuration

Both servers registered in `.cursor/mcp.json`:

```json
{
  "wintermute-memory": {
    "command": "python3",
    "args": ["mcp_memory/server.py"],
    "cwd": "/home/beardface/work/wintermute",
    "env": {
      "DATABASE_URL": "postgresql://wintermute:wintermute@localhost:5432/wintermute"
    }
  },
  "wintermute-postgres": {
    "command": "python3",
    "args": ["mcp_servers/mcp_postgres/server.py"],
    "cwd": "/home/beardface/work/wintermute",
    "env": {
      "DATABASE_URL": "postgresql://wintermute:wintermute@localhost:5432/wintermute"
    }
  }
}
```

## Prerequisites

- PostgreSQL with pgvector: `cd infra && docker compose up -d`
- Python packages: `pip3 install --break-system-packages --user fastmcp psycopg2-binary sentence-transformers sqlalchemy pgvector pyyaml`

## Config Changes Made

- `config/shared_api_config.json`: Changed `mcp_memory.port` from 8001 to 8002 (was colliding with vLLM)
- `config/shared_api_config.json`: Replaced dead `gaming-pc-linux` vLLM host with `${VLLM_HOST}` env var + AWS infrastructure config (account, instance type, ECR image, S3 model path)
- `shared/config_loader.py`: Added `${VLLM_HOST}` env var substitution in `load_vllm_config()`, added `load_vllm_aws_config()` for launch scripts
- `requirements.txt`: Added `fastmcp`

## Files Created/Modified

| File | Action |
|------|--------|
| `mcp_memory/server.py` | **Created** — FastMCP MCP server |
| `mcp_memory/__init__.py` | **Created** — package init |
| `mcp_memory/app/__init__.py` | **Created** — package init |
| `mcp_memory/app/db/session.py` | **Modified** — echo=False |
| `mcp_servers/mcp_postgres/__init__.py` | **Created** |
| `mcp_servers/mcp_postgres/server.py` | **Created** — FastMCP MCP server |
| `agents/__init__.py` | **Created** |
| `agents/runner.py` | **Created** — AgentRunner |
| `agents/sql_agent.py` | **Created** — test-driven SQL agent |
| `agents/test_cases/memory_queries.yaml` | **Created** — 6 test cases |
| `.cursor/mcp.json` | **Modified** — added wintermute-memory and wintermute-postgres |
| `config/shared_api_config.json` | **Modified** — port fix |
| `requirements.txt` | **Modified** — added fastmcp |

## Linear Issues

| ID | Title | Status |
|----|-------|--------|
| CLA-136 | [Program] Wintermute Core MCP Infrastructure | In Progress |
| CLA-137 | Build mcp-memory MCP server | Done |
| CLA-138 | Build mcp-postgres MCP server | Done |
| CLA-139 | Build test-driven SQL agent PoC | Done |
| CLA-140 | Wire mcp-memory into talkingHead chat flow | Done |
| CLA-141 | Connect SQL agent to live vLLM | Done |
| CLA-142 | Implement Freud sanity auditor | Planned |

## Phase 2: talkingHead Memory Integration (CLA-140)

**Status: COMPLETE (2026-04-19)**

### What Was Built

Integrated mcp-memory into the talkingHead WebSocket chat pipeline so that:
- Every user message triggers a semantic search over strategic memory
- Relevant past exchanges are injected into the LLM prompt as context
- Every conversation exchange is automatically stored in mcp-memory (live zone)
- Memory entries are tagged with session_id, source, and type for Freud auditing

### Architecture Decision

**Direct function import** — The `@mcp.tool` functions in `mcp_memory/server.py` are called directly via `asyncio.to_thread()` (they're synchronous SQLAlchemy). This avoids MCP protocol overhead while reusing the exact same code that Cursor's MCP integration uses.

### Files Created/Modified

| File | Action |
|------|--------|
| `talkingHead/backend/memory/strategic.py` | **Created** — Async bridge: `search_relevant_memories()`, `store_conversation()`, `format_memory_context()` |
| `talkingHead/backend/memory/__init__.py` | **Created** — Package init |
| `talkingHead/backend/app/websocket/chat_ws.py` | **Modified** — Added pre-response memory search and post-response memory storage |

### Message Flow (updated)

```
User Message
    │
    ├─► store_message() → Postgres messages table (short-term history)
    │
    ├─► search_relevant_memories(user_message) → mcp-memory pgvector search
    │       └─► format_memory_context() → "[Relevant Memory] ..." block
    │
    ├─► get_recent_messages() → conversation history
    │
    ├─► Build prompt: [memory context] + [history] + user turn
    │
    ├─► ChatProcessor.stream_response() → vLLM → streamed tokens → WebSocket
    │
    ├─► store_message() → assistant response in Postgres
    │
    └─► store_conversation() → mcp-memory (live zone, tagged)
```

### Failure Handling

All memory operations are wrapped in try/except. If mcp-memory is unavailable (DB down, import fails, etc.), the chat flow continues normally — memory is best-effort, never a blocker.

## Phase 3: Live vLLM SQL Agent (CLA-141)

**Status: COMPLETE (2026-04-19)**

### What Was Built

Made the SQL agent production-ready for live vLLM operation:

1. **Shared config integration** — `--llm` with no URL reads `gaming-pc-linux:8001` and `wizard-vicuna-7b-awq` from `config/shared_api_config.json` instead of hardcoding localhost.

2. **Pre-flight check** — Queries `/v1/models` before running tests. Validates connectivity, discovers available models, and auto-selects the first model if none specified. Reports clear errors for timeout, connection refused, or missing model.

3. **Robust SQL extraction** — `_clean_sql()` now handles markdown fences (```sql), inline backticks, explanatory text around queries, trailing semicolons, and multiple statements (extracts first SELECT). All patterns tested.

4. **Per-attempt error resilience** — LLM connection errors, timeouts, and unexpected responses are caught per-attempt (not per-test), so the retry loop continues through transient failures.

5. **Timing and structured results** — Per-test and total elapsed time tracked. JSON summary written to `agents/test_cases/last_run.json` with pass rate, model name, timing, and per-test details.

6. **`--max-retries` flag** — Configurable retry count (default 3).

### Usage

```bash
python agents/sql_agent.py                                     # manual mode (6/6 pass)
export VLLM_HOST=<ec2_public_ip>
python agents/sql_agent.py --llm                               # LLM mode (uses AWS vLLM)
python agents/sql_agent.py --llm http://custom:8010/v1         # explicit endpoint
python agents/sql_agent.py --llm --max-retries 5               # more attempts
```

### vLLM Launch (AWS)

```bash
./scripts/aws_commands/vllm_serve.sh launch     # spin up EC2 + start vLLM
./scripts/aws_commands/vllm_serve.sh status      # check IP and health
./scripts/aws_commands/vllm_serve.sh stop        # stop (preserves EBS)
./scripts/aws_commands/vllm_serve.sh terminate   # terminate
```

### Files Modified

| File | Action |
|------|--------|
| `agents/sql_agent.py` | **Modified** — Shared config, preflight, SQL cleaning, timing, error handling |
| `config/shared_api_config.json` | **Modified** — vLLM host → `${VLLM_HOST}` env var, added `aws` config block |
| `shared/config_loader.py` | **Modified** — Env var substitution, `load_vllm_aws_config()` |
| `scripts/aws_commands/vllm_serve.sh` | **Created** — Launch/status/stop/terminate for vLLM on EC2 |

## Phase 4: Freud Sanity Auditor (CLA-142)

**Status: COMPLETE (2026-04-19)**

### What Was Built

Freud is a batch auditor that scans the mcp-memory `live` zone for problematic entries and takes corrective action. It operates as a standalone CLI tool that can be run manually or on a schedule.

#### Audit Checks

| Check | Severity | Description |
|-------|----------|-------------|
| `low_quality` | warning | Entries with text shorter than 20 chars or suspiciously few word boundaries |
| `near_duplicate` | warning | Entry pairs with cosine similarity >= 0.92 |
| `contradiction` | critical | Semantically similar entries (0.65–0.90) where one contains negation markers and the other doesn't |
| `stale` | info | Live entries older than 14 days that were never promoted |

#### Actions

| Action | Description |
|--------|-------------|
| `flag` | Sets `audit_flagged=True` on entries with warning/critical findings, records reason in tags |
| `trust_update` | Penalizes flagged entries (-0.2 trust), boosts clean entries (+0.15 trust) |
| `promote` | Auto-promotes clean entries with trust >= 0.8 to cold zone (opt-in via `--promote-ready`) |

#### Usage

```bash
export DATABASE_URL="postgresql://wintermute:wintermute@localhost:5432/wintermute"
python agents/freud.py                       # full audit of live zone
python agents/freud.py --dry-run             # report only, no modifications
python agents/freud.py --zone all            # audit both live and cold zones
python agents/freud.py --flagged-only        # re-audit already-flagged entries
python agents/freud.py --promote-ready       # auto-promote entries passing all checks
```

#### Architecture

Freud imports mcp-memory functions directly (same pattern as `talkingHead/backend/memory/strategic.py`), using `memory_flag`, `memory_update_trust`, and `memory_promote` for actions. Embedding comparisons use the same BAAI/bge-small-en-v1.5 vectors stored in pgvector.

#### Test Results (2026-04-19)

Tested against 13 live entries (8 real conversation entries + 5 seeded test entries):
- Correctly identified 1 low-quality entry ("ok", 2 chars)
- Correctly detected 2 near-duplicate pairs (sim 0.99 and 0.94)
- Correctly detected 6 contradiction pairs (negation-based)
- Flagged 8 entries, calibrated trust on 5

### Files Created

| File | Action |
|------|--------|
| `agents/freud.py` | **Created** — Freud sanity auditor with 4 checks + 3 actions + CLI |

## Linear Issues

| ID | Title | Status |
|----|-------|--------|
| CLA-136 | [Program] Wintermute Core MCP Infrastructure | In Progress |
| CLA-137 | Build mcp-memory MCP server | Done |
| CLA-138 | Build mcp-postgres MCP server | Done |
| CLA-139 | Build test-driven SQL agent PoC | Done |
| CLA-140 | Wire mcp-memory into talkingHead chat flow | Done |
| CLA-141 | Connect SQL agent to live vLLM | Done |
| CLA-142 | Implement Freud sanity auditor | Done |

## Phase 5: Titan GPT-Medium (407M) Training

**Status: PRETRAINING + SFT COMPLETE (2026-04-30)**

GPT-Medium pretraining (125k steps, 8.19B tokens) and general SFT (5k steps) are both finished.
See `.cursor/docs/TRAINING_RUN_GPT_MEDIUM_20260419.md` for the full run log and `.cursor/docs/llm_training_project.md` for the project summary.

**Key results:**
- Pretraining: val loss 2.9641, val ppl **19.38** (passed <50 SFT gate)
- SFT: eval loss 1.92, eval ppl **6.79**
- Pipeline unified: `train.py` now handles single-GPU, multi-GPU DDP, and CPU/MPS
- SFT pipeline supports 4 auto-detected formats (HF messages, ShareGPT, Alpaca, chat text)
- Weights-only checkpoint: `saved_models/gpt_medium_407m_sft_5000_weights.pt` (1.63 GB)

**Next**: Domain-specific SFT forks, 7B model scale-up.

## What Comes Next

1. **Domain-specific SFT forks**: Specialize the 407M general SFT checkpoint for different use cases (cybersecurity, coding, etc.).
2. **7B model scale-up**: Train a larger model using the proven pipeline on multi-GPU instances.
3. **Jung / Adler auditors**: Secondary analysis passes for pattern drift and motivational consistency.
4. **Scheduled Freud runs**: Integrate Freud into a cron or systemd timer for daily audits.
5. **Live → Cold promotion policies**: Formalize rules for automatic memory promotion based on audit results.
