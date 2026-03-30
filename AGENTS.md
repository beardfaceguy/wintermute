# AGENTS.md

## Cursor Cloud specific instructions

### Product Overview

Wintermute is a self-hosted AI assistant with a React frontend (Vite) + FastAPI backend, backed by vLLM for LLM inference, Whisper.cpp for speech-to-text, and PostgreSQL (pgvector) for persistence.

### Services

| Service | Port | How to Start |
|---|---|---|
| PostgreSQL (pgvector) | 5432 | `cd infra && docker compose up -d` |
| FastAPI Backend | 8000 | `cd talkingHead/backend && PYTHONPATH=/workspace:/workspace/talkingHead/backend:$PYTHONPATH DEBUG=true uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` |
| React Frontend (Vite) | 5173 | `cd talkingHead/frontend && npm run dev` |
| vLLM Server | 8001 | Requires NVIDIA GPU - not available in Cloud Agent VMs |

### Startup Order

1. Start PostgreSQL first (`cd infra && docker compose up -d`)
2. Run Alembic migrations: `cd talkingHead/backend && PYTHONPATH=/workspace:/workspace/talkingHead/backend:$PYTHONPATH alembic upgrade head`
3. Start backend (requires PostgreSQL ready and Whisper model at `thVoice/models/ggml-base.en.bin`)
4. Start frontend

### Key Gotchas

- The backend requires `PYTHONPATH` to include both `/workspace` and `/workspace/talkingHead/backend` so it can find the `shared` module and local `db`/`app` packages.
- The backend loads the Whisper model at import time from `thVoice/models/ggml-base.en.bin`. If this file is missing, the backend will crash on startup. Download it from: `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin`
- vLLM is configured to connect to `gaming-pc-linux:8001` in `config/shared_api_config.json`. Without a running vLLM server, chat messages will return "Name or service not known" errors - this is expected behavior when no GPU/LLM is available.
- Docker requires `sudo dockerd` to start the daemon, and `sudo chmod 666 /var/run/docker.sock` for non-root access. Docker is configured with `fuse-overlayfs` storage driver and `iptables-legacy` for nested container support.
- The `requirements.txt` at repo root includes heavy ML dependencies (torch, vllm, llama-index). For backend dev, only the core deps are needed: `fastapi uvicorn[standard] httpx python-dotenv pydantic sqlalchemy asyncpg psycopg2-binary alembic pgvector pywhispercpp pydub soundfile python-multipart`.

### Lint / Test / Build Commands

- **Backend tests**: `cd talkingHead/backend && PYTHONPATH=/workspace:/workspace/talkingHead/backend:$PYTHONPATH pytest tests/ -v --tb=short`
- **Frontend lint**: `cd talkingHead/frontend && npx eslint .`
- **Frontend type check**: `cd talkingHead/frontend && npx tsc -b`
- **Frontend build**: `cd talkingHead/frontend && npm run build`

### Pre-existing Test Failures

12 of 30 backend tests have pre-existing failures due to async mock issues in test code (not environment-related). 17 tests pass, 1 is skipped. The failures are in `test_llm.py` (async iterator mocking), `test_websocket.py` (missing `@pytest.mark.asyncio` and `AsyncMock`), and `test_db_ops.py` (incorrect assertion order).
