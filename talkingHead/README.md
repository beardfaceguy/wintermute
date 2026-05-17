# talkingHead

Browser chat UI (React + Vite) and FastAPI backend with WebSocket streaming, optional **Piper TTS** and **Whisper STT** via **thVoice** (models live under `../thVoice/`, not in git).

## Voice / thVoice setup

Binary weights are **gitignored**—fetch once per clone (or after deleting `thVoice/` model files):

```bash
# From repo root
bash thVoice/scripts/fetch_voice.sh    # Piper ONNX (~63MB)
bash thVoice/scripts/fetch_whisper.sh # Whisper ggml-base.en (~148MB)
# or both:
bash thVoice/scripts/fetch_all.sh
```

**Python:** Install repo root dependencies (includes `piper-tts`, `pywhispercpp`):

```bash
cd /path/to/wintermute
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**STT:** `POST /api/chat/voice` runs `ffmpeg` to decode WebM → 16 kHz WAV before Whisper. Install **`ffmpeg`** on the host (`apt install ffmpeg`, etc.).

**Sanity check:** With the venv active and models present:

```bash
curl -s http://127.0.0.1:8010/api/chat/speak/health | jq .
# "enabled": true when Piper (TTS) loads

curl -s http://127.0.0.1:8010/api/chat/voice/health | jq .
# "enabled": true when Whisper (STT) loads
```

(Default API port comes from `config/shared_api_config.json` → `web_interface.port`; adjust host/port if yours differs.)

## Run locally

**Backend** (from `talkingHead/backend`, with venv activated and `PYTHONPATH` including repo root—the `shared` package lives at repo root):

```bash
cd talkingHead/backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8010
```

**Frontend:**

```bash
cd talkingHead/frontend
npm install
npm run dev
```

## Tests

```bash
cd talkingHead/backend && pytest tests/ -q
cd talkingHead/frontend && npm test
```

E2E (Playwright) may use a mocked backend; see `frontend/e2e/README.md`.
