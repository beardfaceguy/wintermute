# Handles audio upload + STT transcription for thVoice integration
# Replaces legacy chat.py with voice-only routing

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, UploadFile

logger = logging.getLogger(__name__)

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[4]
WHISPER_MODEL_PATH = REPO_ROOT / "thVoice" / "models" / "ggml-base.en.bin"

whisper_model: Any = None
_whisper_lock = threading.Lock()
_load_error: str | None = None


def _try_load_whisper_model() -> Any:
    """Load Whisper once; return None if unavailable (mirrors tts.py lazy Piper load)."""
    global whisper_model, _load_error
    if whisper_model is not None:
        return whisper_model
    with _whisper_lock:
        if whisper_model is not None:
            return whisper_model
        try:
            from pywhispercpp.model import Model as _WhisperModel
        except ImportError as e:
            _load_error = f"pywhispercpp not installed: {e}"
            logger.warning("%s — voice chat disabled", _load_error)
            return None

        if not WHISPER_MODEL_PATH.exists() or WHISPER_MODEL_PATH.stat().st_size < 1024 * 1024:
            _load_error = (
                f"Whisper model missing or too small at {WHISPER_MODEL_PATH} — "
                "run thVoice/scripts/fetch_whisper.sh"
            )
            logger.warning("%s", _load_error)
            return None

        try:
            whisper_model = _WhisperModel(str(WHISPER_MODEL_PATH))
            logger.info("Voice chat enabled (whisper model loaded)")
            return whisper_model
        except Exception as e:
            _load_error = f"Whisper load failed: {e}"
            logger.exception("Failed to load Whisper model")
            return None


def _voice_stt_ready() -> bool:
    return _try_load_whisper_model() is not None


def sha256sum(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


router = APIRouter()


@router.post("/chat/voice")
async def voice_input(file: UploadFile = File(...)):
    model = _try_load_whisper_model()
    if model is None:
        err = _load_error or "Whisper model not loaded"
        return {"error": f"Voice chat is not available — {err}"}

    tmp_path = None
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp_path = tmp.name
            contents = await file.read()
            tmp.write(contents)
            tmp.flush()
            logger.debug("Checksum: %s", sha256sum(Path(tmp_path)))

        wav_path = tmp_path.replace(".webm", ".wav")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                tmp_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                wav_path,
            ],
            check=True,
        )

        segments = model.transcribe(wav_path)
        result_text = " ".join([seg.text for seg in segments]).strip()
        logger.debug("Segments: %s", segments)
        logger.debug("Transcript: %s", result_text)
        return {"transcript": result_text}
    except Exception as e:
        return {"error": f"STT failed: {str(e)}"}
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        if wav_path:
            try:
                os.remove(wav_path)
            except OSError:
                pass


@router.get("/chat/voice/health")
def voice_stt_health() -> dict[str, Any]:
    """Probe STT readiness (Whisper binary + pywhispercpp) without loading twice."""
    ready = _voice_stt_ready()
    return {
        "enabled": ready,
        "model_path": str(WHISPER_MODEL_PATH),
        "error": None if ready else (_load_error or "model not loaded"),
    }
