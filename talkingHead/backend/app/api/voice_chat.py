# Handles audio upload + STT transcription for thVoice integration
# Replaces legacy chat.py with voice-only routing

import hashlib
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

logger = logging.getLogger(__name__)

_VOICE_ENABLED = False
whisper_model = None

try:
    from pywhispercpp.model import Model as _WhisperModel

    CURRENT_FILE = Path(__file__).resolve()
    REPO_ROOT = CURRENT_FILE.parents[4]
    WHISPER_MODEL_PATH = REPO_ROOT / "thVoice" / "models" / "ggml-base.en.bin"

    if WHISPER_MODEL_PATH.exists() and WHISPER_MODEL_PATH.stat().st_size > 0:
        whisper_model = _WhisperModel(str(WHISPER_MODEL_PATH))
        _VOICE_ENABLED = True
        logger.info("Voice chat enabled (whisper model loaded)")
    else:
        logger.warning("Whisper model not found at %s — voice chat disabled", WHISPER_MODEL_PATH)
except ImportError:
    logger.warning("pywhispercpp not installed — voice chat disabled")


def sha256sum(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


router = APIRouter()


@router.post("/chat/voice")
async def voice_input(file: UploadFile = File(...)):
    if not _VOICE_ENABLED or whisper_model is None:
        return {"error": "Voice chat is not available — whisper model not loaded"}

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
                "ffmpeg", "-y", "-i", tmp_path,
                "-ac", "1", "-ar", "16000", "-f", "wav", wav_path,
            ],
            check=True,
        )

        segments = whisper_model.transcribe(wav_path)
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
