"""Text-to-speech endpoint backed by Piper TTS.

Design A (whole-response REST): the frontend posts the assistant's complete
text after the WebSocket stream ends, and this endpoint returns a single
WAV blob. Lower-latency sentence streaming (Design C) is tracked separately.

The Piper voice model is loaded lazily on first request to avoid penalising
test imports and keep cold-start of the FastAPI app fast. Reload happens
implicitly when the path / scale env vars change between requests are *not*
supported on purpose — restart the process to swap voices.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import wave
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[4]
DEFAULT_VOICE_PATH = REPO_ROOT / "thVoice" / "piper" / "models" / "en_GB-cori-medium.onnx"

_VOICE_PATH = Path(os.getenv("PIPER_VOICE_PATH", str(DEFAULT_VOICE_PATH)))
_LENGTH_SCALE = float(os.getenv("PIPER_LENGTH_SCALE", "1.0"))
_NOISE_SCALE = float(os.getenv("PIPER_NOISE_SCALE", "0.667"))
_NOISE_W = float(os.getenv("PIPER_NOISE_W", "0.8"))
_ENABLED_FLAG = os.getenv("PIPER_ENABLED", "true").lower() != "false"
_MAX_INPUT_CHARS = int(os.getenv("PIPER_MAX_INPUT_CHARS", "5000"))

_voice: Any = None
_voice_lock = threading.Lock()
_load_error: str | None = None


def _try_load_voice() -> Any:
    """Load the Piper voice on first call. Returns None on failure (caller decides)."""
    global _voice, _load_error
    if _voice is not None:
        return _voice
    with _voice_lock:
        if _voice is not None:
            return _voice
        if not _ENABLED_FLAG:
            _load_error = "PIPER_ENABLED=false"
            return None
        if not _VOICE_PATH.exists() or _VOICE_PATH.stat().st_size < 1024:
            _load_error = (
                f"voice model missing or empty at {_VOICE_PATH} — "
                "run thVoice/scripts/fetch_voice.sh"
            )
            logger.warning(_load_error)
            return None
        try:
            from piper.voice import PiperVoice
        except ImportError as e:
            _load_error = f"piper-tts not installed: {e}"
            logger.warning(_load_error)
            return None
        try:
            _voice = PiperVoice.load(str(_VOICE_PATH))
            logger.info("Piper voice loaded from %s", _VOICE_PATH)
            return _voice
        except Exception as e:
            _load_error = f"PiperVoice.load failed: {e}"
            logger.exception("Failed to load Piper voice")
            return None


def _is_enabled() -> bool:
    return _try_load_voice() is not None


router = APIRouter()


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesise.")


@router.get("/chat/speak/health")
def tts_health() -> dict[str, Any]:
    """Lightweight probe so the frontend can hide the speaker toggle gracefully."""
    enabled = _is_enabled()
    return {
        "enabled": enabled,
        "voice_path": str(_VOICE_PATH),
        "length_scale": _LENGTH_SCALE,
        "noise_scale": _NOISE_SCALE,
        "noise_w": _NOISE_W,
        "error": None if enabled else _load_error,
    }


@router.post("/chat/speak")
def synthesize(req: SpeakRequest) -> Response:
    """Synthesize `text` to a single WAV blob and return it inline."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")
    if len(text) > _MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"text exceeds {_MAX_INPUT_CHARS} chars",
        )
    voice = _try_load_voice()
    if voice is None:
        raise HTTPException(
            status_code=503,
            detail=f"TTS unavailable: {_load_error or 'unknown'}",
        )

    buf = io.BytesIO()
    try:
        with wave.open(buf, "wb") as wav_file:
            voice.synthesize(
                text,
                wav_file,
                length_scale=_LENGTH_SCALE,
                noise_scale=_NOISE_SCALE,
                noise_w=_NOISE_W,
            )
    except Exception as e:
        logger.exception("Piper synthesize failed")
        raise HTTPException(status_code=500, detail=f"synthesis failed: {e}") from e

    audio = buf.getvalue()
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-store",
            "Content-Length": str(len(audio)),
        },
    )
