# Handles audio upload + STT transcription for thVoice integration
# Replaces legacy chat.py with voice-only routing

from fastapi import APIRouter, UploadFile, File
import wave
import os
import tempfile
from pathlib import Path
from pywhispercpp.model import Model

router = APIRouter()

# Resolve Whisper model path
CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[4]  # .../wintermute/
WHISPER_MODEL_PATH = REPO_ROOT / "thVoice" / "models" / "whisper.cpp" / "models" / "ggml-base.en.bin"

if not WHISPER_MODEL_PATH.exists():
    raise RuntimeError(f"❌ Whisper model not found at: {WHISPER_MODEL_PATH}")

# Load whisper model once
whisper_model = Model(str(WHISPER_MODEL_PATH))

@router.post("/chat/voice")
async def voice_input(file: UploadFile = File(...)):
    # Save uploaded audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp_path = tmp.name

    try:
        # Optional WAV check (keep if you want stricter control)
        wf = wave.open(tmp_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            raise ValueError("Invalid WAV format. Must be mono PCM.")
        wf.close()

        # Transcribe using Whisper.cpp
        segments = whisper_model.transcribe(tmp_path)
        result_text = " ".join([seg.text for seg in segments]).strip()

        return {"transcript": result_text}
    except Exception as e:
        return {"error": f"STT failed: {str(e)}"}
    finally:
        os.remove(tmp_path)
