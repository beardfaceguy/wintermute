# Handles audio upload + STT transcription for thVoice integration
# Replaces legacy chat.py with voice-only routing

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import cast

from fastapi import APIRouter, File, UploadFile
from pywhispercpp.model import Model

router = APIRouter()

# Resolve Whisper model path
CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[4]  # .../wintermute/
WHISPER_MODEL_PATH = REPO_ROOT / "thVoice" / "models" / "ggml-base.en.bin"

if not WHISPER_MODEL_PATH.exists():
    raise RuntimeError(f"❌ Whisper model not found at: {WHISPER_MODEL_PATH}")

# Load whisper model once
whisper_model = Model(str(WHISPER_MODEL_PATH))


def sha256sum(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@router.post("/chat/voice")
async def voice_input(file: UploadFile = File(...)):
    # Save uploaded audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp_path = tmp.name
        print("Checksum:", sha256sum(cast(Path, tmp_path)))

    wav_path = tmp_path.replace(".webm", ".wav")
    try:
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
        # audio: AudioSegment = AudioSegment.from_file(wav_path)
        # min_duration_ms = 1000
        # if len(audio) < min_duration_ms:
        #     print(f"Audio too short ({len(audio)} ms), padding to {min_duration_ms} ms")
        #     silence = AudioSegment.silent(duration=min_duration_ms - len(audio))
        #     audio += silence
        #     audio.export(
        #         wav_path, format="wav", parameters=["-ac", "1", "-ar", "16000"]
        #     )

        # Transcribe using Whisper.cpp
        segments = whisper_model.transcribe(wav_path)
        result_text = " ".join([seg.text for seg in segments]).strip()
        print("Segments:", segments)
        print("Transcript:", result_text)
        return {"transcript": result_text}
    except Exception as e:
        return {"error": f"STT failed: {str(e)}"}
    finally:
        os.remove(tmp_path)
