"""
Pytest configuration and fixtures for thVoice tests.
"""

# Add the thVoice directory to the path
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import piper modules
try:
    from piper.src.python_run.piper.config import PhonemeType, PiperConfig
    from piper.src.python_run.piper.voice import PiperVoice

    piper_available = True
except ImportError:
    piper_available = False


@pytest.fixture
def sample_audio_data() -> bytes:
    """Create sample audio data for testing."""
    # Create a minimal WAV file header + some audio data
    sample_rate = 16000
    duration_ms = 100
    num_samples = int(sample_rate * duration_ms / 1000)

    # WAV header (44 bytes)
    header = (
        b"RIFF"  # Chunk ID
        + (36 + num_samples * 2).to_bytes(4, "little")  # Chunk size
        + b"WAVE"  # Format
        + b"fmt "  # Subchunk1 ID
        + (16).to_bytes(4, "little")  # Subchunk1 size
        + (1).to_bytes(2, "little")  # Audio format (PCM)
        + (1).to_bytes(2, "little")  # Num channels (mono)
        + sample_rate.to_bytes(4, "little")  # Sample rate
        + (sample_rate * 2).to_bytes(4, "little")  # Byte rate
        + (2).to_bytes(2, "little")  # Block align
        + (16).to_bytes(2, "little")  # Bits per sample
        + b"data"  # Subchunk2 ID
        + (num_samples * 2).to_bytes(4, "little")  # Subchunk2 size
    )

    # Generate some simple audio data (sine wave)
    import numpy as np

    t = np.linspace(0, duration_ms / 1000, num_samples)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    return header + audio_data.tobytes()


@pytest.fixture
def mock_whisper_model() -> Generator[MagicMock, None, None]:
    """Mock Whisper model for testing."""
    with patch("pywhispercpp.model.Model") as mock_model_class:
        mock_instance = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_instance.transcribe.return_value = [mock_segment]
        mock_model_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_piper_config() -> dict[str, Any]:
    """Sample Piper configuration for testing."""
    return {
        "num_symbols": 100,
        "num_speakers": 1,
        "audio": {"sample_rate": 22050},
        "espeak": {"voice": "en-us"},
        "phoneme_id_map": {
            "BOS": [1],
            "EOS": [2],
            "PAD": [3],
            "a": [4],
            "b": [5],
            "c": [6],
        },
        "phoneme_type": "espeak",
        "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
    }


@pytest.fixture
def mock_piper_voice(
    sample_piper_config: dict[str, Any],
) -> Generator[MagicMock, None, None]:
    """Mock PiperVoice for testing."""
    if not piper_available:
        pytest.skip("Piper not available")

    with patch("piper.src.python_run.piper.voice.PiperVoice") as mock_voice:
        mock_instance = MagicMock()
        # Create a mock config since PiperConfig might not be available
        mock_config = MagicMock()
        mock_config.sample_rate = 22050
        mock_config.num_speakers = 1
        mock_config.length_scale = 1.0
        mock_config.noise_scale = 0.667
        mock_config.noise_w = 0.8
        mock_instance.config = mock_config
        mock_instance.phonemize.return_value = [["a", "b", "c"]]
        mock_instance.phonemes_to_ids.return_value = [1, 4, 5, 6, 2]
        mock_instance.synthesize_ids_to_raw.return_value = b"fake_audio_data"
        mock_voice.load.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def temp_audio_file() -> Generator[Path, None, None]:
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        yield Path(tmp.name)
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing."""
    return "Hello, this is a test message for speech synthesis."


@pytest.fixture
def sample_phonemes() -> list[str]:
    """Sample phonemes for testing."""
    return ["h", "ə", "l", "oʊ"]


@pytest.fixture
def sample_phoneme_ids() -> list[int]:
    """Sample phoneme IDs for testing."""
    return [1, 4, 5, 6, 2]  # BOS + phonemes + EOS
