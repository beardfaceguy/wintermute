"""Unit tests for the Piper TTS endpoint (Design A, whole-response REST)."""

from __future__ import annotations

import wave
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def reset_tts_module():
    """Force the lazy loader to re-evaluate between tests."""
    from app.api import tts

    tts._voice = None
    tts._load_error = None
    yield tts
    tts._voice = None
    tts._load_error = None


def _writes_minimal_wav(text, wav_file, **_kwargs):
    """Stand-in for PiperVoice.synthesize: writes a tiny silent WAV."""
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(22050)
    wav_file.writeframes(b"\x00\x00" * 100)


class TestTTSHealth:
    def test_health_disabled_when_voice_missing(
        self, client: TestClient, reset_tts_module, tmp_path
    ) -> None:
        missing = tmp_path / "absent.onnx"
        with patch.object(reset_tts_module, "_VOICE_PATH", missing):
            resp = client.get("/api/chat/speak/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is False
        assert "missing" in (body["error"] or "").lower()

    def test_health_disabled_when_flag_off(
        self, client: TestClient, reset_tts_module
    ) -> None:
        with patch.object(reset_tts_module, "_ENABLED_FLAG", False):
            resp = client.get("/api/chat/speak/health")
        body = resp.json()
        assert body["enabled"] is False
        assert "PIPER_ENABLED" in (body["error"] or "")

    def test_health_enabled_when_voice_loaded(
        self, client: TestClient, reset_tts_module
    ) -> None:
        reset_tts_module._voice = MagicMock()
        resp = client.get("/api/chat/speak/health")
        body = resp.json()
        assert body["enabled"] is True
        assert body["error"] is None


class TestTTSSynthesize:
    def test_synthesize_returns_wav(
        self, client: TestClient, reset_tts_module
    ) -> None:
        fake_voice = MagicMock()
        fake_voice.synthesize.side_effect = _writes_minimal_wav
        reset_tts_module._voice = fake_voice

        resp = client.post("/api/chat/speak", json={"text": "Hello there."})

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert int(resp.headers["content-length"]) == len(resp.content)
        # Confirm the body is a real WAV (RIFF header).
        assert resp.content[:4] == b"RIFF"
        assert resp.content[8:12] == b"WAVE"
        fake_voice.synthesize.assert_called_once()
        called_args, called_kwargs = fake_voice.synthesize.call_args
        assert called_args[0] == "Hello there."
        assert isinstance(called_args[1], wave.Wave_write)
        assert "length_scale" in called_kwargs
        assert "noise_scale" in called_kwargs
        assert "noise_w" in called_kwargs

    def test_synthesize_rejects_empty_text(
        self, client: TestClient, reset_tts_module
    ) -> None:
        # Pydantic validation should fire before lazy load.
        resp = client.post("/api/chat/speak", json={"text": ""})
        assert resp.status_code == 422

    def test_synthesize_rejects_whitespace_text(
        self, client: TestClient, reset_tts_module
    ) -> None:
        reset_tts_module._voice = MagicMock()
        resp = client.post("/api/chat/speak", json={"text": "   "})
        assert resp.status_code == 400

    def test_synthesize_rejects_oversize_text(
        self, client: TestClient, reset_tts_module
    ) -> None:
        reset_tts_module._voice = MagicMock()
        with patch.object(reset_tts_module, "_MAX_INPUT_CHARS", 10):
            resp = client.post(
                "/api/chat/speak", json={"text": "this string is far too long"}
            )
        assert resp.status_code == 413

    def test_synthesize_503_when_voice_unavailable(
        self, client: TestClient, reset_tts_module, tmp_path
    ) -> None:
        missing = tmp_path / "absent.onnx"
        with patch.object(reset_tts_module, "_VOICE_PATH", missing):
            resp = client.post("/api/chat/speak", json={"text": "Hi"})
        assert resp.status_code == 503
        assert "TTS unavailable" in resp.json()["detail"]

    def test_synthesize_500_on_piper_error(
        self, client: TestClient, reset_tts_module
    ) -> None:
        fake_voice = MagicMock()
        fake_voice.synthesize.side_effect = RuntimeError("boom")
        reset_tts_module._voice = fake_voice

        resp = client.post("/api/chat/speak", json={"text": "Hello"})
        assert resp.status_code == 500
        assert "synthesis failed" in resp.json()["detail"]
