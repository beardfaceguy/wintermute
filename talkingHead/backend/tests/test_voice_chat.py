"""
Unit tests for voice chat API endpoint.
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestVoiceChatAPI:
    """Test cases for voice chat API endpoint."""

    def test_voice_input_success(
        self, 
        client: TestClient, 
        mock_whisper_model: MagicMock,
        sample_audio_file: bytes
    ) -> None:
        """Test successful voice transcription."""
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(sample_audio_file)
            tmp.flush()
            
            # Mock ffmpeg subprocess
            with patch("app.api.voice_chat.subprocess.run") as mock_ffmpeg:
                mock_ffmpeg.return_value = MagicMock()
                
                # Test the endpoint
                with open(tmp.name, "rb") as audio_file:
                    response = client.post(
                        "/api/chat/voice",
                        files={"file": ("test.webm", audio_file, "audio/webm")}
                    )
                
                # Cleanup
                Path(tmp.name).unlink()
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "transcript" in data
        assert data["transcript"] == "Hello world"
        
        # Verify ffmpeg was called
        mock_ffmpeg.assert_called_once()
        
        # Verify whisper model was called
        mock_whisper_model.transcribe.assert_called_once()

    def test_voice_input_no_file(self, client: TestClient) -> None:
        """Test voice input without file."""
        response = client.post("/api/chat/voice")
        assert response.status_code == 422  # Validation error

    def test_voice_input_ffmpeg_failure(
        self, 
        client: TestClient, 
        sample_audio_file: bytes
    ) -> None:
        """Test voice input when ffmpeg conversion fails."""
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(sample_audio_file)
            tmp.flush()
            
            # Mock ffmpeg to raise an exception
            with patch("app.api.voice_chat.subprocess.run") as mock_ffmpeg:
                mock_ffmpeg.side_effect = Exception("FFmpeg failed")
                
                with open(tmp.name, "rb") as audio_file:
                    response = client.post(
                        "/api/chat/voice",
                        files={"file": ("test.webm", audio_file, "audio/webm")}
                    )
                
                Path(tmp.name).unlink()
        
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert "STT failed" in data["error"]

    def test_voice_input_whisper_failure(
        self, 
        client: TestClient, 
        mock_whisper_model: MagicMock,
        sample_audio_file: bytes
    ) -> None:
        """Test voice input when whisper transcription fails."""
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(sample_audio_file)
            tmp.flush()
            
            # Mock ffmpeg success but whisper failure
            with patch("app.api.voice_chat.subprocess.run") as mock_ffmpeg:
                mock_ffmpeg.return_value = MagicMock()
                mock_whisper_model.transcribe.side_effect = Exception("Whisper failed")
                
                with open(tmp.name, "rb") as audio_file:
                    response = client.post(
                        "/api/chat/voice",
                        files={"file": ("test.webm", audio_file, "audio/webm")}
                    )
                
                Path(tmp.name).unlink()
        
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert "STT failed" in data["error"]

    def test_sha256sum_function(self) -> None:
        """Test the sha256sum utility function."""
        from app.api.voice_chat import sha256sum
        
        # Create a temporary file with known content
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test content")
            tmp.flush()
            
            # Calculate hash
            hash_result = sha256sum(Path(tmp.name))
            
            # Cleanup
            Path(tmp.name).unlink()
        
        # Verify it's a valid SHA256 hash (64 hex characters)
        assert len(hash_result) == 64
        assert all(c in "0123456789abcdef" for c in hash_result)

    @patch("app.api.voice_chat.WHISPER_MODEL_PATH")
    def test_whisper_model_not_found(self, mock_path: MagicMock) -> None:
        """Test that the app fails to start if whisper model is missing."""
        mock_path.exists.return_value = False
        
        # This test would need to be run in isolation since it affects module loading
        # For now, we'll skip it in the main test suite
        pytest.skip("Module loading test requires isolation") 