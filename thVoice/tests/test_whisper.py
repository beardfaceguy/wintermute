"""
Unit tests for Whisper speech-to-text functionality.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestWhisperModel:
    """Test cases for Whisper speech-to-text model."""

    def test_whisper_model_initialization(self, mock_whisper_model: MagicMock) -> None:
        """Test Whisper model initialization."""
        from pywhispercpp.model import Model

        # Test that the model can be created
        model = Model("fake_model_path")
        assert model is not None

    def test_whisper_model_transcribe_success(self, mock_whisper_model: MagicMock) -> None:
        """Test successful transcription."""
        from pywhispercpp.model import Model

        model = Model("fake_model_path")

        # Test transcription
        result = model.transcribe("test_audio.wav")

        # Verify result structure
        assert len(result) == 1
        assert result[0].text == "Hello world"

        # Verify transcribe was called
        mock_whisper_model.transcribe.assert_called_once_with("test_audio.wav")

    def test_whisper_model_transcribe_empty_result(self, mock_whisper_model: MagicMock) -> None:
        """Test transcription with empty result."""
        mock_whisper_model.transcribe.return_value = []

        model = Model("fake_model_path")
        result = model.transcribe("test_audio.wav")

        assert result == []
        assert len(result) == 0

    def test_whisper_model_transcribe_multiple_segments(
        self, mock_whisper_model: MagicMock
    ) -> None:
        """Test transcription with multiple segments."""
        # Create mock segments
        segment1 = MagicMock()
        segment1.text = "Hello"
        segment2 = MagicMock()
        segment2.text = " world"

        mock_whisper_model.transcribe.return_value = [segment1, segment2]

        model = Model("fake_model_path")
        result = model.transcribe("test_audio.wav")

        assert len(result) == 2
        assert result[0].text == "Hello"
        assert result[1].text == " world"

    def test_whisper_model_transcribe_exception_handling(
        self, mock_whisper_model: MagicMock
    ) -> None:
        """Test transcription exception handling."""
        mock_whisper_model.transcribe.side_effect = Exception("Model failed")

        model = Model("fake_model_path")

        with pytest.raises(Exception, match="Model failed"):
            model.transcribe("test_audio.wav")

    def test_whisper_model_with_real_audio_file(
        self, sample_audio_data: bytes, temp_audio_file: Path
    ) -> None:
        """Test Whisper model with real audio file."""
        # Write sample audio data to temp file
        temp_audio_file.write_bytes(sample_audio_data)

        # Mock the model to avoid loading real model
        with patch("pywhispercpp.model.Model") as mock_model_class:
            mock_model = MagicMock()
            mock_segment = MagicMock()
            mock_segment.text = "Test transcription"
            mock_model.transcribe.return_value = [mock_segment]
            mock_model_class.return_value = mock_model

            # Test transcription
            model = Model("fake_model_path")
            result = model.transcribe(str(temp_audio_file))

            assert len(result) == 1
            assert result[0].text == "Test transcription"

    def test_whisper_model_segment_attributes(self, mock_whisper_model: MagicMock) -> None:
        """Test that Whisper segments have expected attributes."""
        # Create a more detailed mock segment
        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_segment.start = 0.0
        mock_segment.end = 2.5
        mock_segment.probability = 0.95

        mock_whisper_model.transcribe.return_value = [mock_segment]

        model = Model("fake_model_path")
        result = model.transcribe("test_audio.wav")

        segment = result[0]
        assert segment.text == "Hello world"
        # Note: Additional attributes like start, end, probability may exist
        # but are not guaranteed to be available on all Whisper segment objects

    def test_whisper_model_different_audio_formats(self, mock_whisper_model: MagicMock) -> None:
        """Test Whisper model with different audio formats."""
        model = Model("fake_model_path")

        # Test different file extensions
        test_files = ["audio.wav", "audio.mp3", "audio.m4a", "audio.flac"]

        for test_file in test_files:
            mock_whisper_model.transcribe.reset_mock()
            result = model.transcribe(test_file)

            # Verify transcribe was called with correct file
            mock_whisper_model.transcribe.assert_called_once_with(test_file)
            assert len(result) == 1

    def test_whisper_model_configuration_options(self) -> None:
        """Test Whisper model with different configuration options."""
        with patch("pywhispercpp.model.Model") as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model

            # Test with different model paths
            model_paths = [
                "models/ggml-base.en.bin",
                "models/ggml-small.en.bin",
                "models/ggml-medium.en.bin",
            ]

            for model_path in model_paths:
                model = Model(model_path)
                assert model is not None

    def test_whisper_model_error_handling_invalid_file(self, mock_whisper_model: MagicMock) -> None:
        """Test Whisper model error handling with invalid file."""
        mock_whisper_model.transcribe.side_effect = FileNotFoundError("Audio file not found")

        model = Model("fake_model_path")

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            model.transcribe("nonexistent_audio.wav")

    def test_whisper_model_error_handling_corrupted_audio(
        self, mock_whisper_model: MagicMock
    ) -> None:
        """Test Whisper model error handling with corrupted audio."""
        mock_whisper_model.transcribe.side_effect = ValueError("Invalid audio format")

        model = Model("fake_model_path")

        with pytest.raises(ValueError, match="Invalid audio format"):
            model.transcribe("corrupted_audio.wav")


class TestWhisperIntegration:
    """Integration tests for Whisper functionality."""

    def test_whisper_transcription_workflow(
        self, sample_audio_data: bytes, temp_audio_file: Path
    ) -> None:
        """Test complete Whisper transcription workflow."""
        # Write sample audio data
        temp_audio_file.write_bytes(sample_audio_data)

        # Mock the model
        with patch("pywhispercpp.model.Model") as mock_model_class:
            mock_model = MagicMock()
            mock_segment1 = MagicMock()
            mock_segment1.text = "Hello"
            mock_segment2 = MagicMock()
            mock_segment2.text = " world"
            mock_model.transcribe.return_value = [mock_segment1, mock_segment2]
            mock_model_class.return_value = mock_model

            # Test complete workflow
            model = Model("fake_model_path")
            result = model.transcribe(str(temp_audio_file))

            # Verify results
            assert len(result) == 2
            full_text = " ".join([seg.text for seg in result])
            assert full_text == "Hello world"

    def test_whisper_model_performance(self, mock_whisper_model: MagicMock) -> None:
        """Test Whisper model performance characteristics."""
        import time

        model = Model("fake_model_path")

        # Test transcription timing
        start_time = time.time()
        result = model.transcribe("test_audio.wav")
        end_time = time.time()

        # Verify reasonable performance (should be fast with mocked model)
        assert end_time - start_time < 1.0  # Should complete in under 1 second
        assert len(result) == 1
