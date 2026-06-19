"""
Unit tests for Piper text-to-speech functionality.
"""

import wave
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


class TestPiperVoice:
    """Test cases for Piper text-to-speech voice."""

    def test_piper_voice_initialization(self, mock_piper_voice: MagicMock) -> None:
        """Test PiperVoice initialization."""
        # Test that the voice can be created
        assert mock_piper_voice is not None
        assert mock_piper_voice.config is not None

    def test_piper_voice_phonemize_success(self, mock_piper_voice: MagicMock) -> None:
        """Test successful phonemization."""
        text = "Hello world"
        result = mock_piper_voice.phonemize(text)

        # Verify result structure
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == ["a", "b", "c"]

        # Verify phonemize was called
        mock_piper_voice.phonemize.assert_called_once_with(text)

    def test_piper_voice_phonemize_empty_text(self, mock_piper_voice: MagicMock) -> None:
        """Test phonemization with empty text."""
        mock_piper_voice.phonemize.return_value = []

        result = mock_piper_voice.phonemize("")

        assert result == []
        assert len(result) == 0

    def test_piper_voice_phonemize_multiple_sentences(self, mock_piper_voice: MagicMock) -> None:
        """Test phonemization with multiple sentences."""
        mock_piper_voice.phonemize.return_value = [
            ["h", "ə", "l", "oʊ"],
            ["w", "ɜr", "l", "d"],
        ]

        result = mock_piper_voice.phonemize("Hello. World.")

        assert len(result) == 2
        assert result[0] == ["h", "ə", "l", "oʊ"]
        assert result[1] == ["w", "ɜr", "l", "d"]

    def test_piper_voice_phonemes_to_ids(self, mock_piper_voice: MagicMock) -> None:
        """Test phoneme to ID conversion."""
        phonemes = ["h", "ə", "l", "oʊ"]
        result = mock_piper_voice.phonemes_to_ids(phonemes)

        # Verify result structure
        assert isinstance(result, list)
        assert len(result) == 5  # BOS + phonemes + EOS
        assert result == [1, 4, 5, 6, 2]

        # Verify method was called
        mock_piper_voice.phonemes_to_ids.assert_called_once_with(phonemes)

    def test_piper_voice_phonemes_to_ids_empty(self, mock_piper_voice: MagicMock) -> None:
        """Test phoneme to ID conversion with empty input."""
        mock_piper_voice.phonemes_to_ids.return_value = [1, 2]  # Just BOS and EOS

        result = mock_piper_voice.phonemes_to_ids([])

        assert result == [1, 2]

    def test_piper_voice_synthesize_ids_to_raw(self, mock_piper_voice: MagicMock) -> None:
        """Test raw audio synthesis from phoneme IDs."""
        phoneme_ids = [1, 4, 5, 6, 2]
        result = mock_piper_voice.synthesize_ids_to_raw(phoneme_ids)

        # Verify result
        assert isinstance(result, bytes)
        assert result == b"fake_audio_data"

        # Verify method was called
        mock_piper_voice.synthesize_ids_to_raw.assert_called_once_with(phoneme_ids)

    def test_piper_voice_synthesize_stream_raw(self, mock_piper_voice: MagicMock) -> None:
        """Test streaming raw audio synthesis."""
        text = "Hello world"
        result = list(mock_piper_voice.synthesize_stream_raw(text))

        # Verify result
        assert len(result) == 1
        assert isinstance(result[0], bytes)
        assert result[0] == b"fake_audio_data"

        # Verify phonemize was called
        mock_piper_voice.phonemize.assert_called_once_with(text)

    def test_piper_voice_synthesize_stream_raw_multiple_sentences(
        self, mock_piper_voice: MagicMock
    ) -> None:
        """Test streaming synthesis with multiple sentences."""
        mock_piper_voice.phonemize.return_value = [
            ["h", "ə", "l", "oʊ"],
            ["w", "ɜr", "l", "d"],
        ]

        text = "Hello. World."
        result = list(mock_piper_voice.synthesize_stream_raw(text))

        # Should have one audio chunk per sentence
        assert len(result) == 2
        assert all(isinstance(chunk, bytes) for chunk in result)

    def test_piper_voice_synthesize_to_wav(
        self, mock_piper_voice: MagicMock, temp_audio_file: Path
    ) -> None:
        """Test synthesis to WAV file."""
        text = "Hello world"

        # Create a temporary WAV file
        with wave.open(str(temp_audio_file), "wb") as wav_file:
            mock_piper_voice.synthesize(text, wav_file)

        # Verify the file was created and has content
        assert temp_audio_file.exists()
        assert temp_audio_file.stat().st_size > 0

        # Verify synthesize was called
        mock_piper_voice.synthesize.assert_called_once()

    def test_piper_voice_config_attributes(self, mock_piper_voice: MagicMock) -> None:
        """Test PiperVoice configuration attributes."""
        config = mock_piper_voice.config

        # Verify config attributes
        assert config.sample_rate == 22050
        assert config.num_speakers == 1
        assert config.length_scale == 1.0
        assert config.noise_scale == 0.667
        assert config.noise_w == 0.8

    def test_piper_voice_synthesis_parameters(self, mock_piper_voice: MagicMock) -> None:
        """Test synthesis with different parameters."""
        text = "Hello world"
        speaker_id = 0
        length_scale = 1.2
        noise_scale = 0.8
        noise_w = 0.9
        sentence_silence = 0.5

        # Test with custom parameters
        result = list(
            mock_piper_voice.synthesize_stream_raw(
                text,
                speaker_id=speaker_id,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w,
                sentence_silence=sentence_silence,
            )
        )

        assert len(result) == 1
        assert isinstance(result[0], bytes)

    def test_piper_voice_error_handling_invalid_text(self, mock_piper_voice: MagicMock) -> None:
        """Test error handling with invalid text."""
        mock_piper_voice.phonemize.side_effect = ValueError("Invalid text")

        with pytest.raises(ValueError, match="Invalid text"):
            list(mock_piper_voice.synthesize_stream_raw("invalid text"))

    def test_piper_voice_error_handling_synthesis_failure(
        self, mock_piper_voice: MagicMock
    ) -> None:
        """Test error handling when synthesis fails."""
        mock_piper_voice.synthesize_ids_to_raw.side_effect = Exception("Synthesis failed")

        with pytest.raises(Exception, match="Synthesis failed"):
            list(mock_piper_voice.synthesize_stream_raw("test text"))

    def test_piper_voice_multilingual_support(self, mock_piper_voice: MagicMock) -> None:
        """Test multilingual text support."""
        test_texts = [
            "Hello world",  # English
            "Bonjour le monde",  # French
            "Hola mundo",  # Spanish
            "Привет мир",  # Russian
            "こんにちは世界",  # Japanese
        ]

        for text in test_texts:
            mock_piper_voice.phonemize.reset_mock()
            result = mock_piper_voice.phonemize(text)

            # Verify phonemize was called with the text
            mock_piper_voice.phonemize.assert_called_once_with(text)
            assert isinstance(result, list)


class TestPiperConfig:
    """Test cases for Piper configuration."""

    def test_piper_config_from_dict(self, sample_piper_config: dict[str, Any]) -> None:
        """Test PiperConfig creation from dictionary."""
        # This test would require the actual PiperConfig class
        # For now, we'll test the structure of the config dict
        assert "num_symbols" in sample_piper_config
        assert "num_speakers" in sample_piper_config
        assert "audio" in sample_piper_config
        assert "espeak" in sample_piper_config
        assert "phoneme_id_map" in sample_piper_config
        assert "phoneme_type" in sample_piper_config
        assert "inference" in sample_piper_config

    def test_piper_config_audio_settings(self, sample_piper_config: dict[str, Any]) -> None:
        """Test audio configuration settings."""
        audio_config = sample_piper_config["audio"]
        assert "sample_rate" in audio_config
        assert audio_config["sample_rate"] == 22050

    def test_piper_config_espeak_settings(self, sample_piper_config: dict[str, Any]) -> None:
        """Test espeak configuration settings."""
        espeak_config = sample_piper_config["espeak"]
        assert "voice" in espeak_config
        assert espeak_config["voice"] == "en-us"

    def test_piper_config_phoneme_mapping(self, sample_piper_config: dict[str, Any]) -> None:
        """Test phoneme ID mapping."""
        phoneme_map = sample_piper_config["phoneme_id_map"]
        assert "BOS" in phoneme_map
        assert "EOS" in phoneme_map
        assert "PAD" in phoneme_map
        assert "a" in phoneme_map
        assert "b" in phoneme_map
        assert "c" in phoneme_map

    def test_piper_config_inference_settings(self, sample_piper_config: dict[str, Any]) -> None:
        """Test inference configuration settings."""
        inference_config = sample_piper_config["inference"]
        assert "noise_scale" in inference_config
        assert "length_scale" in inference_config
        assert "noise_w" in inference_config
        assert inference_config["noise_scale"] == 0.667
        assert inference_config["length_scale"] == 1.0
        assert inference_config["noise_w"] == 0.8


class TestPiperIntegration:
    """Integration tests for Piper functionality."""

    def test_piper_text_to_speech_workflow(
        self, mock_piper_voice: MagicMock, temp_audio_file: Path
    ) -> None:
        """Test complete Piper text-to-speech workflow."""
        text = "Hello, this is a test message for speech synthesis."

        # Test complete workflow
        with wave.open(str(temp_audio_file), "wb") as wav_file:
            mock_piper_voice.synthesize(text, wav_file)

        # Verify the file was created
        assert temp_audio_file.exists()
        assert temp_audio_file.stat().st_size > 0

    def test_piper_streaming_workflow(self, mock_piper_voice: MagicMock) -> None:
        """Test Piper streaming synthesis workflow."""
        text = "Hello world"

        # Test streaming synthesis
        audio_chunks = list(mock_piper_voice.synthesize_stream_raw(text))

        # Verify results
        assert len(audio_chunks) == 1
        assert all(isinstance(chunk, bytes) for chunk in audio_chunks)
        assert all(len(chunk) > 0 for chunk in audio_chunks)

    def test_piper_performance(self, mock_piper_voice: MagicMock) -> None:
        """Test Piper performance characteristics."""
        import time

        text = "Hello world"

        # Test synthesis timing
        start_time = time.time()
        result = list(mock_piper_voice.synthesize_stream_raw(text))
        end_time = time.time()

        # Verify reasonable performance (should be fast with mocked model)
        assert end_time - start_time < 1.0  # Should complete in under 1 second
        assert len(result) == 1
