"""
Real-world test cases for Whisper speech-to-text using JFK sample files.
"""

import os
from pathlib import Path

import pytest
from pywhispercpp.model import Model, Segment  # type: ignore


class TestWhisperRealSamples:
    """Test cases for Whisper using real audio samples."""

    @pytest.fixture
    def jfk_wav_path(self) -> Path:
        """Path to JFK WAV sample file."""
        return Path("models/whisper.cpp/samples/jfk.wav")

    @pytest.fixture
    def jfk_mp3_path(self) -> Path:
        """Path to JFK MP3 sample file."""
        return Path("models/whisper.cpp/samples/jfk.mp3")

    @pytest.fixture
    def expected_jfk_text(self) -> str:
        """Expected transcription text for JFK speech."""
        return "And so, my fellow Americans: ask not what your country can do for you--ask what you can do for your country."

    def test_jfk_wav_file_exists(self, jfk_wav_path: Path) -> None:
        """Test that JFK WAV file exists and is accessible."""
        assert jfk_wav_path.exists(), f"JFK WAV file not found at {jfk_wav_path}"
        assert jfk_wav_path.is_file(), f"JFK WAV path is not a file: {jfk_wav_path}"
        assert jfk_wav_path.stat().st_size > 0, f"JFK WAV file is empty: {jfk_wav_path}"

    def test_jfk_mp3_file_exists(self, jfk_mp3_path: Path) -> None:
        """Test that JFK MP3 file exists and is accessible."""
        assert jfk_mp3_path.exists(), f"JFK MP3 file not found at {jfk_mp3_path}"
        assert jfk_mp3_path.is_file(), f"JFK MP3 path is not a file: {jfk_mp3_path}"
        assert jfk_mp3_path.stat().st_size > 0, f"JFK MP3 file is empty: {jfk_mp3_path}"

    def test_whisper_model_initialization_with_real_model(self) -> None:
        """Test Whisper model initialization with real model file."""
        # Check if we have a real model file
        model_paths = [
            "models/ggml-base.en.bin",
            "models/ggml-small.en.bin",
            "models/ggml-medium.en.bin",
        ]

        available_model: str | None = None
        for model_path in model_paths:
            if Path(model_path).exists():
                available_model = model_path
                break

        if available_model is None:
            pytest.skip("No real Whisper model files found")

        # Test model initialization
        model: Model = Model(available_model)
        assert model is not None
        assert hasattr(model, "transcribe")

    def test_jfk_wav_transcription_accuracy(
        self, jfk_wav_path: Path, expected_jfk_text: str
    ) -> None:
        """Test transcription accuracy on JFK WAV file."""
        # Check if we have a real model
        model_paths = [
            "models/ggml-base.en.bin",
            "models/ggml-small.en.bin",
            "models/ggml-medium.en.bin",
        ]

        available_model: str | None = None
        for model_path in model_paths:
            if Path(model_path).exists():
                available_model = model_path
                break

        if available_model is None:
            pytest.skip("No real Whisper model files found")

        # Load model and transcribe
        model: Model = Model(available_model)
        result: list[Segment] = model.transcribe(str(jfk_wav_path))

        # Verify we got transcription results
        assert len(result) > 0, "No transcription segments returned"

        # Combine all segments into full text
        full_text: str = " ".join([segment.text for segment in result])

        # Normalize text for comparison (remove extra spaces, punctuation variations)
        normalized_result: str = self._normalize_text(full_text)

        # Check if the transcription contains the key phrases
        key_phrases = [
            "fellow Americans",
            "ask not what your country can do for you",
            "ask what you can do for your country",
        ]

        for phrase in key_phrases:
            assert (
                phrase.lower() in normalized_result.lower()
            ), f"Missing key phrase: {phrase}"

        # Log the actual transcription for debugging
        print(f"\nActual transcription: '{full_text}'")
        print(f"Expected transcription: '{expected_jfk_text}'")

    def test_jfk_mp3_transcription_accuracy(
        self, jfk_mp3_path: Path, expected_jfk_text: str
    ) -> None:
        """Test transcription accuracy on JFK MP3 file."""
        # Check if we have a real model
        model_paths = [
            "models/ggml-base.en.bin",
            "models/ggml-small.en.bin",
            "models/ggml-medium.en.bin",
        ]

        available_model: str | None = None
        for model_path in model_paths:
            if Path(model_path).exists():
                available_model = model_path
                break

        if available_model is None:
            pytest.skip("No real Whisper model files found")

        # Load model and transcribe
        model: Model = Model(available_model)
        result: list[Segment] = model.transcribe(str(jfk_mp3_path))

        # Verify we got transcription results
        assert len(result) > 0, "No transcription segments returned"

        # Combine all segments into full text
        full_text: str = " ".join([segment.text for segment in result])

        # Normalize text for comparison
        normalized_result: str = self._normalize_text(full_text)

        # Check if the transcription contains the key phrases
        key_phrases = [
            "fellow Americans",
            "ask not what your country can do for you",
            "ask what you can do for your country",
        ]

        for phrase in key_phrases:
            assert (
                phrase.lower() in normalized_result.lower()
            ), f"Missing key phrase: {phrase}"

        # Log the actual transcription for debugging
        print(f"\nActual transcription: '{full_text}'")
        print(f"Expected transcription: '{expected_jfk_text}'")

    def test_jfk_transcription_segment_details(self, jfk_wav_path: Path) -> None:
        """Test that transcription segments have proper attributes."""
        # Check if we have a real model
        model_paths = [
            "models/ggml-base.en.bin",
            "models/ggml-small.en.bin",
            "models/ggml-medium.en.bin",
        ]

        available_model: str | None = None
        for model_path in model_paths:
            if Path(model_path).exists():
                available_model = model_path
                break

        if available_model is None:
            pytest.skip("No real Whisper model files found")

        # Load model and transcribe
        model: Model = Model(available_model)
        result: list[Segment] = model.transcribe(str(jfk_wav_path))

        # Verify segment structure
        assert len(result) > 0, "No transcription segments returned"

        for segment in result:
            # Check required attributes
            assert hasattr(segment, "text"), "Segment missing 'text' attribute"
            assert isinstance(segment.text, str), "Segment text is not a string"
            assert len(segment.text.strip()) > 0, "Segment text is empty"

            # Check optional attributes if they exist
            # Note: These attributes may not be available on all Whisper segment objects
            if hasattr(segment, "t0"):
                assert isinstance(
                    segment.t0, (int, float)
                ), "Segment start time is not numeric"
            if hasattr(segment, "t1"):
                assert isinstance(
                    segment.t1, (int, float)
                ), "Segment end time is not numeric"

    def test_jfk_transcription_performance(self, jfk_wav_path: Path) -> None:
        """Test transcription performance on JFK sample."""
        import time

        # Check if we have a real model
        model_paths = [
            "models/ggml-base.en.bin",
            "models/ggml-small.en.bin",
            "models/ggml-medium.en.bin",
        ]

        available_model: str | None = None
        for model_path in model_paths:
            if Path(model_path).exists():
                available_model = model_path
                break

        if available_model is None:
            pytest.skip("No real Whisper model files found")

        # Load model
        model: Model = Model(available_model)

        # Test transcription timing
        start_time = time.time()
        result: list[Segment] = model.transcribe(str(jfk_wav_path))
        end_time = time.time()

        transcription_time = end_time - start_time

        # Verify reasonable performance (should complete in reasonable time)
        assert (
            transcription_time < 30.0
        ), f"Transcription took too long: {transcription_time:.2f}s"
        assert len(result) > 0, "No transcription results returned"

        print(f"\nTranscription completed in {transcription_time:.2f} seconds")
        print(f"Number of segments: {len(result)}")

    def test_jfk_transcription_consistency(self, jfk_wav_path: Path) -> None:
        """Test that transcription is consistent across multiple runs."""
        # Check if we have a real model
        model_paths = [
            "models/ggml-base.en.bin",
            "models/ggml-small.en.bin",
            "models/ggml-medium.en.bin",
        ]

        available_model: str | None = None
        for model_path in model_paths:
            if Path(model_path).exists():
                available_model = model_path
                break

        if available_model is None:
            pytest.skip("No real Whisper model files found")

        # Load model
        model: Model = Model(available_model)

        # Run transcription multiple times
        results: list[str] = []
        for i in range(3):
            transcription_result: list[Segment] = model.transcribe(str(jfk_wav_path))
            full_text: str = " ".join(
                [segment.text for segment in transcription_result]
            )
            normalized_text: str = self._normalize_text(full_text)
            results.append(normalized_text)

        # Check that all results are similar (allowing for minor variations)
        first_result: str = results[0]
        for i, result_text in enumerate(results[1:], 1):
            # Check if results are similar (at least 80% similarity)
            similarity: float = self._calculate_text_similarity(
                first_result, result_text
            )
            assert (
                similarity > 0.8
            ), f"Transcription results too different (similarity: {similarity:.2f})"
            print(f"Run {i} similarity: {similarity:.2f}")

    def test_jfk_transcription_with_different_models(self, jfk_wav_path: Path) -> None:
        """Test transcription with different model sizes."""
        model_paths = [
            "models/ggml-base.en.bin",
            "models/ggml-small.en.bin",
            "models/ggml-medium.en.bin",
        ]

        available_models: list[str] = [
            path for path in model_paths if Path(path).exists()
        ]

        if len(available_models) < 2:
            pytest.skip("Need at least 2 different models for comparison")

        results: dict[str, str] = {}

        for model_path in available_models:
            print(f"\nTesting with model: {model_path}")
            model: Model = Model(model_path)
            result: list[Segment] = model.transcribe(str(jfk_wav_path))
            full_text: str = " ".join([segment.text for segment in result])
            normalized_text: str = self._normalize_text(full_text)
            results[model_path] = normalized_text

            # Verify basic transcription quality
            assert len(result) > 0, f"No transcription results for {model_path}"
            assert (
                len(normalized_text) > 50
            ), f"Transcription too short for {model_path}"

            # Check for key phrases
            key_phrases = ["fellow Americans", "country", "ask"]
            for phrase in key_phrases:
                assert (
                    phrase.lower() in normalized_text.lower()
                ), f"Missing '{phrase}' in {model_path}"

        # Compare results across models
        model_names: list[str] = list(results.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                similarity: float = self._calculate_text_similarity(
                    results[model_names[i]], results[model_names[j]]
                )
                print(
                    f"Similarity between {model_names[i]} and {model_names[j]}: {similarity:.2f}"
                )
                # Models should produce reasonably similar results
                assert (
                    similarity > 0.6
                ), f"Models produce too different results (similarity: {similarity:.2f})"

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison by removing extra spaces and punctuation variations."""
        import re

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove common punctuation variations
        text = text.replace("--", " ")
        text = text.replace("-", " ")
        text = text.replace(":", " ")
        text = text.replace(";", " ")
        text = text.replace(",", " ")
        text = text.replace(".", " ")

        # Remove extra spaces again
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using simple word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)


class TestWhisperErrorHandling:
    """Test error handling for Whisper with real files."""

    def test_whisper_nonexistent_file(self) -> None:
        """Test Whisper error handling with nonexistent file."""
        # Check if we have a real model
        model_paths = [
            "models/ggml-base.en.bin",
            "models/ggml-small.en.bin",
            "models/ggml-medium.en.bin",
        ]

        available_model: str | None = None
        for model_path in model_paths:
            if Path(model_path).exists():
                available_model = model_path
                break

        if available_model is None:
            pytest.skip("No real Whisper model files found")

        model: Model = Model(available_model)

        with pytest.raises(FileNotFoundError):
            model.transcribe("nonexistent_audio_file.wav")  # type: ignore

    def test_whisper_invalid_model_path(self) -> None:
        """Test Whisper error handling with invalid model path."""
        # The Model constructor may not raise immediately, but should fail when used
        model: Model = Model("nonexistent_model.bin")
        # The error should occur when trying to transcribe
        with pytest.raises(Exception):
            model.transcribe("test.wav")  # type: ignore

    def test_whisper_empty_audio_file(self) -> None:
        """Test Whisper error handling with empty audio file."""
        import tempfile

        # Check if we have a real model
        model_paths = [
            "models/ggml-base.en.bin",
            "models/ggml-small.en.bin",
            "models/ggml-medium.en.bin",
        ]

        available_model: str | None = None
        for model_path in model_paths:
            if Path(model_path).exists():
                available_model = model_path
                break

        if available_model is None:
            pytest.skip("No real Whisper model files found")

        model: Model = Model(available_model)

        # Create an empty file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"")
            tmp_path = tmp.name

        try:
            with pytest.raises(Exception):
                model.transcribe(tmp_path)  # type: ignore
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)  # type: ignore
