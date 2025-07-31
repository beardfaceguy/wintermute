# thVoice Test Suite

This directory contains comprehensive unit tests for the thVoice speech processing
components.

## Test Structure

### Test Categories

- **Unit Tests**: Test individual components in isolation
  - `test_whisper.py`: Whisper speech-to-text functionality
  - `test_piper.py`: Piper text-to-speech functionality

- **Integration Tests**: Test component interactions
  - End-to-end speech processing workflows
  - Performance and reliability tests

### Test Files

- `conftest.py`: Pytest configuration and shared fixtures
- `test_whisper.py`: Whisper STT tests
- `test_piper.py`: Piper TTS tests
- `test_runner.py`: Test execution script

## Running Tests

### Prerequisites

1. Install test dependencies:

```bash
pip install -r requirements-test.txt
```

1. Ensure you have the required models:
   - Whisper model: `models/ggml-base.en.bin`
   - Piper model: `models/en_GB-cori-medium.onnx`

### Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_whisper.py -v
python -m pytest tests/test_piper.py -v

# Run with test runner
python tests/test_runner.py unit
python tests/test_runner.py integration
python tests/test_runner.py all
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)

- **Test Discovery**: Automatically finds tests in `tests/` directory
- **Async Support**: Configured for async test execution
- **Markers**: Defined markers for test categorization
- **Output**: Verbose output with short tracebacks

### Fixtures (`conftest.py`)

#### Audio Processing Fixtures

- `sample_audio_data`: Generated WAV audio data for testing
- `temp_audio_file`: Temporary audio file for I/O tests
- `sample_text`: Sample text for TTS testing
- `sample_phonemes`: Sample phonemes for testing
- `sample_phoneme_ids`: Sample phoneme IDs for testing

#### Mock Fixtures

- `mock_whisper_model`: Mocked Whisper model
- `mock_piper_voice`: Mocked Piper voice
- `sample_piper_config`: Sample Piper configuration

## Writing Tests

### Async Tests

Use `@pytest.mark.asyncio` for async test methods:

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

### Mocking

Use the provided mock fixtures for external dependencies:

```python
def test_with_mock(mock_whisper_model):
    # Test with mocked Whisper model
    result = mock_whisper_model.transcribe("test.wav")
    assert result is not None
```

### Audio Testing

Use the audio fixtures for file I/O tests:

```python
def test_audio_processing(sample_audio_data, temp_audio_file):
    # Write test audio data
    temp_audio_file.write_bytes(sample_audio_data)

    # Test processing
    result = process_audio(temp_audio_file)
    assert result is not None
```

## Test Implementation

### Whisper Tests (`test_whisper.py`)

- **Model Initialization**: Test model loading and setup
- **Transcription**: Test speech-to-text conversion
- **Error Handling**: Test invalid inputs and failures
- **Performance**: Test timing and resource usage
- **Integration**: Test complete STT workflows

### Piper Tests (`test_piper.py`)

- **Voice Initialization**: Test voice model loading
- **Phonemization**: Test text-to-phoneme conversion
- **Synthesis**: Test phoneme-to-audio conversion
- **Configuration**: Test voice configuration handling
- **Streaming**: Test real-time audio generation
- **Error Handling**: Test synthesis failures
- **Multilingual**: Test different language support

## Best Practices

### Test Organization

1. **Group Related Tests**: Use test classes to organize related functionality
2. **Descriptive Names**: Use clear, descriptive test method names
3. **Documentation**: Include docstrings explaining test purpose
4. **Isolation**: Each test should be independent and not rely on others

### Mocking Strategy

1. **External Dependencies**: Mock external libraries and APIs
2. **File I/O**: Use temporary files and cleanup
3. **Network Calls**: Mock HTTP requests and responses
4. **Heavy Operations**: Mock computationally expensive operations

### Error Testing

1. **Exception Handling**: Test expected exceptions and error conditions
2. **Edge Cases**: Test boundary conditions and invalid inputs
3. **Recovery**: Test error recovery and cleanup

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure test dependencies are installed
2. **Model Loading**: Check model file paths and availability
3. **Async Issues**: Use proper async test decorators
4. **Mock Setup**: Verify mock configurations and return values

### Debugging

1. **Verbose Output**: Use `-v` flag for detailed test output
2. **Single Test**: Run individual tests for focused debugging
3. **Mock Inspection**: Check mock call history and arguments
4. **Logging**: Add logging to understand test flow

## CI Integration

### GitHub Actions

```yaml
- name: Run thVoice Tests
  run: |
    cd thVoice
    pip install -r requirements-test.txt
    python -m pytest tests/ -v
```

### Coverage

```bash
# Install coverage
pip install pytest-cov

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## Performance Testing

### Timing Tests

```python
def test_performance():
    import time
    start_time = time.time()
    # ... test code ...
    end_time = time.time()
    assert end_time - start_time < 1.0  # Should complete quickly
```

### Memory Testing

```python
def test_memory_usage():
    import psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    # ... test code ...

    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase
```
