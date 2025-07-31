# TalkingHead Backend Tests

This directory contains comprehensive unit tests for the TalkingHead backend components.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest configuration and fixtures
├── test_voice_chat.py       # Voice chat API endpoint tests
├── test_websocket.py        # WebSocket functionality tests
├── test_db_ops.py          # Database operations tests
├── test_llm.py             # LLM chat processor tests
├── test_runner.py          # Test runner script
└── README.md               # This file
```

## Test Categories

### Unit Tests
- **Voice Chat API** (`test_voice_chat.py`): Tests for audio upload and transcription
- **WebSocket** (`test_websocket.py`): Connection management and chat endpoint tests
- **Database Operations** (`test_db_ops.py`): Message storage and retrieval tests
- **LLM Processing** (`test_llm.py`): Chat processor and streaming tests

### Integration Tests
- **End-to-end workflows**: Complete user interaction flows
- **API integration**: Full request/response cycles
- **Database integration**: Real database operations

## Running Tests

### Prerequisites
1. Activate the virtual environment:
   ```bash
   source ../../venv/bin/activate
   ```

2. Install test dependencies:
   ```bash
   pip install -r requirements-test.txt
   ```

### Running All Tests
```bash
# From the backend directory
python -m pytest tests/ -v

# Or use the test runner
python tests/test_runner.py
```

### Running Specific Test Categories
```bash
# Unit tests only
python tests/test_runner.py unit

# Integration tests only
python tests/test_runner.py integration

# All tests
python tests/test_runner.py all

# Specific test
python tests/test_runner.py specific tests/test_voice_chat.py::TestVoiceChatAPI::test_voice_input_success
```

### Direct Pytest Commands
```bash
# Run all tests with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_voice_chat.py -v

# Run specific test class
pytest tests/test_websocket.py::TestConnectionManager -v

# Run specific test method
pytest tests/test_llm.py::TestChatProcessor::test_chat_processor_initialization -v

# Run tests with coverage
pytest tests/ --cov=app --cov-report=html
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)
- **asyncio_mode**: `auto` - Automatically handles async tests
- **testpaths**: `tests` - Test discovery directory
- **python_files**: `test_*.py` - Test file pattern
- **python_classes**: `Test*` - Test class pattern
- **python_functions**: `test_*` - Test function pattern

### Fixtures (`conftest.py`)
- **client**: FastAPI TestClient instance
- **mock_whisper_model**: Mocked Whisper transcription model
- **mock_chat_processor**: Mocked LLM chat processor
- **mock_websocket**: Mocked WebSocket connection
- **sample_audio_file**: Sample audio data for testing
- **sample_message_data**: Sample message data for WebSocket tests

## Test Coverage

### Voice Chat API Tests
- ✅ Successful audio transcription
- ✅ Missing file handling
- ✅ FFmpeg conversion failures
- ✅ Whisper transcription failures
- ✅ SHA256 checksum calculation
- ⏭️ Model loading validation (skipped - requires isolation)

### WebSocket Tests
- ✅ Connection management (connect/disconnect)
- ✅ Personal message sending
- ✅ Broadcast messaging
- ✅ Chat endpoint success flow
- ✅ Empty message handling
- ✅ Invalid JSON handling
- ✅ Exception handling
- ✅ WebSocket disconnect handling

### Database Tests
- ✅ Message storage with required fields
- ✅ Message storage with optional fields
- ✅ Exception handling during storage
- ✅ Message retrieval with limit
- ✅ Empty result handling
- ✅ Message ordering (chronological)

### LLM Tests
- ✅ ChatProcessor initialization
- ✅ Default configuration loading
- ✅ Successful streaming response
- ✅ Empty response handling
- ✅ JSON parse error handling
- ✅ Missing text field handling
- ✅ Empty line skipping
- ✅ Exception handling

## Writing New Tests

### Async Test Pattern
```python
import pytest
from unittest.mock import AsyncMock, MagicMock

class TestMyComponent:
    @pytest.mark.asyncio
    async def test_async_function(self, mock_dependency: MagicMock) -> None:
        """Test async function."""
        # Setup
        mock_dependency.return_value = "expected_result"
        
        # Execute
        result = await my_async_function()
        
        # Assert
        assert result == "expected_result"
```

### Mocking External Dependencies
```python
from unittest.mock import patch

@patch("app.module.external_dependency")
def test_with_mock(self, mock_dependency: MagicMock) -> None:
    """Test with mocked external dependency."""
    mock_dependency.return_value = "mocked_result"
    # ... test implementation
```

### Database Test Pattern
```python
@pytest.mark.asyncio
async def test_database_operation(self, async_session) -> None:
    """Test database operation."""
    # Use the async_session fixture for database tests
    # ... test implementation
```

## Best Practices

1. **Use descriptive test names**: Test names should clearly describe what is being tested
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Mock external dependencies**: Don't rely on external services in unit tests
4. **Test error conditions**: Include tests for failure scenarios
5. **Use fixtures**: Reuse common test setup with fixtures
6. **Keep tests isolated**: Tests should not depend on each other
7. **Use type hints**: Include proper type annotations for better IDE support

## Troubleshooting

### Common Issues

1. **Async test failures**: Ensure `@pytest.mark.asyncio` decorator is used
2. **Import errors**: Check that the virtual environment is activated
3. **Mock not working**: Verify the correct import path is being mocked
4. **Database connection issues**: Ensure test database is properly configured

### Debug Mode
```bash
# Run tests with debug output
pytest tests/ -v -s

# Run specific test with debug
pytest tests/test_voice_chat.py::TestVoiceChatAPI::test_voice_input_success -v -s
```

## Continuous Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    source venv/bin/activate
    cd talkingHead/backend
    python -m pytest tests/ -v --cov=app --cov-report=xml
```

## Coverage Reports

Generate coverage reports to identify untested code:

```bash
# Install coverage
pip install coverage

# Run tests with coverage
pytest tests/ --cov=app --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html
``` 