"""
Pytest configuration and fixtures for talkingHead backend tests.
"""

import asyncio
import os

# Add the backend directory to the path
import sys
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from db.db_models import Base
from db.session_async import AsyncSessionLocal


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_db() -> Generator[str, None, None]:
    """Create a temporary SQLite database for testing."""
    # Create a temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_url = f"sqlite:///{tmp.name}"

    # Create engine and tables
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)

    yield db_url

    # Cleanup
    os.unlink(tmp.name)


@pytest_asyncio.fixture
async def async_session(test_db: str) -> AsyncGenerator[Any, None]:
    """Create an async database session for testing."""
    # Mock the database URL
    with patch("db.session_async.DATABASE_URL", test_db):
        async with AsyncSessionLocal() as session:
            yield session


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_whisper_model() -> Generator[MagicMock, None, None]:
    """Mock the Whisper model for voice transcription tests."""
    with patch("app.api.voice_chat.whisper_model") as mock_model:
        # Mock the transcribe method
        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_model.transcribe.return_value = [mock_segment]
        yield mock_model


@pytest.fixture
def mock_chat_processor() -> Generator[MagicMock, None, None]:
    """Mock the ChatProcessor for LLM tests."""
    with patch("app.chat.llm.ChatProcessor") as mock_processor:
        mock_instance = MagicMock()
        mock_instance.stream_response = AsyncMock(return_value="Mock response")
        mock_processor.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_httpx_client() -> Generator[MagicMock, None, None]:
    """Mock httpx client for LLM API calls."""
    with patch("app.chat.llm.httpx.AsyncClient") as mock_client:
        mock_instance = MagicMock()
        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.__aexit__.return_value = None
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_audio_file() -> Generator[bytes, None, None]:
    """Create a sample audio file for testing voice upload."""
    # Create a minimal WebM audio file (just header bytes)
    # In real tests, you'd want actual audio data
    audio_data = b"\x1a\x45\xdf\xa3"  # WebM header
    yield audio_data


@pytest.fixture
def sample_message_data() -> dict[str, str]:
    """Sample message data for WebSocket tests."""
    return {"message": "Hello, this is a test message"}


@pytest.fixture
def mock_websocket() -> Generator[MagicMock, None, None]:
    """Mock WebSocket for testing."""
    mock_ws = MagicMock()
    mock_ws.send_text = AsyncMock()
    mock_ws.receive_text = AsyncMock()
    mock_ws.accept = AsyncMock()
    yield mock_ws
