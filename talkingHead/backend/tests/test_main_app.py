"""Tests for app.main — FastAPI application wiring."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_cors_middleware_configured():
    middleware_classes = [type(m).__name__ for m in app.user_middleware]
    route_classes = [type(r).__name__ for r in app.routes]

    has_cors = any("CORS" in name or "cors" in name for name in middleware_classes) or any(
        "CORS" in name for name in route_classes
    )
    if not has_cors:
        from starlette.middleware.cors import CORSMiddleware

        middleware_found = False
        for m in app.user_middleware:
            if m.cls is CORSMiddleware:
                middleware_found = True
                break
        assert middleware_found, "CORSMiddleware not found on app"


def test_voice_chat_router_included():
    paths = [r.path for r in app.routes]
    assert any("/api/chat/voice" in p for p in paths), f"voice route not in {paths}"


def test_chat_ws_router_included():
    paths = [r.path for r in app.routes]
    assert any("/ws/chat" in p for p in paths), f"ws route not in {paths}"


def test_openapi_docs_accessible(client):
    resp = client.get("/docs")
    assert resp.status_code == 200
    assert "swagger" in resp.text.lower() or "openapi" in resp.text.lower()


def test_openapi_json_accessible(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    data = resp.json()
    assert "paths" in data


@pytest.mark.asyncio
async def test_startup_event_creates_tables():
    mock_conn = AsyncMock()
    mock_conn.run_sync = AsyncMock()

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_engine = MagicMock()
    mock_engine.begin.return_value = mock_ctx

    with patch("db.session_async.engine", mock_engine):
        from app.main import _init_db

        await _init_db()

    mock_engine.begin.assert_called_once()
    mock_conn.run_sync.assert_called_once()
