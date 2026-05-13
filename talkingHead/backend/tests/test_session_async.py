"""Tests for db.session_async — async engine and session factory."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _import_fresh():
    """Force-reimport session_async so module-level code re-executes."""
    import importlib

    import db.session_async as mod

    importlib.reload(mod)
    return mod


def test_default_database_url_is_sqlite():
    with patch.dict("os.environ", {}, clear=True):
        mod = _import_fresh()
        assert "sqlite" in mod.CHAT_DB_URL


def test_database_url_env_override():
    custom = "postgresql+asyncpg://user:pass@host/db"
    with patch.dict("os.environ", {"CHAT_DB_URL": custom}):
        mod = _import_fresh()
        assert mod.CHAT_DB_URL == custom


def test_engine_is_created():
    mod = _import_fresh()
    assert mod.engine is not None
    assert hasattr(mod.engine, "begin")


def test_async_session_local_is_session_maker():
    mod = _import_fresh()
    assert callable(mod.AsyncSessionLocal)


@pytest.mark.asyncio
async def test_session_local_produces_session():
    """AsyncSessionLocal() should yield an AsyncSession context manager."""
    from db.session_async import AsyncSessionLocal

    session = AsyncSessionLocal()
    assert hasattr(session, "__aenter__")
    assert hasattr(session, "__aexit__")
