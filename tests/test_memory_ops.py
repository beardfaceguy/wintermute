"""
Tests for mcp_memory/app/services/memory_ops.py — mutable default and basic behavior.

The module uses relative imports (app.models.memory_entry) so we mock at
the module level to avoid needing pgvector/postgres at test time.
"""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Stub out pgvector + SQLAlchemy model imports so memory_ops can be imported
_mcp_keys = [
    "app", "app.models", "app.models.memory_entry",
    "app.services", "app.services.memory_ops",
    "pgvector", "pgvector.sqlalchemy",
]
_saved = {}
for _k in _mcp_keys:
    _saved[_k] = sys.modules.pop(_k, None)


class _FakeMemoryEntry:
    """Captures constructor kwargs so tests can inspect them."""
    _instances: list = []

    def __init__(self, **kw):
        self.__dict__.update(kw)
        _FakeMemoryEntry._instances.append(self)


# Build the module tree manually
_app_mod = ModuleType("app")
_models_mod = ModuleType("app.models")
_me_mod = ModuleType("app.models.memory_entry")
_me_mod.MemoryEntry = _FakeMemoryEntry
_services_mod = ModuleType("app.services")

sys.modules["app"] = _app_mod
sys.modules["app.models"] = _models_mod
sys.modules["app.models.memory_entry"] = _me_mod
sys.modules["app.services"] = _services_mod

from mcp_memory.app.services.memory_ops import add_memory_entry, get_recent_entries  # noqa: E402

# Restore originals
for _k in _mcp_keys:
    if _saved[_k] is not None:
        sys.modules[_k] = _saved[_k]
    else:
        sys.modules.pop(_k, None)
del _saved, _mcp_keys


@pytest.fixture(autouse=True)
def clear_instances():
    _FakeMemoryEntry._instances.clear()
    yield


@pytest.fixture
def mock_db():
    db = MagicMock()
    return db


def test_mutable_default_tags_not_shared(mock_db):
    """Calling add_memory_entry twice without tags must NOT share the same dict object."""
    add_memory_entry(mock_db, "first entry", [0.1, 0.2])
    add_memory_entry(mock_db, "second entry", [0.3, 0.4])

    first_tags = _FakeMemoryEntry._instances[0].tags
    second_tags = _FakeMemoryEntry._instances[1].tags

    assert first_tags is not second_tags, (
        "Mutable default argument: both calls share the same dict object. "
        "If one caller mutates it, all future calls see the mutation."
    )


def test_explicit_tags_passed_through(mock_db):
    """Explicit tags should be forwarded to MemoryEntry."""
    add_memory_entry(mock_db, "text", [0.1], tags={"key": "val"})
    assert _FakeMemoryEntry._instances[0].tags == {"key": "val"}


def test_entry_committed_and_refreshed(mock_db):
    """add_memory_entry should add, commit, and refresh."""
    add_memory_entry(mock_db, "text", [0.1])
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once()


def test_zone_defaults_to_live(mock_db):
    """New entries should default to zone='live'."""
    add_memory_entry(mock_db, "text", [0.1])
    assert _FakeMemoryEntry._instances[0].zone == "live"
