"""
Tests for mcp_memory/server.py — MCP strategic memory server.

All database and embedding access is mocked; no real PostgreSQL or
sentence-transformers model is required.
"""

import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAKE_UUID = uuid.UUID("12345678-1234-5678-1234-567812345678")
FAKE_NOW = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


def _make_entry(**overrides):
    """Build a mock MemoryEntry with sensible defaults."""
    entry = MagicMock()
    entry.id = overrides.get("id", FAKE_UUID)
    entry.text = overrides.get("text", "test memory text")
    entry.tags = overrides.get("tags", {"source": "test"})
    entry.zone = overrides.get("zone", "live")
    entry.trust_score = overrides.get("trust_score", 0.5)
    entry.audit_flagged = overrides.get("audit_flagged", False)
    entry.created_at = overrides.get("created_at", FAKE_NOW)
    entry.embedding = overrides.get("embedding", None)
    return entry


# ---------------------------------------------------------------------------
# _entry_to_dict — import once, function is pure (no DB/embed)
# ---------------------------------------------------------------------------

from mcp_memory.server import _entry_to_dict


class TestEntryToDict:
    """Tests for the _entry_to_dict helper (pure function, no mocking needed)."""

    def test_converts_entry_to_dict(self):
        """All fields are present and correctly typed."""
        entry = _make_entry()
        result = _entry_to_dict(entry)
        assert result["id"] == str(FAKE_UUID)
        assert result["text"] == "test memory text"
        assert result["tags"] == {"source": "test"}
        assert result["zone"] == "live"
        assert result["trust_score"] == 0.5
        assert result["audit_flagged"] is False
        assert result["created_at"] == FAKE_NOW.isoformat()

    def test_handles_none_tags(self):
        """None tags become an empty dict."""
        result = _entry_to_dict(_make_entry(tags=None))
        assert result["tags"] == {}

    def test_handles_none_trust_score(self):
        """None trust_score becomes 0.0."""
        result = _entry_to_dict(_make_entry(trust_score=None))
        assert result["trust_score"] == 0.0

    def test_handles_none_created_at(self):
        """None created_at becomes None in the dict."""
        result = _entry_to_dict(_make_entry(created_at=None))
        assert result["created_at"] is None

    def test_handles_zero_trust_score(self):
        """Zero trust_score (falsy but valid) is preserved as 0.0."""
        result = _entry_to_dict(_make_entry(trust_score=0.0))
        assert result["trust_score"] == 0.0


# ---------------------------------------------------------------------------
# Tool function tests — patch at the call site
# ---------------------------------------------------------------------------

import mcp_memory.server as _srv


def _patch_server():
    """Return a dict of patches for all server-module DB/embed dependencies."""
    mock_session = MagicMock()
    mock_session_cls = MagicMock(return_value=mock_session)
    patches = {
        "session": mock_session,
        "session_cls": mock_session_cls,
    }
    return patches


class TestMemoryAdd:
    """Tests for the memory_add tool."""

    def test_creates_entry_with_correct_fields(self):
        """memory_add creates a MemoryEntry with the provided text, tags, and zone."""
        mock_session = MagicMock()
        mock_entry_instance = _make_entry(zone="live", tags={"role": "agent"})

        with (
            patch.object(_srv, "SessionLocal", return_value=mock_session),
            patch.object(_srv, "_embed", return_value=[0.0] * 384),
            patch.object(_srv, "_ensure_tables"),
            patch.object(_srv, "MemoryEntry", return_value=mock_entry_instance) as MockEntry,
        ):
            _srv.memory_add(text="learned something", tags={"role": "agent"}, zone="live")
            MockEntry.assert_called_once()
            kw = MockEntry.call_args[1]
            assert kw["text"] == "learned something"
            assert kw["tags"] == {"role": "agent"}
            assert kw["zone"] == "live"
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

    def test_default_tags_are_empty_dict(self):
        """Omitting tags defaults to an empty dict."""
        mock_session = MagicMock()
        with (
            patch.object(_srv, "SessionLocal", return_value=mock_session),
            patch.object(_srv, "_embed", return_value=[0.0] * 384),
            patch.object(_srv, "_ensure_tables"),
            patch.object(_srv, "MemoryEntry", return_value=_make_entry(tags={})) as MockEntry,
        ):
            _srv.memory_add(text="something")
            kw = MockEntry.call_args[1]
            assert kw["tags"] == {}

    def test_calls_embed(self):
        """memory_add calls _embed with the entry text."""
        mock_session = MagicMock()
        mock_embed = MagicMock(return_value=[0.0] * 384)
        with (
            patch.object(_srv, "SessionLocal", return_value=mock_session),
            patch.object(_srv, "_embed", mock_embed),
            patch.object(_srv, "_ensure_tables"),
            patch.object(_srv, "MemoryEntry", return_value=_make_entry()),
        ):
            _srv.memory_add(text="embed this")
            mock_embed.assert_called_once_with("embed this")


class TestMemoryPromote:
    """Tests for the memory_promote tool."""

    def test_rejects_low_trust_score(self):
        """trust_score < 0.7 is rejected before any DB access."""
        result = _srv.memory_promote(str(FAKE_UUID), trust_score=0.5)
        assert "error" in result
        assert "0.7" in result["error"]

    def test_rejects_trust_score_boundary(self):
        """trust_score of exactly 0.69 is rejected."""
        result = _srv.memory_promote(str(FAKE_UUID), trust_score=0.69)
        assert "error" in result

    def test_accepts_trust_score_at_threshold(self):
        """trust_score of exactly 0.7 is accepted."""
        mock_session = MagicMock()
        entry = _make_entry(zone="live")
        mock_session.query.return_value.filter.return_value.first.return_value = entry
        with (
            patch.object(_srv, "SessionLocal", return_value=mock_session),
            patch.object(_srv, "_ensure_tables"),
        ):
            result = _srv.memory_promote(str(FAKE_UUID), trust_score=0.7)
            assert "error" not in result

    def test_returns_error_for_nonexistent_entry(self):
        """Missing entry returns an error dict."""
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None
        with (
            patch.object(_srv, "SessionLocal", return_value=mock_session),
            patch.object(_srv, "_ensure_tables"),
        ):
            result = _srv.memory_promote(str(FAKE_UUID), trust_score=0.8)
            assert "error" in result
            assert "not found" in result["error"]

    def test_returns_error_for_non_live_entry(self):
        """Entries already in 'cold' zone cannot be promoted again."""
        mock_session = MagicMock()
        entry = _make_entry(zone="cold")
        mock_session.query.return_value.filter.return_value.first.return_value = entry
        with (
            patch.object(_srv, "SessionLocal", return_value=mock_session),
            patch.object(_srv, "_ensure_tables"),
        ):
            result = _srv.memory_promote(str(FAKE_UUID), trust_score=0.9)
            assert "error" in result
            assert "cold" in result["error"]

    def test_successful_promotion(self):
        """Successful promote sets zone='cold' and updates trust_score."""
        mock_session = MagicMock()
        entry = _make_entry(zone="live", trust_score=0.3)
        mock_session.query.return_value.filter.return_value.first.return_value = entry
        with (
            patch.object(_srv, "SessionLocal", return_value=mock_session),
            patch.object(_srv, "_ensure_tables"),
        ):
            _srv.memory_promote(str(FAKE_UUID), trust_score=0.95)
            assert entry.zone == "cold"
            assert entry.trust_score == 0.95
            assert entry.audit_flagged is False
            mock_session.commit.assert_called_once()


class TestMemoryFlag:
    """Tests for the memory_flag tool."""

    def test_returns_error_for_nonexistent_entry(self):
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None
        with (
            patch.object(_srv, "SessionLocal", return_value=mock_session),
            patch.object(_srv, "_ensure_tables"),
        ):
            result = _srv.memory_flag(str(FAKE_UUID))
            assert "error" in result

    def test_sets_audit_flagged(self):
        mock_session = MagicMock()
        entry = _make_entry(audit_flagged=False, tags={"source": "test"})
        mock_session.query.return_value.filter.return_value.first.return_value = entry
        with (
            patch.object(_srv, "SessionLocal", return_value=mock_session),
            patch.object(_srv, "_ensure_tables"),
        ):
            _srv.memory_flag(str(FAKE_UUID))
            assert entry.audit_flagged is True
            mock_session.commit.assert_called_once()

    def test_adds_reason_to_tags(self):
        mock_session = MagicMock()
        entry = _make_entry(tags={"source": "test"})
        mock_session.query.return_value.filter.return_value.first.return_value = entry
        with (
            patch.object(_srv, "SessionLocal", return_value=mock_session),
            patch.object(_srv, "_ensure_tables"),
        ):
            _srv.memory_flag(str(FAKE_UUID), reason="contradicts entry 42")
            assert entry.tags["audit_reason"] == "contradicts entry 42"

    def test_flag_without_reason_leaves_tags(self):
        mock_session = MagicMock()
        entry = _make_entry(tags={"source": "test"})
        mock_session.query.return_value.filter.return_value.first.return_value = entry
        with (
            patch.object(_srv, "SessionLocal", return_value=mock_session),
            patch.object(_srv, "_ensure_tables"),
        ):
            _srv.memory_flag(str(FAKE_UUID), reason="")
            assert "audit_reason" not in entry.tags


class TestMemoryUpdateTrust:
    """Tests for the memory_update_trust tool."""

    def _run(self, trust_score):
        mock_session = MagicMock()
        entry = _make_entry()
        mock_session.query.return_value.filter.return_value.first.return_value = entry
        with (
            patch.object(_srv, "SessionLocal", return_value=mock_session),
            patch.object(_srv, "_ensure_tables"),
        ):
            _srv.memory_update_trust(str(FAKE_UUID), trust_score=trust_score)
        return entry

    def test_clamps_score_to_zero(self):
        entry = self._run(-0.5)
        assert entry.trust_score == 0.0

    def test_clamps_score_to_one(self):
        entry = self._run(2.5)
        assert entry.trust_score == 1.0

    def test_accepts_valid_score(self):
        entry = self._run(0.42)
        assert entry.trust_score == pytest.approx(0.42)

    def test_boundary_zero(self):
        entry = self._run(0.0)
        assert entry.trust_score == 0.0

    def test_boundary_one(self):
        entry = self._run(1.0)
        assert entry.trust_score == 1.0

    def test_returns_error_for_nonexistent(self):
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None
        with (
            patch.object(_srv, "SessionLocal", return_value=mock_session),
            patch.object(_srv, "_ensure_tables"),
        ):
            result = _srv.memory_update_trust(str(FAKE_UUID), trust_score=0.5)
            assert "error" in result


class TestMemoryRecallRecent:
    """Tests for limit clamping in memory_recall_recent."""

    def _run(self, limit):
        mock_session = MagicMock()
        q = mock_session.query.return_value
        q.filter.return_value = q
        q.order_by.return_value = q
        q.limit.return_value = q
        q.all.return_value = []
        with (
            patch.object(_srv, "SessionLocal", return_value=mock_session),
            patch.object(_srv, "_ensure_tables"),
        ):
            _srv.memory_recall_recent(limit=limit)
        return q

    def test_clamps_limit_low(self):
        q = self._run(0)
        q.limit.assert_called_with(1)

    def test_clamps_limit_high(self):
        q = self._run(500)
        q.limit.assert_called_with(100)

    def test_accepts_valid_limit(self):
        q = self._run(25)
        q.limit.assert_called_with(25)


class TestMemorySearch:
    """Tests for limit clamping and embed call in memory_search."""

    def _run(self, query="test", limit=5):
        mock_session = MagicMock()
        q = mock_session.query.return_value
        q.filter.return_value = q
        q.order_by.return_value = q
        q.limit.return_value = q
        q.all.return_value = []
        mock_embed = MagicMock(return_value=[0.0] * 384)
        with (
            patch.object(_srv, "SessionLocal", return_value=mock_session),
            patch.object(_srv, "_embed", mock_embed),
            patch.object(_srv, "_ensure_tables"),
        ):
            _srv.memory_search(query, limit=limit)
        return q, mock_embed

    def test_clamps_limit_low(self):
        q, _ = self._run(limit=-10)
        q.limit.assert_called_with(1)

    def test_clamps_limit_high(self):
        q, _ = self._run(limit=200)
        q.limit.assert_called_with(50)

    def test_calls_embed_with_query(self):
        _, mock_embed = self._run(query="find this memory")
        mock_embed.assert_called_once_with("find this memory")
