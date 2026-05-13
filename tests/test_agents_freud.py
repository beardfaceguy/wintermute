"""Tests for agents/freud.py — pure-logic audit checks and data structures.

Only tests functions that don't require a database connection.
Uses mock MemoryEntry objects with the required attributes.
"""

import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import with mocked-out heavy dependencies that need a DB / MCP server.
# We temporarily inject mocks into sys.modules so agents/freud.py can be imported,
# then immediately restore originals so other test files get the real modules.
_mcp_keys = [
    "mcp_memory", "mcp_memory.server",
    "mcp_memory.app", "mcp_memory.app.db",
    "mcp_memory.app.db.session", "mcp_memory.app.models",
    "mcp_memory.app.models.memory_entry",
]
_saved = {}
for _k in _mcp_keys:
    _saved[_k] = sys.modules.pop(_k, None)

_mock_mcp_memory = MagicMock()
for _k in _mcp_keys:
    sys.modules[_k] = _mock_mcp_memory if ("server" in _k or _k == "mcp_memory") else MagicMock()

import agents.freud as _freud_mod
from agents.freud import (
    AuditFinding,
    AuditReport,
    FreudAuditor,
    _cosine_sim,
    _is_pair_owner,
    check_contradictions,
    check_contradictions_ann,
    check_low_quality,
    check_near_duplicates,
    check_near_duplicates_ann,
    check_stale_entries,
    DUPLICATE_SIMILARITY_THRESHOLD,
    STALE_DAYS,
    TRUST_BOOST_CLEAN,
    TRUST_PENALTY_FLAGGED,
    AUTO_PROMOTE_TRUST_THRESHOLD,
)

# Restore original sys.modules so other test files get the real mcp_memory modules.
for _k in _mcp_keys:
    if _saved[_k] is not None:
        sys.modules[_k] = _saved[_k]
    else:
        sys.modules.pop(_k, None)
del _saved, _mcp_keys, _mock_mcp_memory


# ── Mock MemoryEntry ─────────────────────────────────────────────────────────


@dataclass
class FakeMemoryEntry:
    """Lightweight stand-in for mcp_memory MemoryEntry ORM model."""
    id: str = "entry-001"
    text: str = "This is a normal memory entry with enough words."
    embedding: list[float] | None = None
    zone: str = "live"
    trust_score: float = 0.5
    audit_flagged: bool = False
    created_at: datetime | None = None
    tags: list[str] = field(default_factory=list)


def _make_entry(id="e1", text="Normal text with enough words here.", **kwargs):
    return FakeMemoryEntry(id=id, text=text, **kwargs)


# ── _cosine_sim ──────────────────────────────────────────────────────────────


def test_cosine_sim_parallel_vectors():
    """Identical direction vectors should have similarity 1.0."""
    a = [1.0, 0.0, 0.0]
    b = [2.0, 0.0, 0.0]
    assert math.isclose(_cosine_sim(a, b), 1.0, abs_tol=1e-9)


def test_cosine_sim_orthogonal_vectors():
    """Perpendicular vectors should have similarity 0.0."""
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert math.isclose(_cosine_sim(a, b), 0.0, abs_tol=1e-9)


def test_cosine_sim_anti_parallel_vectors():
    """Opposite direction vectors should have similarity -1.0."""
    a = [1.0, 0.0, 0.0]
    b = [-1.0, 0.0, 0.0]
    assert math.isclose(_cosine_sim(a, b), -1.0, abs_tol=1e-9)


def test_cosine_sim_zero_vector():
    """Zero vector should return 0.0 (avoid division by zero)."""
    a = [0.0, 0.0, 0.0]
    b = [1.0, 2.0, 3.0]
    assert _cosine_sim(a, b) == 0.0
    assert _cosine_sim(b, a) == 0.0


def test_cosine_sim_both_zero():
    """Two zero vectors should return 0.0."""
    assert _cosine_sim([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_cosine_sim_known_angle():
    """45-degree vectors should give cos(45°) ≈ 0.7071."""
    a = [1.0, 0.0]
    b = [1.0, 1.0]
    expected = 1.0 / math.sqrt(2)
    assert math.isclose(_cosine_sim(a, b), expected, abs_tol=1e-6)


def test_cosine_sim_mismatched_lengths_returns_zero():
    """Vectors of different lengths should return 0.0, not silently truncate."""
    a = [1.0, 0.0, 1.0]  # 3D
    b = [1.0, 0.0]        # 2D
    assert _cosine_sim(a, b) == 0.0


# ── AuditFinding ─────────────────────────────────────────────────────────────


def test_audit_finding_construction():
    """AuditFinding should store all fields including optional related_entry_id."""
    f = AuditFinding(
        entry_id="abc",
        check="low_quality",
        severity="warning",
        detail="Too short",
    )
    assert f.entry_id == "abc"
    assert f.check == "low_quality"
    assert f.severity == "warning"
    assert f.detail == "Too short"
    assert f.related_entry_id is None


def test_audit_finding_with_related():
    """AuditFinding should accept a related_entry_id."""
    f = AuditFinding(
        entry_id="a",
        check="near_duplicate",
        severity="warning",
        detail="sim=0.95",
        related_entry_id="b",
    )
    assert f.related_entry_id == "b"


# ── AuditReport ──────────────────────────────────────────────────────────────


def test_audit_report_defaults():
    """AuditReport should have sensible zero/empty defaults."""
    r = AuditReport()
    assert r.entries_scanned == 0
    assert r.findings == []
    assert r.actions_taken == []
    assert r.entries_flagged == 0
    assert r.entries_trust_updated == 0
    assert r.entries_promoted == 0


def test_audit_report_finding_counts():
    """finding_counts should group findings by check name."""
    r = AuditReport(findings=[
        AuditFinding("a", "low_quality", "warning", "short"),
        AuditFinding("b", "low_quality", "warning", "short"),
        AuditFinding("c", "stale", "info", "old"),
        AuditFinding("d", "contradiction", "critical", "conflict"),
    ])
    counts = r.finding_counts
    assert counts == {"low_quality": 2, "stale": 1, "contradiction": 1}


def test_audit_report_finding_counts_empty():
    """finding_counts on an empty report should return empty dict."""
    r = AuditReport()
    assert r.finding_counts == {}


def test_audit_report_to_dict():
    """to_dict() should serialize the full report to a plain dict."""
    f = AuditFinding("x", "stale", "info", "14 days old", related_entry_id="y")
    r = AuditReport(
        started_at="2026-01-01T00:00:00",
        finished_at="2026-01-01T00:01:00",
        entries_scanned=10,
        findings=[f],
        actions_taken=[{"action": "flag", "entry_id": "x"}],
        entries_flagged=1,
    )
    d = r.to_dict()

    assert d["started_at"] == "2026-01-01T00:00:00"
    assert d["entries_scanned"] == 10
    assert d["total_findings"] == 1
    assert d["finding_counts"] == {"stale": 1}
    assert d["entries_flagged"] == 1
    assert len(d["findings"]) == 1
    assert d["findings"][0]["entry_id"] == "x"
    assert d["findings"][0]["related_entry_id"] == "y"
    assert len(d["actions_taken"]) == 1


def test_audit_report_to_dict_roundtrips_json():
    """to_dict() output should be JSON-serializable."""
    import json
    r = AuditReport(
        started_at="t0", finished_at="t1", entries_scanned=5,
        findings=[AuditFinding("a", "low_quality", "warning", "x")],
    )
    serialized = json.dumps(r.to_dict())
    assert isinstance(json.loads(serialized), dict)


# ── check_low_quality ────────────────────────────────────────────────────────


def test_check_low_quality_short_text():
    """Entries shorter than MIN_TEXT_LENGTH should be flagged."""
    entry = _make_entry(id="short", text="hi")
    findings = check_low_quality([entry])
    assert len(findings) == 1
    assert findings[0].check == "low_quality"
    assert findings[0].entry_id == "short"
    assert "too short" in findings[0].detail.lower()


def test_check_low_quality_empty_text():
    """Empty text should be flagged as low quality."""
    entry = _make_entry(id="empty", text="")
    findings = check_low_quality([entry])
    assert len(findings) == 1
    assert findings[0].entry_id == "empty"


def test_check_low_quality_none_text():
    """None text (treated as '') should be flagged."""
    entry = _make_entry(id="none", text=None)
    # text attribute is None; the function does `(e.text or "").strip()`
    entry.text = None
    findings = check_low_quality([entry])
    assert len(findings) == 1


def test_check_low_quality_few_word_boundaries():
    """Text with < 2 spaces (but >= MIN_TEXT_LENGTH) should be flagged."""
    entry = _make_entry(id="nowords", text="a" * 25 + " " + "b" * 5)
    # 1 space → count(" ") < 2 → flagged
    findings = check_low_quality([entry])
    assert len(findings) == 1
    assert "word boundaries" in findings[0].detail.lower()


def test_check_low_quality_normal_text_passes():
    """Normal text with enough length and word boundaries should pass."""
    entry = _make_entry(id="ok", text="This is a perfectly normal memory entry.")
    findings = check_low_quality([entry])
    assert findings == []


def test_check_low_quality_multiple_entries():
    """Should correctly audit a mix of good and bad entries."""
    entries = [
        _make_entry(id="bad1", text="hi"),
        _make_entry(id="good", text="This is a valid entry with many words."),
        _make_entry(id="bad2", text=""),
    ]
    findings = check_low_quality(entries)
    flagged_ids = {f.entry_id for f in findings}
    assert "bad1" in flagged_ids
    assert "bad2" in flagged_ids
    assert "good" not in flagged_ids


# ── check_near_duplicates ───────────────────────────────────────────────────


def test_check_near_duplicates_identical_embeddings():
    """Identical embeddings should be flagged as near-duplicates."""
    vec = [1.0, 0.0, 0.0]
    entries = [
        _make_entry(id="dup1", embedding=vec),
        _make_entry(id="dup2", embedding=vec),
    ]
    findings = check_near_duplicates(entries)
    assert len(findings) == 1
    assert findings[0].check == "near_duplicate"
    assert findings[0].related_entry_id == "dup2"


def test_check_near_duplicates_orthogonal_no_flag():
    """Orthogonal embeddings should not be flagged."""
    entries = [
        _make_entry(id="a", embedding=[1.0, 0.0, 0.0]),
        _make_entry(id="b", embedding=[0.0, 1.0, 0.0]),
    ]
    findings = check_near_duplicates(entries)
    assert findings == []


def test_check_near_duplicates_skips_none_embedding():
    """Entries with no embedding should be silently skipped."""
    entries = [
        _make_entry(id="a", embedding=[1.0, 0.0]),
        _make_entry(id="b", embedding=None),
    ]
    findings = check_near_duplicates(entries)
    assert findings == []


def test_check_near_duplicates_just_below_threshold():
    """Similarity just below threshold should not trigger a finding."""
    # Build two vectors with similarity slightly below 0.92
    a = [1.0, 0.0, 0.0]
    b = [0.92, math.sqrt(1 - 0.92**2), 0.0]  # cos(a,b) = 0.92 exactly
    sim = _cosine_sim(a, b)
    # We need strictly below, so nudge b
    b = [0.91, math.sqrt(1 - 0.91**2), 0.0]
    sim = _cosine_sim(a, b)
    assert sim < DUPLICATE_SIMILARITY_THRESHOLD

    entries = [
        _make_entry(id="x", embedding=a),
        _make_entry(id="y", embedding=b),
    ]
    findings = check_near_duplicates(entries)
    assert findings == []


# ── check_contradictions ─────────────────────────────────────────────────────


def _similar_pair(sim_target=0.75):
    """Create two embeddings with a specific cosine similarity in the contradiction range."""
    # cos(θ) = sim_target → θ = acos(sim_target)
    # a = [1, 0], b = [cos(θ), sin(θ)]
    theta = math.acos(sim_target)
    a = [math.cos(0), math.sin(0)]
    b = [math.cos(theta), math.sin(theta)]
    return a, b


def test_check_contradictions_detects_negation_mismatch():
    """Entry A without negation + Entry B with negation should flag contradiction."""
    a_emb, b_emb = _similar_pair(0.75)
    entries = [
        _make_entry(id="pos", text="The system is working correctly", embedding=a_emb),
        _make_entry(id="neg", text="The system is not working correctly", embedding=b_emb),
    ]
    findings = check_contradictions(entries)
    assert len(findings) == 1
    assert findings[0].check == "contradiction"
    assert findings[0].severity == "critical"


def test_check_contradictions_no_flag_when_both_have_negation():
    """If both entries contain negations, no contradiction is flagged."""
    a_emb, b_emb = _similar_pair(0.75)
    entries = [
        _make_entry(id="a", text="System can't handle the never case", embedding=a_emb),
        _make_entry(id="b", text="The system won't process incorrect data", embedding=b_emb),
    ]
    findings = check_contradictions(entries)
    assert findings == []


def test_check_contradictions_no_flag_when_neither_has_negation():
    """If neither entry has negation markers, no contradiction is flagged."""
    a_emb, b_emb = _similar_pair(0.75)
    entries = [
        _make_entry(id="a", text="The system handles all cases", embedding=a_emb),
        _make_entry(id="b", text="It processes every request", embedding=b_emb),
    ]
    findings = check_contradictions(entries)
    assert findings == []


def test_check_contradictions_outside_similarity_range_no_flag():
    """Entries with similarity outside [0.65, 0.90] should not be checked."""
    # Identical vectors: sim=1.0 (above range)
    entries = [
        _make_entry(id="a", text="The system works", embedding=[1.0, 0.0]),
        _make_entry(id="b", text="The system does not work", embedding=[1.0, 0.0]),
    ]
    findings = check_contradictions(entries)
    assert findings == []


def test_check_contradictions_skips_none_embedding():
    """Entries without embeddings should be silently skipped."""
    entries = [
        _make_entry(id="a", text="Works fine", embedding=None),
        _make_entry(id="b", text="Does not work", embedding=[1.0, 0.0]),
    ]
    findings = check_contradictions(entries)
    assert findings == []


# ── check_stale_entries ──────────────────────────────────────────────────────


def test_check_stale_entries_old_live_entry():
    """Live entries older than STALE_DAYS should be flagged."""
    old_date = datetime.now(timezone.utc) - timedelta(days=STALE_DAYS + 1)
    entry = _make_entry(id="old", zone="live", created_at=old_date)
    findings = check_stale_entries([entry])
    assert len(findings) == 1
    assert findings[0].check == "stale"
    assert findings[0].severity == "info"


def test_check_stale_entries_recent_live_entry():
    """Live entries younger than STALE_DAYS should not be flagged."""
    recent = datetime.now(timezone.utc) - timedelta(days=1)
    entry = _make_entry(id="fresh", zone="live", created_at=recent)
    findings = check_stale_entries([entry])
    assert findings == []


def test_check_stale_entries_exact_threshold():
    """Entry exactly at STALE_DAYS boundary should be flagged (>= threshold)."""
    boundary = datetime.now(timezone.utc) - timedelta(days=STALE_DAYS)
    entry = _make_entry(id="boundary", zone="live", created_at=boundary)
    findings = check_stale_entries([entry])
    assert len(findings) == 1


def test_check_stale_entries_non_live_zone_ignored():
    """Entries not in the 'live' zone should never be flagged as stale."""
    old_date = datetime.now(timezone.utc) - timedelta(days=STALE_DAYS + 100)
    entry = _make_entry(id="cold", zone="cold", created_at=old_date)
    findings = check_stale_entries([entry])
    assert findings == []


def test_check_stale_entries_none_created_at_skipped():
    """Entries without created_at should be silently skipped."""
    entry = _make_entry(id="nodate", zone="live", created_at=None)
    findings = check_stale_entries([entry])
    assert findings == []


def test_check_stale_entries_naive_datetime_handled():
    """Naive datetime (no tzinfo) should be treated as UTC."""
    old_naive = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=STALE_DAYS + 5)
    entry = _make_entry(id="naive", zone="live", created_at=old_naive)
    findings = check_stale_entries([entry])
    assert len(findings) == 1


# ── FreudAuditor.__init__ ───────────────────────────────────────────────────


def test_freud_auditor_init_defaults():
    """Default constructor should set dry_run=False, zone='live', etc."""
    auditor = FreudAuditor()
    assert auditor.dry_run is False
    assert auditor.zone == "live"
    assert auditor.flagged_only is False
    assert auditor.promote_ready is False


def test_freud_auditor_init_custom():
    """Custom args should be stored correctly."""
    auditor = FreudAuditor(
        dry_run=True,
        zone="cold",
        flagged_only=True,
        promote_ready=True,
    )
    assert auditor.dry_run is True
    assert auditor.zone == "cold"
    assert auditor.flagged_only is True
    assert auditor.promote_ready is True


def test_freud_auditor_init_zone_none():
    """zone=None should be allowed (audit all zones)."""
    auditor = FreudAuditor(zone=None)
    assert auditor.zone is None


# ── Batch resilience (CLA-261 pattern) ───────────────────────────────────────


def test_apply_flags_continues_past_failed_entry():
    """_apply_flags should not abort on a memory_flag exception — it should skip and continue."""
    auditor = FreudAuditor(dry_run=False)
    report = AuditReport(
        findings=[
            AuditFinding(entry_id="aaa", check="low_quality", severity="warning", detail="too short"),
            AuditFinding(entry_id="bbb", check="low_quality", severity="warning", detail="too short"),
            AuditFinding(entry_id="ccc", check="low_quality", severity="warning", detail="too short"),
        ]
    )

    call_count = {"n": 0}

    def mock_flag(entry_id, reason=""):
        call_count["n"] += 1
        if entry_id == "bbb":
            raise ConnectionError("DB connection lost")
        return {"status": "flagged"}

    with patch.object(_freud_mod, "memory_flag", side_effect=mock_flag):
        flagged = auditor._apply_flags(report)

    assert "aaa" in flagged, "Entry before the error should have been flagged"
    assert "ccc" in flagged, "Entry after the error should have been flagged"
    assert call_count["n"] == 3, "All three entries should have been attempted"


def test_calibrate_trust_continues_past_failed_entry():
    """_calibrate_trust should not abort when memory_update_trust raises for one entry."""
    auditor = FreudAuditor(dry_run=False)
    report = AuditReport()

    entries = [
        _make_entry(id="aaa", trust_score=0.3),
        _make_entry(id="bbb", trust_score=0.3),
        _make_entry(id="ccc", trust_score=0.3),
    ]

    call_count = {"n": 0}

    def mock_update_trust(entry_id, trust_score):
        call_count["n"] += 1
        if entry_id == "bbb":
            raise ConnectionError("DB connection lost")
        return {"status": "updated"}

    with patch.object(_freud_mod, "memory_update_trust", side_effect=mock_update_trust):
        auditor._calibrate_trust(entries, flagged_ids=set(), report=report)

    assert call_count["n"] == 3, "All three entries should have been attempted"
    assert report.entries_trust_updated == 2, "Two successful updates should be counted"


def test_auto_promote_continues_past_failed_entry():
    """_auto_promote should not abort when memory_promote raises for one entry."""
    auditor = FreudAuditor(dry_run=False, promote_ready=True)
    report = AuditReport()

    entries = [
        _make_entry(id="aaa", trust_score=0.9, zone="live"),
        _make_entry(id="bbb", trust_score=0.9, zone="live"),
        _make_entry(id="ccc", trust_score=0.9, zone="live"),
    ]

    call_count = {"n": 0}

    def mock_promote(entry_id, trust_score):
        call_count["n"] += 1
        if entry_id == "bbb":
            raise ConnectionError("DB connection lost")
        return {"status": "promoted"}

    with patch.object(_freud_mod, "memory_promote", side_effect=mock_promote):
        auditor._auto_promote(entries, flagged_ids=set(), report=report)

    assert call_count["n"] == 3, "All three entries should have been attempted"
    assert report.entries_promoted == 2, "Two successful promotions should be counted"


# ── _is_pair_owner ───────────────────────────────────────────────────────────


def test_pair_owner_older_created_at_wins():
    a = _make_entry(id="aaa", created_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    # neighbor newer → entry A owns the pair
    assert _is_pair_owner(a, datetime(2025, 6, 1, tzinfo=timezone.utc), "bbb") is True
    # neighbor older → entry A does NOT own the pair
    assert _is_pair_owner(a, datetime(2024, 1, 1, tzinfo=timezone.utc), "bbb") is False


def test_pair_owner_tiebreak_by_id_when_created_at_equal():
    same = datetime(2025, 1, 1, tzinfo=timezone.utc)
    a = _make_entry(id="aaa", created_at=same)
    assert _is_pair_owner(a, same, "bbb") is True   # "aaa" < "bbb"
    assert _is_pair_owner(a, same, "000") is False  # "aaa" > "000"


def test_pair_owner_with_missing_created_at_falls_back_to_id():
    a = _make_entry(id="aaa", created_at=None)
    assert _is_pair_owner(a, None, "bbb") is True
    assert _is_pair_owner(a, None, "000") is False


# ── check_near_duplicates_ann (DB-backed) ────────────────────────────────────


def _mock_query_chain(rows):
    """Build a MagicMock chain emulating SQLAlchemy's fluent query API.

    Every fluent call (.filter / .order_by / .limit) returns the same chain,
    and .all() returns the supplied rows. Adequate for the freud ANN path.
    """
    chain = MagicMock()
    chain.filter.return_value = chain
    chain.order_by.return_value = chain
    chain.limit.return_value = chain
    chain.all.return_value = rows
    return chain


class _MockColumn:
    """Lightweight stand-in for a SQLAlchemy InstrumentedAttribute.

    Supports the operators freud's queries need (comparison, isnot, and
    pgvector's cosine_distance) and returns MagicMock so the resulting
    expressions are accepted by the mock query chain without evaluation.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other): return MagicMock(name=f"{self.name}==")
    def __ne__(self, other): return MagicMock(name=f"{self.name}!=")
    def __gt__(self, other): return MagicMock(name=f"{self.name}>")
    def __ge__(self, other): return MagicMock(name=f"{self.name}>=")
    def __lt__(self, other): return MagicMock(name=f"{self.name}<")
    def __le__(self, other): return MagicMock(name=f"{self.name}<=")
    def __hash__(self): return id(self)

    def isnot(self, _other): return MagicMock(name=f"{self.name}.isnot")
    def asc(self): return MagicMock(name=f"{self.name}.asc")
    def desc(self): return MagicMock(name=f"{self.name}.desc")

    def cosine_distance(self, _vec):
        return _MockExpr(f"{self.name}.cosine_distance")


class _MockExpr:
    """Expression-like stand-in for ``column.cosine_distance(vec).label('d')``.

    Comparison operators return a MagicMock so SQLAlchemy-style filters like
    ``expr <= max_dist`` work in tests against a mock query chain.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def label(self, _alias: str) -> "_MockExpr":
        return self

    def __le__(self, _other): return MagicMock(name=f"{self.name}<=")
    def __ge__(self, _other): return MagicMock(name=f"{self.name}>=")
    def __lt__(self, _other): return MagicMock(name=f"{self.name}<")
    def __gt__(self, _other): return MagicMock(name=f"{self.name}>")
    def __eq__(self, _other): return MagicMock(name=f"{self.name}==")
    def __ne__(self, _other): return MagicMock(name=f"{self.name}!=")
    def __hash__(self): return id(self)


class _MockEntry:
    """Stand-in for the MemoryEntry ORM class used during streaming/ANN tests."""
    id = _MockColumn("id")
    text = _MockColumn("text")
    embedding = _MockColumn("embedding")
    created_at = _MockColumn("created_at")
    zone = _MockColumn("zone")
    audit_flagged = _MockColumn("audit_flagged")


def _patch_sqla(*extras):
    """Patch out SQLAlchemy expression builders + the MemoryEntry class so that
    the freud streaming code can run against MagicMock query chains. ``extras``
    are additional patches to layer on top.
    """
    return [
        patch.object(_freud_mod, "MemoryEntry", _MockEntry),
        patch.object(_freud_mod, "or_", new=lambda *a, **kw: MagicMock(name="or_")),
        patch.object(_freud_mod, "and_", new=lambda *a, **kw: MagicMock(name="and_")),
        *extras,
    ]


from contextlib import ExitStack


def _enter_all(stack: ExitStack, ctxs):
    return [stack.enter_context(c) for c in ctxs]


def test_check_near_duplicates_ann_returns_no_findings_when_no_embedding():
    db = MagicMock()
    entry = _make_entry(id="a", embedding=None)
    findings = check_near_duplicates_ann(db, entry, k=10, threshold=0.9)
    assert findings == []
    db.query.assert_not_called()


def test_check_near_duplicates_ann_emits_finding_when_owner_of_pair():
    """Entry with older created_at owns the pair → finding is emitted."""
    db = MagicMock()
    db.query.return_value = _mock_query_chain([
        # neighbor row: (id, text, created_at, distance)
        ("nbr-1", "duplicate text", datetime(2025, 6, 1, tzinfo=timezone.utc), 0.05),
    ])
    entry = _make_entry(
        id="aaa",
        embedding=[0.1, 0.2, 0.3],
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    with ExitStack() as stack:
        _enter_all(stack, _patch_sqla())
        findings = check_near_duplicates_ann(db, entry, k=10, threshold=0.9)
    assert len(findings) == 1
    assert findings[0].check == "near_duplicate"
    assert findings[0].entry_id == "aaa"
    assert findings[0].related_entry_id == "nbr-1"
    # similarity = 1 - distance = 0.95
    assert "0.9500" in findings[0].detail


def test_check_near_duplicates_ann_skips_when_neighbor_is_older():
    """When the neighbor is older, ownership belongs to the neighbor → no finding."""
    db = MagicMock()
    db.query.return_value = _mock_query_chain([
        ("nbr-1", "older dup", datetime(2024, 6, 1, tzinfo=timezone.utc), 0.05),
    ])
    entry = _make_entry(
        id="aaa",
        embedding=[0.1, 0.2, 0.3],
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    with ExitStack() as stack:
        _enter_all(stack, _patch_sqla())
        findings = check_near_duplicates_ann(db, entry, k=10, threshold=0.9)
    assert findings == []


def test_check_near_duplicates_ann_pushes_distance_filter_to_db():
    """The query chain must apply at least one distance filter (≤ 1 - threshold)."""
    db = MagicMock()
    db.query.return_value = _mock_query_chain([])
    entry = _make_entry(id="a", embedding=[1.0, 0.0])
    with ExitStack() as stack:
        _enter_all(stack, _patch_sqla())
        check_near_duplicates_ann(db, entry, k=10, threshold=0.92)

    chain = db.query.return_value
    # _neighbors_query applies 2 filters (id, embedding-not-null) then the
    # caller adds 1 more (distance ≤ max). Total ≥ 3 filter calls.
    assert chain.filter.call_count >= 3
    chain.limit.assert_called_with(10)
    chain.order_by.assert_called_once()


# ── check_contradictions_ann (DB-backed) ─────────────────────────────────────


def test_check_contradictions_ann_no_embedding_returns_empty():
    db = MagicMock()
    entry = _make_entry(id="a", text="The system works", embedding=None)
    assert check_contradictions_ann(db, entry, k=10) == []
    db.query.assert_not_called()


def test_check_contradictions_ann_emits_finding_on_negation_mismatch():
    db = MagicMock()
    db.query.return_value = _mock_query_chain([
        ("nbr-1", "The system is not working", datetime(2025, 6, 1, tzinfo=timezone.utc), 0.25),
    ])
    entry = _make_entry(
        id="aaa",
        text="The system is working correctly",
        embedding=[1.0, 0.0],
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    with ExitStack() as stack:
        _enter_all(stack, _patch_sqla())
        findings = check_contradictions_ann(db, entry, k=10)
    assert len(findings) == 1
    assert findings[0].check == "contradiction"
    assert findings[0].severity == "critical"
    assert findings[0].related_entry_id == "nbr-1"


def test_check_contradictions_ann_no_finding_when_both_have_negation():
    db = MagicMock()
    db.query.return_value = _mock_query_chain([
        ("nbr-1", "It does not work either", datetime(2025, 6, 1, tzinfo=timezone.utc), 0.25),
    ])
    entry = _make_entry(
        id="aaa",
        text="The system never works",
        embedding=[1.0, 0.0],
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    with ExitStack() as stack:
        _enter_all(stack, _patch_sqla())
        findings = check_contradictions_ann(db, entry, k=10)
    assert findings == []


def test_check_contradictions_ann_applies_pair_ownership():
    """Older entry owns the pair, but here entry is younger → no finding."""
    db = MagicMock()
    db.query.return_value = _mock_query_chain([
        ("nbr-1", "It does not work", datetime(2024, 6, 1, tzinfo=timezone.utc), 0.25),
    ])
    entry = _make_entry(
        id="aaa",
        text="The system works",
        embedding=[1.0, 0.0],
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    with ExitStack() as stack:
        _enter_all(stack, _patch_sqla())
        findings = check_contradictions_ann(db, entry, k=10)
    assert findings == []


# ── FreudAuditor._iter_batches (keyset pagination) ───────────────────────────


def test_iter_batches_walks_all_pages():
    """Iterator should keep paging until a short batch arrives."""
    auditor = FreudAuditor(zone="live", batch_size=2)

    # Two full pages, then a half page → stop.
    page1 = [
        _make_entry(id="e1", created_at=datetime(2025, 1, 1, tzinfo=timezone.utc)),
        _make_entry(id="e2", created_at=datetime(2025, 1, 2, tzinfo=timezone.utc)),
    ]
    page2 = [
        _make_entry(id="e3", created_at=datetime(2025, 1, 3, tzinfo=timezone.utc)),
        _make_entry(id="e4", created_at=datetime(2025, 1, 4, tzinfo=timezone.utc)),
    ]
    page3 = [
        _make_entry(id="e5", created_at=datetime(2025, 1, 5, tzinfo=timezone.utc)),
    ]
    pages = [page1, page2, page3]

    db = MagicMock()
    chains = [_mock_query_chain(p) for p in pages]
    db.query.side_effect = chains

    with ExitStack() as stack:
        _enter_all(stack, _patch_sqla())
        batches = list(auditor._iter_batches(db))

    assert [len(b) for b in batches] == [2, 2, 1]
    flat = [e.id for b in batches for e in b]
    assert flat == ["e1", "e2", "e3", "e4", "e5"]


def test_iter_batches_respects_max_entries_cap():
    """When max_entries < total, iteration stops early and trims the final batch."""
    auditor = FreudAuditor(zone="live", batch_size=10, max_entries=3)
    db = MagicMock()
    db.query.return_value = _mock_query_chain([
        _make_entry(id=f"e{i}", created_at=datetime(2025, 1, i + 1, tzinfo=timezone.utc))
        for i in range(10)
    ])
    with ExitStack() as stack:
        _enter_all(stack, _patch_sqla())
        batches = list(auditor._iter_batches(db))
    flat = [e.id for b in batches for e in b]
    assert flat == ["e0", "e1", "e2"]


def test_iter_batches_stops_on_empty_first_page():
    auditor = FreudAuditor(zone="live", batch_size=10)
    db = MagicMock()
    db.query.return_value = _mock_query_chain([])
    with ExitStack() as stack:
        _enter_all(stack, _patch_sqla())
        result = list(auditor._iter_batches(db))
    assert result == []


# ── FreudAuditor.run (streaming integration with mocks) ──────────────────────


def _fresh(now=None):
    """A created_at recent enough to never trip the staleness check."""
    return now or datetime.now(timezone.utc)


def test_run_streams_entries_and_calibrates_trust():
    """run() should stream entries, run checks, flag, and adjust trust per-entry."""
    auditor = FreudAuditor(dry_run=False, zone="live", batch_size=5)

    # Two clean entries → no findings → trust gets boosted.
    entries = [
        _make_entry(
            id="aaa", text="A perfectly normal entry with enough length.",
            embedding=[0.1, 0.2], trust_score=0.5, zone="live",
            created_at=_fresh(),
        ),
        _make_entry(
            id="bbb", text="Another perfectly normal entry that is long enough.",
            embedding=[0.3, 0.4], trust_score=0.5, zone="live",
            created_at=_fresh(),
        ),
    ]

    trust_calls: list[tuple[str, float]] = []

    def fake_update_trust(entry_id, trust_score):
        trust_calls.append((entry_id, trust_score))
        return {"status": "updated"}

    db = MagicMock()
    db.query.side_effect = [
        _mock_query_chain(entries),
        _mock_query_chain([]),
    ]

    with ExitStack() as stack:
        _enter_all(stack, _patch_sqla(
            patch.object(_freud_mod, "_ensure_tables"),
            patch.object(_freud_mod, "SessionLocal", return_value=db),
            patch.object(_freud_mod, "check_near_duplicates_ann", return_value=[]),
            patch.object(_freud_mod, "check_contradictions_ann", return_value=[]),
            patch.object(_freud_mod, "memory_flag"),
            patch.object(
                _freud_mod, "memory_update_trust", side_effect=fake_update_trust,
            ),
        ))
        report = auditor.run()
        mock_flag = _freud_mod.memory_flag

    assert report.entries_scanned == 2
    assert report.findings == []
    assert report.entries_flagged == 0
    # Both entries had findings=[] → trust got boosted by TRUST_BOOST_CLEAN.
    assert len(trust_calls) == 2
    assert {c[0] for c in trust_calls} == {"aaa", "bbb"}
    expected_trust = round(0.5 + TRUST_BOOST_CLEAN, 3)
    assert all(c[1] == expected_trust for c in trust_calls)
    mock_flag.assert_not_called()


def test_run_dry_run_does_not_take_actions():
    """dry_run=True must skip flag/trust/promote even when findings exist."""
    auditor = FreudAuditor(dry_run=True, zone="live", batch_size=5)

    entries = [
        _make_entry(
            id="bad", text="hi",  # too short → low_quality finding
            embedding=[0.1, 0.2], trust_score=0.5, zone="live",
            created_at=_fresh(),
        ),
    ]

    db = MagicMock()
    db.query.side_effect = [_mock_query_chain(entries), _mock_query_chain([])]

    with ExitStack() as stack:
        _enter_all(stack, _patch_sqla(
            patch.object(_freud_mod, "_ensure_tables"),
            patch.object(_freud_mod, "SessionLocal", return_value=db),
            patch.object(_freud_mod, "check_near_duplicates_ann", return_value=[]),
            patch.object(_freud_mod, "check_contradictions_ann", return_value=[]),
            patch.object(_freud_mod, "memory_flag"),
            patch.object(_freud_mod, "memory_update_trust"),
            patch.object(_freud_mod, "memory_promote"),
        ))
        report = auditor.run()
        mock_flag = _freud_mod.memory_flag
        mock_trust = _freud_mod.memory_update_trust
        mock_promote = _freud_mod.memory_promote

    assert report.entries_scanned == 1
    assert len(report.findings) == 1
    assert report.findings[0].check == "low_quality"
    mock_flag.assert_not_called()
    mock_trust.assert_not_called()
    mock_promote.assert_not_called()


def test_run_flags_entry_with_warning_finding_and_penalizes_trust():
    """Entries with low_quality findings should be flagged once and lose trust."""
    auditor = FreudAuditor(dry_run=False, zone="live", batch_size=5)

    entries = [
        _make_entry(
            id="bad", text="hi",  # triggers low_quality
            embedding=[0.1, 0.2], trust_score=0.6, zone="live",
            created_at=_fresh(),
        ),
    ]

    db = MagicMock()
    db.query.side_effect = [_mock_query_chain(entries), _mock_query_chain([])]

    with ExitStack() as stack:
        _enter_all(stack, _patch_sqla(
            patch.object(_freud_mod, "_ensure_tables"),
            patch.object(_freud_mod, "SessionLocal", return_value=db),
            patch.object(_freud_mod, "check_near_duplicates_ann", return_value=[]),
            patch.object(_freud_mod, "check_contradictions_ann", return_value=[]),
            patch.object(_freud_mod, "memory_flag", return_value={"status": "flagged"}),
            patch.object(
                _freud_mod, "memory_update_trust", return_value={"status": "updated"},
            ),
        ))
        report = auditor.run()
        mock_flag = _freud_mod.memory_flag
        mock_trust = _freud_mod.memory_update_trust

    assert report.entries_scanned == 1
    assert report.entries_flagged == 1
    mock_flag.assert_called_once()
    assert mock_flag.call_args.kwargs["entry_id"] == "bad"
    # Trust was decreased by TRUST_PENALTY_FLAGGED.
    expected_trust = round(0.6 - TRUST_PENALTY_FLAGGED, 3)
    mock_trust.assert_called_once_with(entry_id="bad", trust_score=expected_trust)


def test_run_promotes_clean_high_trust_entries_when_promote_ready():
    auditor = FreudAuditor(
        dry_run=False, zone="live", promote_ready=True, batch_size=5,
    )
    entries = [
        _make_entry(
            id="clean",
            text="A perfectly clean and detailed observation here.",
            embedding=[0.1, 0.2],
            trust_score=AUTO_PROMOTE_TRUST_THRESHOLD,
            zone="live",
            created_at=_fresh(),
        ),
    ]

    db = MagicMock()
    db.query.side_effect = [_mock_query_chain(entries), _mock_query_chain([])]

    with ExitStack() as stack:
        _enter_all(stack, _patch_sqla(
            patch.object(_freud_mod, "_ensure_tables"),
            patch.object(_freud_mod, "SessionLocal", return_value=db),
            patch.object(_freud_mod, "check_near_duplicates_ann", return_value=[]),
            patch.object(_freud_mod, "check_contradictions_ann", return_value=[]),
            patch.object(_freud_mod, "memory_flag"),
            patch.object(_freud_mod, "memory_update_trust", return_value={"status": "updated"}),
            patch.object(_freud_mod, "memory_promote", return_value={"status": "promoted"}),
        ))
        report = auditor.run()
        mock_flag = _freud_mod.memory_flag
        mock_promote = _freud_mod.memory_promote

    mock_flag.assert_not_called()
    mock_promote.assert_called_once()
    assert report.entries_promoted == 1
