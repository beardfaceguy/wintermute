"""
Freud — Wintermute Sanity Auditor (CLA-142)

Scans the mcp-memory 'live' zone for problematic entries and takes action:
  - Near-duplicate detection via cosine similarity
  - Contradiction detection between semantically similar entries
  - Low-quality filtering (too short, empty, garbled)
  - Stale entry detection (old live entries never promoted)
  - Trust score calibration based on audit findings

Freud operates as a batch process that can be run manually or on a schedule.
It uses mcp-memory functions directly (same pattern as talkingHead/memory/strategic.py).

Usage:
    python agents/freud.py                       # full audit of live zone
    python agents/freud.py --zone live           # audit specific zone
    python agents/freud.py --dry-run             # report only, no modifications
    python agents/freud.py --flagged-only        # re-audit already-flagged entries
    python agents/freud.py --promote-ready       # auto-promote entries passing all checks
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from mcp_memory.server import (
    _embed,
    _ensure_tables,
    memory_flag,
    memory_promote,
    memory_update_trust,
)
from mcp_memory.app.db.session import SessionLocal
from mcp_memory.app.models.memory_entry import MemoryEntry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("freud")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DUPLICATE_SIMILARITY_THRESHOLD = float(os.getenv("FREUD_DUPLICATE_THRESHOLD", "0.92"))
CONTRADICTION_SIMILARITY_RANGE = (
    float(os.getenv("FREUD_CONTRADICTION_MIN", "0.65")),
    float(os.getenv("FREUD_CONTRADICTION_MAX", "0.90")),
)
MIN_TEXT_LENGTH = int(os.getenv("FREUD_MIN_TEXT_LENGTH", "20"))
STALE_DAYS = int(os.getenv("FREUD_STALE_DAYS", "14"))
TRUST_BOOST_CLEAN = float(os.getenv("FREUD_TRUST_BOOST_CLEAN", "0.15"))
TRUST_PENALTY_FLAGGED = float(os.getenv("FREUD_TRUST_PENALTY_FLAGGED", "0.2"))
AUTO_PROMOTE_TRUST_THRESHOLD = float(os.getenv("FREUD_AUTO_PROMOTE_THRESHOLD", "0.8"))

# Streaming / scalability knobs (CLA-161). Keep memory bounded regardless of
# how big the memory store grows by using keyset-paginated streaming and per-
# entry top-K ANN queries instead of all-pairs in Python.
NEIGHBOR_K = int(os.getenv("FREUD_NEIGHBOR_K", "10"))
BATCH_SIZE = int(os.getenv("FREUD_BATCH_SIZE", "500"))
# 0 = unlimited; positive cap stops the audit after this many entries.
MAX_ENTRIES_PER_RUN = int(os.getenv("FREUD_MAX_ENTRIES_PER_RUN", "0"))

NEGATION_MARKERS: frozenset[str] = frozenset({
    "not", "never", "don't", "doesn't", "didn't", "won't", "can't",
    "cannot", "shouldn't", "failed", "incorrect", "wrong", "false",
    "no", "none", "neither", "nor",
})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AuditFinding:
    """A single finding from an audit check."""
    entry_id: str
    check: str
    severity: str  # "info", "warning", "critical"
    detail: str
    related_entry_id: str | None = None


@dataclass
class AuditReport:
    """Aggregate results from a full audit pass."""
    started_at: str = ""
    finished_at: str = ""
    entries_scanned: int = 0
    findings: list[AuditFinding] = field(default_factory=list)
    actions_taken: list[dict[str, Any]] = field(default_factory=list)
    entries_flagged: int = 0
    entries_trust_updated: int = 0
    entries_promoted: int = 0

    @property
    def finding_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.check] = counts.get(f.check, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "entries_scanned": self.entries_scanned,
            "finding_counts": self.finding_counts,
            "total_findings": len(self.findings),
            "entries_flagged": self.entries_flagged,
            "entries_trust_updated": self.entries_trust_updated,
            "entries_promoted": self.entries_promoted,
            "findings": [
                {
                    "entry_id": f.entry_id,
                    "check": f.check,
                    "severity": f.severity,
                    "detail": f.detail,
                    "related_entry_id": f.related_entry_id,
                }
                for f in self.findings
            ],
            "actions_taken": self.actions_taken,
        }


# ---------------------------------------------------------------------------
# Audit checks
# ---------------------------------------------------------------------------


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        logger.warning("Embedding length mismatch (%d vs %d), returning 0.0", len(a), len(b))
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def check_low_quality(entries: list[MemoryEntry]) -> list[AuditFinding]:
    """Flag entries that are too short, empty, or garbled."""
    findings = []
    for e in entries:
        text = (e.text or "").strip()
        if len(text) < MIN_TEXT_LENGTH:
            findings.append(AuditFinding(
                entry_id=str(e.id),
                check="low_quality",
                severity="warning",
                detail=f"Text too short ({len(text)} chars, min {MIN_TEXT_LENGTH}): {text[:80]!r}",
            ))
        elif text.count(" ") < 2:
            findings.append(AuditFinding(
                entry_id=str(e.id),
                check="low_quality",
                severity="warning",
                detail=f"Suspiciously few word boundaries: {text[:80]!r}",
            ))
    return findings


def check_near_duplicates(entries: list[MemoryEntry]) -> list[AuditFinding]:
    """Detect near-duplicate entries via embedding cosine similarity.

    O(n^2) but fine for typical memory sizes (< 10k entries).
    """
    findings = []
    seen_pairs: set[tuple[str, str]] = set()

    for i, a in enumerate(entries):
        if a.embedding is None:
            continue
        for b in entries[i + 1:]:
            if b.embedding is None:
                continue
            pair = tuple(sorted([str(a.id), str(b.id)]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            sim = _cosine_sim(list(a.embedding), list(b.embedding))
            if sim >= DUPLICATE_SIMILARITY_THRESHOLD:
                findings.append(AuditFinding(
                    entry_id=str(a.id),
                    check="near_duplicate",
                    severity="warning",
                    detail=f"Similarity {sim:.4f} with entry {b.id}: {b.text[:60]!r}",
                    related_entry_id=str(b.id),
                ))
    return findings


def check_contradictions(entries: list[MemoryEntry]) -> list[AuditFinding]:
    """Detect potential contradictions: entries that are semantically similar
    but contain opposing signals (negation words, conflicting outcomes).

    Uses a simple heuristic: high similarity + presence of negation markers
    in one but not the other. O(n^2) — kept for tests and small in-memory
    use; production audits go through check_contradictions_ann.
    """
    findings = []

    for i, a in enumerate(entries):
        if a.embedding is None:
            continue
        a_words = set((a.text or "").lower().split())
        a_negations = a_words & NEGATION_MARKERS

        for b in entries[i + 1:]:
            if b.embedding is None:
                continue
            sim = _cosine_sim(list(a.embedding), list(b.embedding))
            lo, hi = CONTRADICTION_SIMILARITY_RANGE
            if lo <= sim <= hi:
                b_words = set((b.text or "").lower().split())
                b_negations = b_words & NEGATION_MARKERS
                if bool(a_negations) != bool(b_negations):
                    findings.append(AuditFinding(
                        entry_id=str(a.id),
                        check="contradiction",
                        severity="critical",
                        detail=(
                            f"Possible contradiction (sim={sim:.4f}). "
                            f"Entry A negations: {a_negations or 'none'}, "
                            f"Entry B negations: {b_negations or 'none'}. "
                            f"B: {b.text[:60]!r}"
                        ),
                        related_entry_id=str(b.id),
                    ))
    return findings


# ---------------------------------------------------------------------------
# Streaming / ANN-backed audit checks (CLA-161)
# ---------------------------------------------------------------------------
#
# Each pair of entries (A, B) is reported exactly once: when the *older* of
# the pair is being processed. This preserves the prior semantics where the
# entry encountered first (in created_at-ascending order) emitted the finding.

def _is_pair_owner(
    entry: MemoryEntry,
    neighbor_created: datetime | None,
    neighbor_id: Any,
) -> bool:
    """Return True iff `entry` should own the pair (entry, neighbor) finding.

    Older entry wins by ``created_at``; ties broken by stringified id so the
    decision is deterministic and symmetric across the two streaming visits.
    """
    a_created = entry.created_at
    if a_created is not None and neighbor_created is not None:
        if a_created < neighbor_created:
            return True
        if a_created > neighbor_created:
            return False
    return str(entry.id) < str(neighbor_id)


def _neighbors_query(db: Session, entry: MemoryEntry, zone: str | None):
    """Build the base SQLAlchemy query yielding (id, text, created_at, distance)
    rows for an entry's nearest neighbors in cosine space. Filters None-embeddings
    and the entry itself but applies no distance bounds — callers add them.
    """
    distance_expr = MemoryEntry.embedding.cosine_distance(entry.embedding).label("distance")
    q = (
        db.query(
            MemoryEntry.id,
            MemoryEntry.text,
            MemoryEntry.created_at,
            distance_expr,
        )
        .filter(MemoryEntry.id != entry.id)
        .filter(MemoryEntry.embedding.isnot(None))
    )
    if zone:
        q = q.filter(MemoryEntry.zone == zone)
    return q, distance_expr


def check_near_duplicates_ann(
    db: Session,
    entry: MemoryEntry,
    *,
    k: int = NEIGHBOR_K,
    threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
    zone: str | None = None,
) -> list[AuditFinding]:
    """Top-K ANN duplicate check for a single entry.

    Pushes cosine_distance to Postgres/pgvector — works in O(log n) with the
    HNSW index on memory_entries.embedding (or sequential scan without it).
    Emits at most one finding per (entry, neighbor) pair, owned by the older
    entry, so streaming over all entries covers each pair exactly once.
    """
    if entry.embedding is None:
        return []

    max_distance = 1.0 - threshold
    q, distance_expr = _neighbors_query(db, entry, zone)
    rows = (
        q.filter(distance_expr <= max_distance)
        .order_by(distance_expr)
        .limit(k)
        .all()
    )

    findings: list[AuditFinding] = []
    for nid, ntext, ncreated, distance in rows:
        if not _is_pair_owner(entry, ncreated, nid):
            continue
        sim = 1.0 - float(distance)
        findings.append(AuditFinding(
            entry_id=str(entry.id),
            check="near_duplicate",
            severity="warning",
            detail=f"Similarity {sim:.4f} with entry {nid}: {(ntext or '')[:60]!r}",
            related_entry_id=str(nid),
        ))
    return findings


def check_contradictions_ann(
    db: Session,
    entry: MemoryEntry,
    *,
    k: int = NEIGHBOR_K,
    sim_range: tuple[float, float] = CONTRADICTION_SIMILARITY_RANGE,
    zone: str | None = None,
) -> list[AuditFinding]:
    """Top-K ANN contradiction check for a single entry.

    Restricts neighbors to similarity in ``sim_range``; in-Python applies the
    negation-marker heuristic. Pair ownership goes to the older entry so each
    candidate pair is evaluated exactly once across a full streaming pass.
    """
    if entry.embedding is None:
        return []

    sim_lo, sim_hi = sim_range
    # similarity ∈ [lo, hi]  ⇔  distance ∈ [1-hi, 1-lo]
    min_distance = 1.0 - sim_hi
    max_distance = 1.0 - sim_lo

    q, distance_expr = _neighbors_query(db, entry, zone)
    rows = (
        q.filter(distance_expr >= min_distance)
        .filter(distance_expr <= max_distance)
        .order_by(distance_expr)
        .limit(k)
        .all()
    )

    a_words = set((entry.text or "").lower().split())
    a_negations = a_words & NEGATION_MARKERS

    findings: list[AuditFinding] = []
    for nid, ntext, ncreated, distance in rows:
        if not _is_pair_owner(entry, ncreated, nid):
            continue
        sim = 1.0 - float(distance)
        b_words = set((ntext or "").lower().split())
        b_negations = b_words & NEGATION_MARKERS
        if bool(a_negations) != bool(b_negations):
            findings.append(AuditFinding(
                entry_id=str(entry.id),
                check="contradiction",
                severity="critical",
                detail=(
                    f"Possible contradiction (sim={sim:.4f}). "
                    f"Entry A negations: {a_negations or 'none'}, "
                    f"Entry B negations: {b_negations or 'none'}. "
                    f"B: {(ntext or '')[:60]!r}"
                ),
                related_entry_id=str(nid),
            ))
    return findings


def check_stale_entries(entries: list[MemoryEntry]) -> list[AuditFinding]:
    """Flag live entries that have sat unreviewed beyond the staleness window."""
    findings = []
    now = datetime.now(timezone.utc)
    for e in entries:
        if e.zone != "live":
            continue
        if e.created_at is None:
            continue
        created = e.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = (now - created).days
        if age_days >= STALE_DAYS:
            findings.append(AuditFinding(
                entry_id=str(e.id),
                check="stale",
                severity="info",
                detail=f"Live entry is {age_days} days old (threshold: {STALE_DAYS})",
            ))
    return findings


# ---------------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------------


class FreudAuditor:
    """Batch auditor for mcp-memory entries.

    Operates as a single streaming pass via keyset pagination on
    (created_at, id), running per-entry low-quality / staleness checks plus
    top-K ANN duplicate / contradiction checks against pgvector. Memory stays
    bounded regardless of store size; complexity is O(n × k × log n) with the
    HNSW index, O(n × k × n) without it (still better than the prior O(n²)
    pure-Python all-pairs implementation).
    """

    def __init__(
        self,
        dry_run: bool = False,
        zone: str | None = "live",
        flagged_only: bool = False,
        promote_ready: bool = False,
        *,
        batch_size: int = BATCH_SIZE,
        neighbor_k: int = NEIGHBOR_K,
        max_entries: int = MAX_ENTRIES_PER_RUN,
    ):
        self.dry_run = dry_run
        self.zone = zone
        self.flagged_only = flagged_only
        self.promote_ready = promote_ready
        self.batch_size = max(1, batch_size)
        self.neighbor_k = max(1, neighbor_k)
        # 0 means "no cap"; any positive value caps total entries audited.
        self.max_entries = max(0, max_entries)

    # -----------------------------------------------------------------
    # Streaming entry iterator (keyset-paginated)
    # -----------------------------------------------------------------

    def _iter_batches(self, db: Session) -> Iterator[list[MemoryEntry]]:
        """Yield batches of entries ordered by (created_at, id) ascending.

        Uses keyset pagination instead of LIMIT/OFFSET so cost stays constant
        per page even as the table grows. Honors ``max_entries`` as a soft cap
        on how many entries the audit will scan in one run.
        """
        last_created: datetime | None = None
        last_id: Any = None
        yielded = 0

        while True:
            q = db.query(MemoryEntry)
            if self.zone:
                q = q.filter(MemoryEntry.zone == self.zone)
            if self.flagged_only:
                q = q.filter(MemoryEntry.audit_flagged == True)  # noqa: E712
            if last_created is not None:
                q = q.filter(or_(
                    MemoryEntry.created_at > last_created,
                    and_(
                        MemoryEntry.created_at == last_created,
                        MemoryEntry.id > last_id,
                    ),
                ))
            q = q.order_by(MemoryEntry.created_at.asc(), MemoryEntry.id.asc())
            q = q.limit(self.batch_size)

            batch = q.all()
            if not batch:
                return

            if self.max_entries and yielded + len(batch) > self.max_entries:
                batch = batch[: self.max_entries - yielded]

            yield batch
            yielded += len(batch)

            last = batch[-1]
            last_created = last.created_at
            last_id = last.id

            if self.max_entries and yielded >= self.max_entries:
                logger.info(
                    "Hit FREUD_MAX_ENTRIES_PER_RUN cap (%d); stopping audit",
                    self.max_entries,
                )
                return
            if len(batch) < self.batch_size:
                return

    # -----------------------------------------------------------------
    # Per-entry checks + actions
    # -----------------------------------------------------------------

    def _check_entry(self, db: Session, entry: MemoryEntry) -> list[AuditFinding]:
        """Run all four checks against a single entry. Pure read-only."""
        findings: list[AuditFinding] = []
        findings.extend(check_low_quality([entry]))
        findings.extend(check_stale_entries([entry]))
        findings.extend(check_near_duplicates_ann(
            db, entry, k=self.neighbor_k, zone=self.zone,
        ))
        findings.extend(check_contradictions_ann(
            db, entry, k=self.neighbor_k, zone=self.zone,
        ))
        return findings

    def _flag_entry(
        self,
        entry_id: str,
        entry_findings: list[AuditFinding],
        report: AuditReport,
    ) -> bool:
        """Flag an entry once if any of its findings are warning/critical.

        Returns True iff the entry was successfully flagged. Per-finding
        failures are logged and skipped.
        """
        for f in entry_findings:
            if f.severity not in ("warning", "critical"):
                continue
            reason = f"{f.check}: {f.detail[:120]}"
            try:
                result = memory_flag(entry_id=entry_id, reason=reason)
            except Exception as exc:
                logger.error("Failed to flag %s: %s", entry_id[:8], exc)
                continue
            if "error" not in result:
                report.entries_flagged += 1
                report.actions_taken.append({
                    "action": "flag",
                    "entry_id": entry_id,
                    "reason": reason,
                })
                logger.info("Flagged %s: %s", entry_id[:8], f.check)
                return True
        return False

    def _calibrate_trust_one(
        self,
        entry: MemoryEntry,
        was_flagged: bool,
        report: AuditReport,
    ) -> None:
        """Per-entry trust calibration; mirrors the body of _calibrate_trust."""
        eid = str(entry.id)
        current_trust = float(entry.trust_score) if entry.trust_score else 0.0
        if was_flagged:
            new_trust = max(0.0, current_trust - TRUST_PENALTY_FLAGGED)
        else:
            new_trust = min(1.0, current_trust + TRUST_BOOST_CLEAN)
        new_trust = round(new_trust, 3)
        if abs(new_trust - current_trust) < 0.001:
            return
        try:
            result = memory_update_trust(entry_id=eid, trust_score=new_trust)
        except Exception as exc:
            logger.error("Failed to update trust for %s: %s", eid[:8], exc)
            return
        if "error" not in result:
            report.entries_trust_updated += 1
            report.actions_taken.append({
                "action": "trust_update",
                "entry_id": eid,
                "old_trust": current_trust,
                "new_trust": new_trust,
            })

    def _auto_promote_one(
        self,
        entry: MemoryEntry,
        was_flagged: bool,
        report: AuditReport,
    ) -> None:
        """Per-entry auto-promote; mirrors the body of _auto_promote."""
        if entry.zone != "live" or was_flagged:
            return
        eid = str(entry.id)
        current_trust = float(entry.trust_score) if entry.trust_score else 0.0
        effective_trust = min(1.0, current_trust + TRUST_BOOST_CLEAN)
        if effective_trust < AUTO_PROMOTE_TRUST_THRESHOLD:
            return
        try:
            result = memory_promote(entry_id=eid, trust_score=effective_trust)
        except Exception as exc:
            logger.error("Failed to promote %s: %s", eid[:8], exc)
            return
        if "error" not in result:
            report.entries_promoted += 1
            report.actions_taken.append({
                "action": "promote",
                "entry_id": eid,
                "trust_score": effective_trust,
            })
            logger.info("Promoted %s to cold (trust=%.3f)", eid[:8], effective_trust)

    # -----------------------------------------------------------------
    # Run loop
    # -----------------------------------------------------------------

    def run(self) -> AuditReport:
        report = AuditReport(
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(
            "Freud audit starting (zone=%s, dry_run=%s, flagged_only=%s, "
            "promote_ready=%s, batch=%d, k=%d, max_entries=%s)",
            self.zone, self.dry_run, self.flagged_only, self.promote_ready,
            self.batch_size, self.neighbor_k, self.max_entries or "unlimited",
        )

        _ensure_tables()
        db = SessionLocal()
        try:
            for batch in self._iter_batches(db):
                for entry in batch:
                    entry_findings = self._check_entry(db, entry)
                    report.entries_scanned += 1

                    if entry_findings:
                        report.findings.extend(entry_findings)

                    if self.dry_run:
                        continue

                    was_flagged = self._flag_entry(
                        str(entry.id), entry_findings, report,
                    )
                    self._calibrate_trust_one(entry, was_flagged, report)
                    if self.promote_ready:
                        self._auto_promote_one(entry, was_flagged, report)
        finally:
            db.close()

        logger.info(
            "Audit complete: scanned=%d findings=%d flagged=%d trust_updates=%d promoted=%d",
            report.entries_scanned, len(report.findings),
            report.entries_flagged, report.entries_trust_updated,
            report.entries_promoted,
        )
        report.finished_at = datetime.now(timezone.utc).isoformat()
        return report

    def _apply_flags(self, report: AuditReport) -> set[str]:
        """Flag entries that have warning/critical findings."""
        flagged_ids: set[str] = set()
        for f in report.findings:
            if f.severity in ("warning", "critical") and f.entry_id not in flagged_ids:
                reason = f"{f.check}: {f.detail[:120]}"
                try:
                    result = memory_flag(entry_id=f.entry_id, reason=reason)
                except Exception as exc:
                    logger.error("Failed to flag %s: %s", f.entry_id[:8], exc)
                    continue
                if "error" not in result:
                    flagged_ids.add(f.entry_id)
                    report.entries_flagged += 1
                    report.actions_taken.append({
                        "action": "flag",
                        "entry_id": f.entry_id,
                        "reason": reason,
                    })
                    logger.info("Flagged %s: %s", f.entry_id[:8], f.check)
        return flagged_ids

    def _calibrate_trust(
        self,
        entries: list[MemoryEntry],
        flagged_ids: set[str],
        report: AuditReport,
    ) -> None:
        """Adjust trust scores: penalize flagged entries, boost clean ones.

        Back-compat list-based wrapper that delegates to ``_calibrate_trust_one``.
        The streaming run loop calls the per-entry helper directly to avoid
        materializing the full entry list.
        """
        for e in entries:
            self._calibrate_trust_one(e, str(e.id) in flagged_ids, report)

    def _auto_promote(
        self,
        entries: list[MemoryEntry],
        flagged_ids: set[str],
        report: AuditReport,
    ) -> None:
        """Promote clean, high-trust live entries to cold zone.

        Back-compat list-based wrapper that delegates to ``_auto_promote_one``.
        """
        for e in entries:
            self._auto_promote_one(e, str(e.id) in flagged_ids, report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_report(report: AuditReport) -> None:
    """Pretty-print the audit report to stdout."""
    print("\n" + "=" * 60)
    print("  FREUD AUDIT REPORT")
    print("=" * 60)
    print(f"  Started:  {report.started_at}")
    print(f"  Finished: {report.finished_at}")
    print(f"  Entries scanned: {report.entries_scanned}")
    print(f"  Total findings:  {len(report.findings)}")
    print()

    if report.finding_counts:
        print("  Findings by check:")
        for check, count in sorted(report.finding_counts.items()):
            print(f"    {check}: {count}")
        print()

    if report.findings:
        print("  Details:")
        for f in report.findings:
            sev_marker = {"info": "·", "warning": "▲", "critical": "✖"}
            marker = sev_marker.get(f.severity, "?")
            print(f"    {marker} [{f.severity}] {f.check}: {f.detail[:100]}")
            print(f"      entry: {f.entry_id[:8]}...")
            if f.related_entry_id:
                print(f"      related: {f.related_entry_id[:8]}...")
        print()

    print(f"  Actions taken:")
    print(f"    Entries flagged:       {report.entries_flagged}")
    print(f"    Trust scores updated:  {report.entries_trust_updated}")
    print(f"    Entries promoted:      {report.entries_promoted}")
    print("=" * 60 + "\n")


def main() -> None:
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    flagged_only = "--flagged-only" in args
    promote_ready = "--promote-ready" in args

    zone: str | None = "live"
    if "--zone" in args:
        idx = args.index("--zone")
        if idx + 1 < len(args):
            zone = args[idx + 1]
            if zone == "all":
                zone = None

    auditor = FreudAuditor(
        dry_run=dry_run,
        zone=zone,
        flagged_only=flagged_only,
        promote_ready=promote_ready,
    )

    t0 = time.time()
    report = auditor.run()
    elapsed = time.time() - t0

    _print_report(report)
    print(f"  Elapsed: {elapsed:.2f}s\n")

    report_path = Path(__file__).parent / "test_cases" / "last_freud_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
    logger.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
