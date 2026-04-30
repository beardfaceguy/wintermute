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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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

DUPLICATE_SIMILARITY_THRESHOLD = 0.92
CONTRADICTION_SIMILARITY_RANGE = (0.65, 0.90)
MIN_TEXT_LENGTH = 20
STALE_DAYS = 14
TRUST_BOOST_CLEAN = 0.15
TRUST_PENALTY_FLAGGED = 0.2
AUTO_PROMOTE_TRUST_THRESHOLD = 0.8


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
    in one but not the other.
    """
    negation_markers = {
        "not", "never", "don't", "doesn't", "didn't", "won't", "can't",
        "cannot", "shouldn't", "failed", "incorrect", "wrong", "false",
        "no", "none", "neither", "nor",
    }
    findings = []

    for i, a in enumerate(entries):
        if a.embedding is None:
            continue
        a_words = set((a.text or "").lower().split())
        a_negations = a_words & negation_markers

        for b in entries[i + 1:]:
            if b.embedding is None:
                continue
            sim = _cosine_sim(list(a.embedding), list(b.embedding))
            lo, hi = CONTRADICTION_SIMILARITY_RANGE
            if lo <= sim <= hi:
                b_words = set((b.text or "").lower().split())
                b_negations = b_words & negation_markers
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
    """Batch auditor for mcp-memory entries."""

    def __init__(
        self,
        dry_run: bool = False,
        zone: str | None = "live",
        flagged_only: bool = False,
        promote_ready: bool = False,
    ):
        self.dry_run = dry_run
        self.zone = zone
        self.flagged_only = flagged_only
        self.promote_ready = promote_ready

    def _load_entries(self) -> list[MemoryEntry]:
        _ensure_tables()
        db = SessionLocal()
        try:
            q = db.query(MemoryEntry)
            if self.zone:
                q = q.filter(MemoryEntry.zone == self.zone)
            if self.flagged_only:
                q = q.filter(MemoryEntry.audit_flagged == True)  # noqa: E712
            entries = q.order_by(MemoryEntry.created_at.asc()).all()
            db.expunge_all()
            return entries
        finally:
            db.close()

    def run(self) -> AuditReport:
        report = AuditReport(
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "Freud audit starting (zone=%s, dry_run=%s, flagged_only=%s, promote_ready=%s)",
            self.zone, self.dry_run, self.flagged_only, self.promote_ready,
        )

        entries = self._load_entries()
        report.entries_scanned = len(entries)
        logger.info("Loaded %d entries for audit", len(entries))

        if not entries:
            report.finished_at = datetime.now(timezone.utc).isoformat()
            return report

        # --- Run all checks ---
        logger.info("Running quality check...")
        report.findings.extend(check_low_quality(entries))

        logger.info("Running near-duplicate check...")
        report.findings.extend(check_near_duplicates(entries))

        logger.info("Running contradiction check...")
        report.findings.extend(check_contradictions(entries))

        logger.info("Running staleness check...")
        report.findings.extend(check_stale_entries(entries))

        logger.info(
            "Checks complete: %d findings across %d entries",
            len(report.findings), len(entries),
        )

        if self.dry_run:
            report.finished_at = datetime.now(timezone.utc).isoformat()
            return report

        # --- Take actions ---
        flagged_ids = self._apply_flags(report)
        self._calibrate_trust(entries, flagged_ids, report)

        if self.promote_ready:
            self._auto_promote(entries, flagged_ids, report)

        report.finished_at = datetime.now(timezone.utc).isoformat()
        return report

    def _apply_flags(self, report: AuditReport) -> set[str]:
        """Flag entries that have warning/critical findings."""
        flagged_ids: set[str] = set()
        for f in report.findings:
            if f.severity in ("warning", "critical") and f.entry_id not in flagged_ids:
                reason = f"{f.check}: {f.detail[:120]}"
                result = memory_flag(entry_id=f.entry_id, reason=reason)
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
        """Adjust trust scores: penalize flagged entries, boost clean ones."""
        for e in entries:
            eid = str(e.id)
            current_trust = float(e.trust_score) if e.trust_score else 0.0

            if eid in flagged_ids:
                new_trust = max(0.0, current_trust - TRUST_PENALTY_FLAGGED)
            else:
                new_trust = min(1.0, current_trust + TRUST_BOOST_CLEAN)

            new_trust = round(new_trust, 3)
            if abs(new_trust - current_trust) < 0.001:
                continue

            result = memory_update_trust(entry_id=eid, trust_score=new_trust)
            if "error" not in result:
                report.entries_trust_updated += 1
                report.actions_taken.append({
                    "action": "trust_update",
                    "entry_id": eid,
                    "old_trust": current_trust,
                    "new_trust": new_trust,
                })

    def _auto_promote(
        self,
        entries: list[MemoryEntry],
        flagged_ids: set[str],
        report: AuditReport,
    ) -> None:
        """Promote clean, high-trust live entries to cold zone."""
        for e in entries:
            eid = str(e.id)
            if e.zone != "live":
                continue
            if eid in flagged_ids:
                continue
            current_trust = float(e.trust_score) if e.trust_score else 0.0
            effective_trust = min(1.0, current_trust + TRUST_BOOST_CLEAN)
            if effective_trust < AUTO_PROMOTE_TRUST_THRESHOLD:
                continue

            result = memory_promote(entry_id=eid, trust_score=effective_trust)
            if "error" not in result:
                report.entries_promoted += 1
                report.actions_taken.append({
                    "action": "promote",
                    "entry_id": eid,
                    "trust_score": effective_trust,
                })
                logger.info("Promoted %s to cold (trust=%.3f)", eid[:8], effective_trust)


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
