"""
Wintermute mcp-memory — MCP server for strategic memory with pgvector.

Run:
    python mcp_memory/server.py                    # stdio (default, for Cursor/Claude Desktop)
    python mcp_memory/server.py --transport http   # HTTP (for remote/testing)

Cursor config (.cursor/mcp.json):
    {
      "mcpServers": {
        "wintermute-memory": {
          "command": "python",
          "args": ["mcp_memory/server.py"]
        }
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any

from fastmcp import FastMCP

try:
    from mcp_memory.app.db.session import SessionLocal, engine
    from mcp_memory.app.models.memory_entry import Base, MemoryEntry
except ImportError:
    from app.db.session import SessionLocal, engine
    from app.models.memory_entry import Base, MemoryEntry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-memory")

# Configurable result-set caps. Defaults preserve historical behavior.
_SEARCH_MAX = int(os.getenv("MCP_MEMORY_SEARCH_MAX", "50"))
_RECALL_MAX = int(os.getenv("MCP_MEMORY_RECALL_MAX", "100"))

# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

_embedder = None


def _get_embedder():
    """Lazy-load sentence-transformers model (384-dim, matches pgvector column)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv("MCP_MEMORY_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
        _embedder = SentenceTransformer(model_name)
        logger.info("Loaded embedding model: %s", model_name)
    return _embedder


def _embed(text: str) -> list[float]:
    model = _get_embedder()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


# ---------------------------------------------------------------------------
# Ensure tables exist (deferred to first DB access)
# ---------------------------------------------------------------------------

_tables_created = False

# HNSW index on memory_entries.embedding accelerates cosine_distance queries
# (memory_search and the Freud auditor) from sequential scans to log(n) graph
# walks. Created idempotently outside any transaction so CONCURRENTLY is legal.
_HNSW_INDEX_NAME = "memory_entries_embedding_hnsw"
# CONCURRENTLY avoids taking ACCESS EXCLUSIVE on memory_entries during the
# initial build against a populated prod DB. Requires running outside any
# transaction (AUTOCOMMIT below) and pgvector >= 0.5.0 + Postgres >= 9.5.
_HNSW_INDEX_DDL = (
    f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_HNSW_INDEX_NAME} "
    "ON memory_entries USING hnsw (embedding vector_cosine_ops)"
)


def _ensure_hnsw_index() -> None:
    """Best-effort HNSW index creation. Postgres + pgvector >= 0.5 only.

    Errors are logged but never raised: the index is a performance optimization,
    not a correctness requirement. Cosine queries still work without it.
    """
    if engine.dialect.name != "postgresql":
        return
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.exec_driver_sql(_HNSW_INDEX_DDL)
        logger.info("HNSW index '%s' ensured", _HNSW_INDEX_NAME)
    except Exception as exc:
        logger.warning("Could not create HNSW index (continuing without it): %s", exc)


def _ensure_tables():
    global _tables_created
    if not _tables_created:
        Base.metadata.create_all(bind=engine)
        _ensure_hnsw_index()
        _tables_created = True
        logger.info("Database tables ensured")


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="wintermute-memory",
    instructions=(
        "Wintermute's strategic memory store. Use memory_add to record strategies, "
        "outcomes, and observations. Use memory_search to find relevant past entries "
        "via semantic similarity. Entries start in the 'live' zone and can be promoted "
        "to 'cold' once verified."
    ),
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool
def memory_add(
    text: str,
    tags: dict[str, str] | None = None,
    zone: str = "live",
) -> dict[str, Any]:
    """Store a new memory entry with auto-generated embedding.

    Args:
        text: The content to remember (strategy, observation, outcome, etc.)
        tags: Optional key-value metadata tags for categorization.
        zone: Memory zone — 'live' (default, unverified) or 'cold' (verified).

    Returns:
        The created entry with its id and timestamp.
    """
    _ensure_tables()
    embedding = _embed(text)
    db = SessionLocal()
    try:
        entry = MemoryEntry(
            text=text,
            embedding=embedding,
            tags=tags or {},
            zone=zone,
            trust_score=0.0,
            audit_flagged=False,
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        return _entry_to_dict(entry)
    finally:
        db.close()


@mcp.tool
def memory_search(
    query: str,
    limit: int = 5,
    zone: str | None = None,
    min_trust: float | None = None,
) -> list[dict[str, Any]]:
    """Semantic search over memory entries using cosine similarity.

    Args:
        query: Natural-language search query.
        limit: Max number of results (default 5, max 50).
        zone: Filter by zone ('live' or 'cold'). None returns both.
        min_trust: Minimum trust_score filter.

    Returns:
        List of matching entries ordered by relevance, with similarity scores.
    """
    _ensure_tables()
    limit = min(max(limit, 1), _SEARCH_MAX)
    query_vec = _embed(query)
    db = SessionLocal()
    try:
        q = db.query(
            MemoryEntry,
            MemoryEntry.embedding.cosine_distance(query_vec).label("distance"),
        )
        if zone:
            q = q.filter(MemoryEntry.zone == zone)
        if min_trust is not None:
            q = q.filter(MemoryEntry.trust_score >= min_trust)
        q = q.order_by("distance").limit(limit)

        results = []
        for entry, distance in q.all():
            d = _entry_to_dict(entry)
            d["similarity"] = round(1.0 - distance, 4)
            results.append(d)
        return results
    finally:
        db.close()


@mcp.tool
def memory_recall_recent(
    limit: int = 10,
    zone: str | None = None,
    tag_key: str | None = None,
    tag_value: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve recent memory entries in reverse chronological order.

    Args:
        limit: Number of entries to return (default 10, max 100).
        zone: Filter by zone ('live' or 'cold'). None returns both.
        tag_key: Filter entries that have this tag key.
        tag_value: If tag_key is set, also match this tag value.

    Returns:
        List of recent entries.
    """
    _ensure_tables()
    limit = min(max(limit, 1), _RECALL_MAX)
    db = SessionLocal()
    try:
        q = db.query(MemoryEntry)
        if zone:
            q = q.filter(MemoryEntry.zone == zone)
        if tag_key:
            if tag_value:
                q = q.filter(MemoryEntry.tags[tag_key].astext == tag_value)
            else:
                q = q.filter(MemoryEntry.tags.has_key(tag_key))  # noqa: SIM118
        q = q.order_by(MemoryEntry.created_at.desc()).limit(limit)
        return [_entry_to_dict(e) for e in q.all()]
    finally:
        db.close()


@mcp.tool
def memory_promote(
    entry_id: str,
    trust_score: float = 1.0,
) -> dict[str, Any]:
    """Promote a memory entry from 'live' to 'cold' (verified) zone.

    Only entries with trust_score >= 0.7 and zone == 'live' can be promoted.

    Args:
        entry_id: UUID of the entry to promote.
        trust_score: New trust score to assign (must be >= 0.7).

    Returns:
        The updated entry.
    """
    if trust_score < 0.7:
        return {"error": "trust_score must be >= 0.7 for promotion to cold zone"}

    _ensure_tables()
    db = SessionLocal()
    try:
        entry = db.query(MemoryEntry).filter(MemoryEntry.id == uuid.UUID(entry_id)).first()
        if not entry:
            return {"error": f"Entry {entry_id} not found"}
        if entry.zone != "live":
            return {"error": f"Entry is already in '{entry.zone}' zone"}

        entry.zone = "cold"
        entry.trust_score = trust_score
        entry.audit_flagged = False
        db.commit()
        db.refresh(entry)
        return _entry_to_dict(entry)
    finally:
        db.close()


@mcp.tool
def memory_flag(
    entry_id: str,
    reason: str = "",
) -> dict[str, Any]:
    """Flag a memory entry for audit review (Freud hook point).

    Flagged entries will be reviewed before promotion to cold storage.

    Args:
        entry_id: UUID of the entry to flag.
        reason: Why this entry is being flagged.

    Returns:
        The updated entry.
    """
    _ensure_tables()
    db = SessionLocal()
    try:
        entry = db.query(MemoryEntry).filter(MemoryEntry.id == uuid.UUID(entry_id)).first()
        if not entry:
            return {"error": f"Entry {entry_id} not found"}

        entry.audit_flagged = True
        if reason:
            tags = dict(entry.tags) if entry.tags else {}
            tags["audit_reason"] = reason
            entry.tags = tags
        db.commit()
        db.refresh(entry)
        return _entry_to_dict(entry)
    finally:
        db.close()


@mcp.tool
def memory_update_trust(
    entry_id: str,
    trust_score: float,
) -> dict[str, Any]:
    """Update the trust score of a memory entry.

    Args:
        entry_id: UUID of the entry.
        trust_score: New trust score (0.0 to 1.0).

    Returns:
        The updated entry.
    """
    _ensure_tables()
    trust_score = max(0.0, min(1.0, trust_score))
    db = SessionLocal()
    try:
        entry = db.query(MemoryEntry).filter(MemoryEntry.id == uuid.UUID(entry_id)).first()
        if not entry:
            return {"error": f"Entry {entry_id} not found"}

        entry.trust_score = trust_score
        db.commit()
        db.refresh(entry)
        return _entry_to_dict(entry)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# DAG-structured retrieval tool
# ---------------------------------------------------------------------------


@mcp.tool
def memory_search_deep(
    query: str,
    limit_per_hop: int = 5,
    max_hops: int = 4,
    zone: str | None = None,
    min_trust: float | None = None,
) -> dict[str, Any]:
    """Multi-hop semantic search using DAG-structured query decomposition.

    For complex queries requiring information across multiple memory entries,
    this decomposes the query into subproblems, determines dependencies, and
    retrieves in logical order. Simple queries are automatically routed to
    direct vector search.

    Args:
        query: Natural-language search query.
        limit_per_hop: Max results per subquery hop (default 5).
        max_hops: Maximum subquery nodes to execute (default 4).
        zone: Filter by zone ('live' or 'cold'). None returns both.
        min_trust: Minimum trust_score filter.

    Returns:
        Dict with answer_summary, per-hop results, DAG edges, and routing info.
    """
    try:
        from mcp_memory.dag_retrieval import dag_search_sync
    except ImportError:
        from dag_retrieval import dag_search_sync

    return dag_search_sync(
        query=query,
        limit_per_hop=limit_per_hop,
        max_hops=max_hops,
        zone=zone,
        min_trust=min_trust,
    )


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("memory://stats")
def memory_stats() -> str:
    """Current memory store statistics: counts by zone, flagged entries, etc."""
    _ensure_tables()
    db = SessionLocal()
    try:
        total = db.query(MemoryEntry).count()
        live = db.query(MemoryEntry).filter(MemoryEntry.zone == "live").count()
        cold = db.query(MemoryEntry).filter(MemoryEntry.zone == "cold").count()
        flagged = (
            db.query(MemoryEntry)
            .filter(
                MemoryEntry.audit_flagged == True  # noqa: E712
            )
            .count()
        )

        from sqlalchemy import func as sqlfunc

        avg_trust_row = db.query(sqlfunc.avg(MemoryEntry.trust_score)).first()
        avg_trust = round(float(avg_trust_row[0]), 3) if avg_trust_row[0] else 0.0

        return json.dumps(
            {
                "total_entries": total,
                "live_entries": live,
                "cold_entries": cold,
                "flagged_for_audit": flagged,
                "avg_trust_score": avg_trust,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry_to_dict(entry: MemoryEntry) -> dict[str, Any]:
    return {
        "id": str(entry.id),
        "text": entry.text,
        "tags": dict(entry.tags) if entry.tags else {},
        "zone": entry.zone,
        "trust_score": float(entry.trust_score) if entry.trust_score else 0.0,
        "audit_flagged": bool(entry.audit_flagged),
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    transport = "stdio"
    port = int(os.getenv("MCP_MEMORY_PORT", "8002"))

    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        if idx + 1 < len(sys.argv):
            transport = sys.argv[idx + 1]

    if transport == "http":
        mcp.run(transport="http", port=port)
    else:
        mcp.run()
