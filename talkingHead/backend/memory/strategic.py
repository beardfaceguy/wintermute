"""
Async bridge to mcp-memory for the talkingHead chat pipeline.

Provides two operations:
  - search_relevant_memories(): semantic search before LLM call
  - store_conversation(): persist a conversation exchange after response

Uses asyncio.to_thread() to wrap the synchronous mcp-memory functions
so they don't block the async WebSocket event loop.

Memory failures are always caught and logged — they must never break
the chat flow.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger("memory.strategic")

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

_mcp_memory_available = False
try:
    from mcp_memory.server import memory_add, memory_search
    _mcp_memory_available = True
except ImportError as exc:
    logger.warning("mcp-memory not importable (%s) — strategic memory disabled", exc)


async def search_relevant_memories(
    query: str,
    limit: int = 3,
    zone: str | None = None,
    min_trust: float | None = None,
) -> list[dict[str, Any]]:
    """Semantic search over mcp-memory for context relevant to the query.

    Returns an empty list on any failure so the chat flow is never interrupted.
    """
    if not _mcp_memory_available:
        return []
    try:
        results = await asyncio.to_thread(
            memory_search,
            query=query,
            limit=limit,
            zone=zone,
            min_trust=min_trust,
        )
        if DEBUG:
            logger.info("Memory search for %r returned %d results", query[:60], len(results))
        return results
    except Exception:
        logger.exception("memory_search failed — continuing without memory context")
        return []


async def store_conversation(
    session_id: str,
    user_message: str,
    assistant_message: str,
    extra_tags: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Store a conversation exchange in mcp-memory as a 'live' entry.

    Tags the entry with session_id and type=conversation so Freud
    can audit it later and it can be promoted to cold storage.

    Returns the created entry dict, or None on failure.
    """
    if not _mcp_memory_available:
        return None

    text = f"User: {user_message}\nAssistant: {assistant_message}"

    tags: dict[str, str] = {
        "type": "conversation",
        "session_id": session_id,
        "source": "talkingHead",
    }
    if extra_tags:
        tags.update(extra_tags)

    try:
        result = await asyncio.to_thread(
            memory_add,
            text=text,
            tags=tags,
            zone="live",
        )
        if DEBUG:
            logger.info("Stored conversation memory: %s", result.get("id", "?"))
        return result
    except Exception:
        logger.exception("memory_add failed — conversation not persisted to memory")
        return None


def format_memory_context(memories: list[dict[str, Any]]) -> str:
    """Format retrieved memories into a prompt-injectable context block.

    Returns an empty string if no memories are provided, so it can be
    safely interpolated into the prompt without conditional logic.
    """
    if not memories:
        return ""

    lines = ["[Relevant Memory]"]
    for i, mem in enumerate(memories, 1):
        similarity = mem.get("similarity", 0)
        zone = mem.get("zone", "?")
        text = mem.get("text", "").strip()
        lines.append(f"  {i}. [{zone}, sim={similarity}] {text}")
    lines.append("[End Memory]\n")

    return "\n".join(lines)
