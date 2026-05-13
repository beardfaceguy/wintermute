import logging
import os
from typing import List, Optional

from sqlalchemy import delete, func
from sqlalchemy.future import select

from .db_models import Message
from .session_async import AsyncSessionLocal

logger = logging.getLogger(__name__)

_DEFAULT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "20"))
# Per-session retention cap. 0 (default) disables automatic pruning to preserve
# prior behavior; set CHAT_MAX_MESSAGES_PER_SESSION>0 to enable trimming of the
# oldest messages once a session exceeds the cap.
_MAX_MESSAGES_PER_SESSION = int(os.getenv("CHAT_MAX_MESSAGES_PER_SESSION", "0"))


async def store_message(
    session_id: str,
    role: str,
    content: str,
    embedding: Optional[List[float]] = None,
    token_count: Optional[int] = None,
) -> None:
    async with AsyncSessionLocal() as session:
        try:
            message = Message(
                session_id=session_id,
                role=role,
                content=content,
                embedding=embedding,
                token_count=token_count,
            )
            session.add(message)
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise e

    if _MAX_MESSAGES_PER_SESSION > 0:
        try:
            await prune_session_messages(session_id, _MAX_MESSAGES_PER_SESSION)
        except Exception:
            # Pruning is best-effort; never fail a write because retention failed.
            logger.exception("Retention prune failed for session %s", session_id)


async def prune_session_messages(session_id: str, max_messages: int) -> int:
    """Delete oldest messages in ``session_id`` while the row count exceeds ``max_messages``.

    Returns the number of rows deleted. Returns 0 when ``max_messages`` is
    non-positive or the session is already within budget. Uses a portable
    two-step query (count + id-list + delete) so it works on both SQLite and
    PostgreSQL backends.
    """
    if max_messages <= 0:
        return 0

    async with AsyncSessionLocal() as session:
        try:
            count_result = await session.execute(
                select(func.count())
                .select_from(Message)
                .where(Message.session_id == session_id)
            )
            total = count_result.scalar() or 0
            excess = total - max_messages
            if excess <= 0:
                return 0

            ids_result = await session.execute(
                select(Message.id)
                .where(Message.session_id == session_id)
                .order_by(Message.timestamp.asc(), Message.id.asc())
                .limit(excess)
            )
            old_ids = list(ids_result.scalars().all())
            if not old_ids:
                return 0

            await session.execute(
                delete(Message).where(Message.id.in_(old_ids))
            )
            await session.commit()
            return len(old_ids)
        except Exception:
            await session.rollback()
            raise


async def get_recent_messages(session_id: str, limit: int = _DEFAULT_HISTORY_LIMIT) -> List[Message]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.timestamp.desc())
            .limit(limit)
        )
        messages = result.scalars().all()
        return list(reversed(messages))
