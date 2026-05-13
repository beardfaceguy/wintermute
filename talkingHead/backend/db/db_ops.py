import os
from typing import List, Optional

from sqlalchemy.future import select

from .db_models import Message
from .session_async import AsyncSessionLocal

_DEFAULT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "20"))


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
