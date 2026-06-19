from db import db_ops


async def remember_message(
    session_id: str,
    role: str,
    content: str,
    embedding: list[float] | None,
    token_count: int | None,
):
    await db_ops.store_message(session_id, role, content, embedding, token_count)


async def recall_recent_messages(session_id: str, limit: int = 20):
    return await db_ops.get_recent_messages(session_id, limit)
