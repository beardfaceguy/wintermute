from db import db_ops
from typing import List, Optional

def remember_message(session_id: str, role: str, content: str, embedding: Optional[List[float]], token_count: Optional[int]):
    db_ops.store_message(session_id, role, content, embedding, token_count)

def recall_recent_messages(session_id: str, limit: int = 20):
    return db_ops.get_recent_messages(session_id, limit)
