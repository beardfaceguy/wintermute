import uuid
from typing import Any, cast

from pgvector.sqlalchemy import Vector  # type: ignore
from sqlalchemy import Boolean, Column, DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class MemoryEntry(Base):
    __tablename__ = "memory_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    text = Column(Text, nullable=False)
    embedding: Column[Any] = cast(Column[Any], Column(Vector(384)))  # type: ignore[arg-type]
    tags = Column(JSONB, default=dict)
    zone = Column(String(16), default="live")  # 'live' or 'cold'
    trust_score: Column[float] = Column(Float, default=0.0)
    audit_flagged = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
