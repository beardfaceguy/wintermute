from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector
import uuid

Base = declarative_base()


class MemoryEntry(Base):
    __tablename__ = "memory_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(384))  # Adjust dimension if needed
    tags = Column(JSONB, default=dict)
    zone = Column(String(16), default="live")  # 'live' or 'cold'
    trust_score = Column(Float, default=0.0)
    audit_flagged = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
