"""
SQLAlchemy stub file for type checking.
"""

from typing import Any

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

Base = declarative_base()

class Message(Base):
    __tablename__: str = "messages"

    id: Any = Column(Integer, primary_key=True)
    session_id: Any = Column(String, nullable=False)
    role: Any = Column(String, nullable=False)
    content: Any = Column(Text, nullable=False)
    timestamp: Any = Column(DateTime(timezone=True), server_default=func.now())
    embedding: Mapped[list[float]] = mapped_column()  # Vector type
    token_count: Any | None = Column(Integer)
