from app.models.memory_entry import MemoryEntry
from sqlalchemy.orm import Session


def add_memory_entry(
    db: Session, text: str, embedding: list[float], tags: dict[str, str] | None = None
):
    if tags is None:
        tags = {}
    entry = MemoryEntry(text=text, embedding=embedding, tags=tags, zone="live")
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def get_recent_entries(db: Session, limit: int = 10):
    return (
        db.query(MemoryEntry).order_by(MemoryEntry.created_at.desc()).limit(limit).all()
    )
