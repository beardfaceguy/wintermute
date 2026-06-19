from app.db.session import SessionLocal
from app.services import memory_ops
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

router: APIRouter = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/ping")
async def ping():
    return {"status": "alive"}


@router.post("/add")
async def add_entry(
    text: str,
    embedding: list[float],
    tags: dict[str, str] | None = None,
    db: Session = Depends(get_db),
):
    if tags is None:
        tags = {}
    return memory_ops.add_memory_entry(db, text, embedding, tags)


@router.get("/recent")
async def recent_entries(limit: int = 10, db: Session = Depends(get_db)):
    return memory_ops.get_recent_entries(db, limit)
