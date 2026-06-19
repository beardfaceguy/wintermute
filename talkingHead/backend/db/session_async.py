import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

CHAT_DB_URL = os.environ.get(
    "CHAT_DB_URL",
    "sqlite+aiosqlite:///./wintermute_chat.db",
)

engine = create_async_engine(CHAT_DB_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
