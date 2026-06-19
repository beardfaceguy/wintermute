import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def _default_db_url() -> str:
    try:
        from shared.config_loader import load_mcp_memory_config

        return load_mcp_memory_config().get(
            "default_db_url",
            "postgresql://postgres:postgres@localhost:5432/wintermute",
        )
    except Exception:
        return "postgresql://postgres:postgres@localhost:5432/wintermute"


DATABASE_URL = os.getenv("DATABASE_URL", _default_db_url())

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
