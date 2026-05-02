import sys
from pathlib import Path

from app.api import voice_chat
from dotenv import load_dotenv

# Automatically add the root directory (which contains 'shared') to sys.path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from shared.setup_path import extend_path  # noqa: E402

extend_path()
from app.websocket import chat_ws  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(voice_chat.router, prefix="/api")
app.include_router(chat_ws.router)


@app.on_event("startup")
async def _init_db():
    from db.db_models import Base
    from db.session_async import engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
