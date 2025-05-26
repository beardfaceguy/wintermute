import sys
from pathlib import Path
from app.api import chat
# Automatically add the root directory (which contains 'shared') to sys.path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))



from shared.setup_path import extend_path
extend_path()
from fastapi import FastAPI
from app.api import chat
from app.websocket import chat_ws

app = FastAPI()
app.include_router(chat.router, prefix="/api")
app.include_router(chat_ws.router)  # This automatically registers /ws/chat

