from fastapi import FastAPI
from app.api import chat
from app.websocket import chat_ws

app = FastAPI()
app.include_router(chat.router, prefix="/api")
app.include_router(chat_ws.router)  # This automatically registers /ws/chat
