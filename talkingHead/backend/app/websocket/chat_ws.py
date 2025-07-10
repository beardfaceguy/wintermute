import json
import uuid
from fastapi import WebSocket, APIRouter, WebSocketDisconnect
from ..chat.llm import ChatProcessor
from .connection_manager import manager
from db.db_ops import store_message
import os
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
router = APIRouter()
chat_processor = ChatProcessor()

@router.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    session_id = str(uuid.uuid4())

    if DEBUG:
        print(f"[DEBUG] New WebSocket session started: {session_id}")

    try:
        while True:
            data = await websocket.receive_text()
            try:
                if DEBUG:
                     print(f"[DEBUG] Raw WebSocket message: {data}")
                payload = json.loads(data)
                user_message = payload.get("message", "").strip()

                if not user_message:
                    await websocket.send_text("Error: Empty message")
                    continue

                if DEBUG:
                    print(f"[DEBUG] Received user message: {user_message}")

                # Store the user message
                await store_message(
                    session_id=session_id,
                    role="user",
                    content=user_message,
                )

                # Generate and stream response
                assistant_message = await chat_processor.stream_response(user_message, websocket.send_text)


                if DEBUG:
                    print(f"[DEBUG] Assistant full response: {assistant_message}")

                # Store assistant's message
                await store_message(
                    session_id=session_id,
                    role="assistant",
                    content=assistant_message,
                )

            except json.JSONDecodeError:
                await websocket.send_text("Error: Invalid JSON format")
            except Exception as e:
                await websocket.send_text(f"Error: {str(e)}")
                if DEBUG:
                    import traceback
                    traceback.print_exc()

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        if DEBUG:
            print(f"[DEBUG] WebSocket session disconnected: {session_id}")
