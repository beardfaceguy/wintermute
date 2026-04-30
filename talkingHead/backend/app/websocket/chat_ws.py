import json
import os
import uuid

from db.db_ops import get_recent_messages, store_message
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from memory.strategic import (
    format_memory_context,
    search_relevant_memories,
    store_conversation,
)

from ..chat.llm import ChatProcessor
from .connection_manager import manager

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

                # Search strategic memory for relevant context
                memories = await search_relevant_memories(user_message, limit=3)
                memory_block = format_memory_context(memories)

                # Fetch recent conversation history
                history = await get_recent_messages(session_id)

                # Format prompt: memory context → conversation history → current turn
                formatted_prompt = ""
                if memory_block:
                    formatted_prompt += memory_block + "\n"
                for msg in history:
                    role = msg.role
                    content = msg.content
                    formatted_prompt += f"{role}: {content}\n"
                formatted_prompt += f"user: {user_message}\nassistant:"

                if DEBUG and memory_block:
                    print(f"[DEBUG] Injected memory context:\n{memory_block}")

                # Generate and stream response
                assistant_message = await chat_processor.stream_response(
                    formatted_prompt, websocket.send_text
                )

                if DEBUG:
                    print(f"[DEBUG] Assistant full response: {assistant_message}")

                # Store assistant's message in conversation history
                await store_message(
                    session_id=session_id,
                    role="assistant",
                    content=assistant_message,
                )

                # Persist the exchange in strategic memory (fire-and-forget)
                await store_conversation(
                    session_id=session_id,
                    user_message=user_message,
                    assistant_message=assistant_message,
                )

            except json.JSONDecodeError:
                await websocket.send_text("Error: Invalid JSON format")
            except Exception as e:
                await websocket.send_text(f"Error: {str(e)}")
                if DEBUG:
                    import traceback

                    traceback.print_exc()

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        if DEBUG:
            print(f"[DEBUG] WebSocket session disconnected: {session_id}")
