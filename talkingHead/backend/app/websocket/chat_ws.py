import json
import logging
import os
import uuid

logger = logging.getLogger(__name__)

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


MAX_MESSAGE_SIZE = int(os.getenv("WS_MAX_MESSAGE_SIZE", str(16 * 1024)))

# Sentinel string the frontend recognises as "assistant message complete".
# Kept simple (not JSON) to preserve the existing tokens-as-strings protocol.
# Bumping this is a coordinated frontend+backend change.
END_OF_STREAM_SENTINEL = "[[DONE]]"


@router.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    accepted = await manager.connect(websocket)
    if not accepted:
        return
    session_id = str(uuid.uuid4())

    if DEBUG:
        logger.debug("New WebSocket session started: %s", session_id)

    try:
        while True:
            data = await websocket.receive_text()
            try:
                if len(data) > MAX_MESSAGE_SIZE:
                    await websocket.send_text("Error: Message too large")
                    continue
                if DEBUG:
                    logger.debug("Raw WebSocket message: %s", data)
                payload = json.loads(data)
                user_message = payload.get("message", "").strip()

                if not user_message:
                    await websocket.send_text("Error: Empty message")
                    continue

                if DEBUG:
                    logger.debug("Received user message: %s", user_message)

                # Store the user message
                await store_message(
                    session_id=session_id,
                    role="user",
                    content=user_message,
                )

                # Search strategic memory for relevant context
                memories = await search_relevant_memories(user_message, limit=3, deep=True)
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
                    logger.debug("Injected memory context:\n%s", memory_block)

                # Generate and stream response
                assistant_message = await chat_processor.stream_response(
                    formatted_prompt, websocket.send_text
                )

                # Tell the frontend the turn is done so it can fire TTS / unlock
                # the input box. Sent even on empty responses so the client
                # never has to guess.
                try:
                    await websocket.send_text(END_OF_STREAM_SENTINEL)
                except Exception:
                    logger.debug("Failed to send end-of-stream sentinel", exc_info=True)

                if DEBUG:
                    logger.debug("Assistant full response: %s", assistant_message)

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
        if DEBUG:
            logger.debug("WebSocket session disconnected: %s", session_id)
    except Exception as e:
        logger.error("WebSocket session %s crashed: %s", session_id, e)
        try:
            await websocket.send_text(f"Error: {str(e)}")
        except Exception:
            pass
    finally:
        manager.disconnect(websocket)
