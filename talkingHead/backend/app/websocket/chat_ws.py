from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import httpx
import asyncio
import json

router = APIRouter()

@router.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection open")

    try:
        user_input = await websocket.receive_text()
        print(f"Prompt: {user_input[:50]}...")

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                "http://192.168.8.3:8001/v1/chat/completions",
                json={
                    "model": "mistral-7b-instruct-awq",
                    "messages": [{"role": "user", "content": user_input}],
                    "stream": True,
                },
            ) as response:
                async for line in response.aiter_lines():
                    if line.strip() == "" or line.startswith(":"):
                        continue
                    if line.strip() == "data: [DONE]":
                        break

                    try:
                        payload = json.loads(line.removeprefix("data: "))
                        content = payload["choices"][0]["delta"].get("content")
                        if content:
                            await websocket.send_text(content)
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Streaming parse error: {e}")

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        print(f"Unexpected error: {e}")

    finally:
        try:
            await websocket.close()
        except RuntimeError as e:
            print(f"WebSocket already closed: {e}")
        print("WebSocket connection closed")
