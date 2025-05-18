from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio

router = APIRouter()

@router.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            #Simulate token generation
            for token in mock_generate_tokens(data):
                await websocket.send_text(token)
                await asyncio.sleep(0.05)  # simulate delay per token
    except WebSocketDisconnect:
        print("Client disconnected")

def mock_generate_tokens(prompt: str):
    return [c for c in f"Echo: {prompt}"]