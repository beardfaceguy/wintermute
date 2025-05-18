from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class Message(BaseModel):
    text: str

@router.post("/chat")
def chat_endpoint(message: Message):
    # Placeholder logic - replace with actual AI model call
    return {"response": f"You said: {message.text}"}
