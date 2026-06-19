from llama_index.core.llms import ChatResponse

class ChatMessage:
    role: str
    content: str

class BaseChatEngine:
    def chat(self, message: str, chat_history: list[ChatMessage] | None = None) -> ChatResponse: ...
