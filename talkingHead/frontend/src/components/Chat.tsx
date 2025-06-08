// src/components/Chat.tsx
import { useState } from "react";
import { useSelector } from "react-redux";
import { type RootState } from "../store";
import useChatSocket from "../hooks/useChatSocket";
import "./Chat.css";

export default function Chat() {
  const messages = useSelector((state: RootState) => state.chat.messages);
  const [inputText, setInputText] = useState("");
  const { sendMessage } = useChatSocket();

  const handleSend = () => {
    if (!inputText.trim()) return;
    sendMessage(inputText);
    setInputText("");
  };

  return (
    <>
      {/* Free-floating message stack */}
      <div className="chat-floating-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-line ${msg.role === "user" ? "chat-user" : "chat-assistant"}`}>
            {msg.text}
          </div>
        ))}
      </div>

      {/* Fixed input box */}
      <div className="chat-modal">
        <input
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleSend();
          }}
          placeholder="Type a message..."
        />
        <button className="chat-button" onClick={handleSend}>
          Send
        </button>
      </div>
    </>
  );
}
