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
    <div className="chat-modal">
      <div className="chat-window">
        {messages.map((msg, i) => (
          <div key={i}>
            <strong>{msg.role}:</strong> {msg.text}
          </div>
        ))}
      </div>
      <input
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") handleSend();
        }}
        placeholder="Type a message..."
      />
      <button className="chat-button" onClick={handleSend}>Send</button>
    </div>
  );
}
