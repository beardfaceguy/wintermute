// src/components/Chat.tsx
import { useState, useEffect, useRef } from "react";
import { useSelector } from "react-redux";
import { type RootState } from "../store";
import useChatSocket from "../hooks/useChatSocket";
import "./Chat.css";
import { VoiceToggleButton } from "./VoiceToggleButton";

export default function Chat() {
  const messages = useSelector((state: RootState) => state.chat.messages);
  const [inputText, setInputText] = useState("");
  const { sendMessage } = useChatSocket();

  const scrollRef = useRef<HTMLDivElement>(null);
  const [isAutoScroll, setIsAutoScroll] = useState(true);

  useEffect(() => {
    const el = scrollRef.current;
    if (el && isAutoScroll) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, isAutoScroll]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;

    const handleScroll = () => {
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 20;
      setIsAutoScroll(atBottom);
    };

    el.addEventListener("scroll", handleScroll);
    return () => el.removeEventListener("scroll", handleScroll);
  }, []);

  const handleSend = () => {
    if (!inputText.trim()) return;
    sendMessage(inputText);
    setInputText("");
  };

  return (
    <>
      <div className="chat-floating-messages" ref={scrollRef}>
        {messages.map((msg, i) => (
          <div key={i} className={`chat-line ${msg.role === "user" ? "chat-user" : "chat-assistant"}`}>
            {msg.text}
          </div>
        ))}
      </div>

      <div className="chat-modal">
        <div className="chat-input-container">
          <textarea
            className="chat-input-textarea"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Type a message..."
          />
          <div className="chat-button-row">
           
            <button className="chat-button" onClick={handleSend}>
              Send
            </button>
             <VoiceToggleButton />
          </div>
        </div>
      </div>
    </>
  );
}
