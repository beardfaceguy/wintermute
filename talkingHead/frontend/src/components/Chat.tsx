// src/components/Chat.tsx
import { useState, useEffect, useRef } from "react";
import { useSelector } from "react-redux";
import { type RootState } from "../store";
import useChatSocket from "../hooks/useChatSocket";
import VoiceToggleButton from "./VoiceToggleButton";
import "./Chat.css";
import { debugLog } from "../utils/debug";
import { useCallback } from "react";

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

  // Shared logic for both text input and voice transcription
  const handleSend = useCallback( (text: string) => {
    debugLog("💬 handleSend called with:", text);
    if (!text.trim()) return;
    sendMessage(text);
    setInputText("");
  }, [sendMessage]);

  return (
    <>
      <div className="chat-floating-messages" ref={scrollRef}>
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`chat-line ${msg.role === "user" ? "chat-user" : "chat-assistant"}`}
          >
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
                handleSend(inputText);
              }
            }}
            placeholder="Type a message..."
          />
          <div className="chat-button-row">
            <button className="chat-button" onClick={() => handleSend(inputText)}>
              Send
            </button>
            <VoiceToggleButton onVoiceSubmit={handleSend} />
          </div>
        </div>
      </div>
    </>
  );
}
