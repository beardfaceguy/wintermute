/*
 * /src/hooks/useChatSocket.ts
 */

import { useEffect, useRef, useCallback } from "react";
import { useDispatch } from "react-redux";
import { addMessage, updateLastAssistantMessage } from "../store/chatSlice";
import { connectToChatWS, buildChatWSUrl } from "../utils/websocket";
import {debugLog} from "../utils/debug";

type UseChatSocketOptions = {
  onAssistantComplete?: (text: string) => void;
};

export default function useChatSocket(options: UseChatSocketOptions = {}) {
  const dispatch = useDispatch();
  const socketRef = useRef<WebSocket | null>(null);
  const isConnected = useRef(false);
  // Buffer the latest assistant tokens so onAssistantComplete sees the
  // full text without a Redux round-trip race.
  const assistantBufferRef = useRef<string>("");
  // Keep the latest callback reference without re-establishing the socket.
  const onCompleteRef = useRef<UseChatSocketOptions["onAssistantComplete"]>(
    options.onAssistantComplete,
  );
  useEffect(() => {
    onCompleteRef.current = options.onAssistantComplete;
  }, [options.onAssistantComplete]);

  useEffect(() => {
    // Clean up on unmount
    return () => {
      socketRef.current?.close();
    };
  }, []);

  const sendMessage = useCallback((inputText: string) => {
    if (!inputText.trim()) return;
    debugLog("inside sendMessage: Sending message:", inputText);
    dispatch(addMessage({ role: "user", text: inputText }));

    dispatch(addMessage({ role: "assistant", text: "" }));
    assistantBufferRef.current = "";
    const payload = JSON.stringify({ message: inputText });
    if (!isConnected.current) {
      debugLog("🔗 Connecting to WebSocket...");
      const socketUrl = buildChatWSUrl();

      const socket = connectToChatWS(socketUrl, {
        onToken: (token) => {
          assistantBufferRef.current += token;
          dispatch(updateLastAssistantMessage(token));
        },
        onAssistantComplete: () => {
          const finalText = assistantBufferRef.current;
          assistantBufferRef.current = "";
          if (onCompleteRef.current) onCompleteRef.current(finalText);
        },
        onClose: () => {
          console.log("WebSocket closed");
          isConnected.current = false;
        },
      });

      socket.onopen = () => {
        isConnected.current = true;
        socket.send(payload);
      };

      socketRef.current = socket;
    } else {
      debugLog("🔗 Reusing existing WebSocket connection.  Payload:", payload);
      socketRef.current?.send(payload);
    }
  }, [dispatch]);

  return { sendMessage };
}
