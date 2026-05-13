/*
 * /src/hooks/useChatSocket.ts
 */

import { useEffect, useRef, useCallback } from "react";
import { useDispatch } from "react-redux";
import { addMessage, updateLastAssistantMessage } from "../store/chatSlice";
import { connectToChatWS, buildChatWSUrl } from "../utils/websocket";
import {debugLog} from "../utils/debug";

export default function useChatSocket() {
  const dispatch = useDispatch();
  const socketRef = useRef<WebSocket | null>(null);
  const isConnected = useRef(false);

  useEffect(() => {
    // Clean up on unmount
    return () => {
      socketRef.current?.close();
    };
  }, []);

  const sendMessage = useCallback((inputText: string) => {
    if (!inputText.trim()) return;
    debugLog("inside sendMessage: Sending message:", inputText);
    // Add user's message to chat
    dispatch(addMessage({ role: "user", text: inputText }));

    // Add assistant placeholder message
    dispatch(addMessage({ role: "assistant", text: "" }));
    const payload = JSON.stringify({ message: inputText });
    if (!isConnected.current) {
      debugLog("🔗 Connecting to WebSocket...");
      const socketUrl = buildChatWSUrl();

      const socket = connectToChatWS(socketUrl, {
        onToken: (token) => {
          dispatch(updateLastAssistantMessage(token));
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
      // Reuse existing socket
      debugLog("🔗 Reusing existing WebSocket connection.  Payload:", payload);
      socketRef.current?.send(payload);
    }
  }, [dispatch]);

  return { sendMessage };
}
