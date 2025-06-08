/*
 * /src/hooks/useChatSocket.ts
 */

import { useEffect, useRef } from "react";
import { useDispatch } from "react-redux";
import { addMessage, updateLastAssistantMessage } from "../store/chatSlice";
import { connectToChatWS, buildChatWSUrl } from "../utils/websocket";

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

  const sendMessage = (inputText: string) => {
    if (!inputText.trim()) return;

    // Add user's message to chat
    dispatch(addMessage({ role: "user", text: inputText }));

    // Add assistant placeholder message
    dispatch(addMessage({ role: "assistant", text: "" }));

    if (!isConnected.current) {
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
        socket.send(inputText);
      };

      socketRef.current = socket;
    } else {
      // Reuse existing socket
      socketRef.current?.send(inputText);
    }
  };

  return { sendMessage };
}
