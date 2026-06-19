// src/utils/websocket.ts

import { debugLog } from "./debug";
export type OnToken = (token: string) => void;
export type OnClose = () => void;
export type OnAssistantComplete = () => void;

// Sentinel string the backend sends when an assistant turn finishes.
// Must stay in sync with chat_ws.END_OF_STREAM_SENTINEL.
export const END_OF_STREAM_SENTINEL = "[[DONE]]";

interface APIConfig {
  vllm: {
    scheme: string;
    host: string;
    port: number;
    path: string;
    model: string;
  };
  web_interface: {
    scheme: string;
    host: string;
    port: number;
    path: string;
  };
}

import rawconfig from '../../../../config/shared_api_config.json'
const config = rawconfig as APIConfig;

interface WebSocketCallbacks {
  onToken: (msg: string) => void;
  onClose?: () => void;
  onAssistantComplete?: () => void;
}

export function buildChatWSUrl(protocol: string = config.web_interface.scheme, hostname: string = config.web_interface.host, port: string = String(config.web_interface.port), path: string = config.web_interface.path): string {
  const effectiveHost = (hostname === "localhost" && typeof window !== "undefined" && window.location.hostname !== "localhost")
    ? window.location.hostname
    : hostname;
  const wsProtocol = protocol === "https" ? "wss" : "ws";
  return (new URL(`${wsProtocol}://${effectiveHost}:${port}${path}`)).toString();
}



export function connectToChatWS(
  socketUrl: string,
  { onToken, onClose, onAssistantComplete }: WebSocketCallbacks
): WebSocket {
  debugLog("websocket.connectToChatWS: Connecting to WebSocket at:", socketUrl);
  const socket = new WebSocket(socketUrl);

  socket.onmessage = (event) => {
    if (event.data === END_OF_STREAM_SENTINEL) {
      if (onAssistantComplete) onAssistantComplete();
      return;
    }
    onToken(event.data);
  };
  socket.onclose = () => {
    if (onClose) onClose();
  };

  return socket;
}


export function buildURLAndConnectToChatWS(
  { onToken, onClose, onAssistantComplete }: WebSocketCallbacks
): WebSocket {
    const url: string = buildChatWSUrl();
    return connectToChatWS(url, { onToken, onClose, onAssistantComplete });
}
