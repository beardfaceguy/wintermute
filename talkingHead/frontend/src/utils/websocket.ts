// src/utils/websocket.ts
export type OnToken = (token: string) => void;
export type OnClose = () => void;
const WEB_PROTOCOL = "ws";
const WEB_HOST = "192.168.8.3";
const WEB_PORT = "8000";
const WEB_PATH = "/ws/chat";



interface WebSocketCallbacks {
  onToken: (msg: string) => void;
  onClose?: () => void;
}

export function buildChatWSUrl(protocol: string = WEB_PROTOCOL, hostname: string = WEB_HOST, port: string = WEB_PORT, path: string = WEB_PATH): string {
  return (new URL(`${protocol}://${hostname}:${port}${path}`)).toString();
}



export function connectToChatWS(
  socketUrl: string,
  { onToken, onClose }: WebSocketCallbacks
): WebSocket {
  const socket = new WebSocket(socketUrl);

  socket.onmessage = (event) => onToken(event.data);
  socket.onclose = () => {
    if (onClose) onClose();
  };

  return socket;
}


export function buildURLAndConnectToChatWS({onToken, onClose}: WebSocketCallbacks): WebSocket {
    const url: string = buildChatWSUrl();
    return connectToChatWS(url, {onToken, onClose});
}