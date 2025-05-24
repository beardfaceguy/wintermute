// src/utils/websocket.ts
export type OnToken = (token: string) => void;
export type OnClose = () => void;

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
}

export function buildChatWSUrl(protocol: string = config.web_interface.scheme, hostname: string = config.web_interface.host, port: string = String(config.web_interface.port), path: string = config.web_interface.path): string {
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