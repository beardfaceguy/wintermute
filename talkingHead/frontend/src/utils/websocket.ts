export type OnToken = (token: string) => void;
export type OnClose = () => void;

const ws_URL = "localhost"
const ws_PORT = "8000"
export function connectToChatWS(onToken: OnToken, onClose?: OnClose): WebSocket {
    const socket = new WebSocket("ws://" + ws_URL + ":" + ws_PORT + "/ws/chat");
    
    socket.onmessage = (event) => {
        onToken(event.data);
    };

    socket.onclose = () => {
        if (onClose) onClose();
    };

    return socket;
}