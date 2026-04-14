"""
Minimal HTTP chat endpoint for Titans checkpoints.

Endpoints:
- GET /health
- POST /chat
- POST /reset
"""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from chat_repl import build_prompt, pick_device, postprocess_completion
from generate import generate, load_config, load_tokenizer, resolve_path
from model import ModelConfig, build_model, load_model_source

CHAT_UI_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Titans Chat Test UI</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 900px; margin: 24px auto; padding: 0 12px; }
    #log { border: 1px solid #ddd; border-radius: 8px; min-height: 280px; padding: 10px; white-space: pre-wrap; background: #fafafa; }
    .row { margin-top: 10px; display: flex; gap: 8px; }
    input, button { font-size: 14px; padding: 8px; }
    #message { flex: 1; }
    .meta { color: #555; font-size: 13px; margin-bottom: 8px; }
  </style>
</head>
<body>
  <h2>Titans Chat Test UI</h2>
  <div id="meta" class="meta">Loading health...</div>
  <div id="log"></div>
  <div class="row">
    <input id="session" value="demo" placeholder="session_id">
    <input id="message" placeholder="Type a message...">
    <button id="send">Send</button>
    <button id="reset">Reset</button>
  </div>
  <script>
    const meta = document.getElementById('meta');
    const log = document.getElementById('log');
    const sessionEl = document.getElementById('session');
    const messageEl = document.getElementById('message');
    const sendBtn = document.getElementById('send');
    const resetBtn = document.getElementById('reset');
    const basePath = window.location.pathname.endsWith('/')
      ? window.location.pathname
      : `${window.location.pathname}/`;
    const apiUrl = (name) => `${basePath}${name}`;

    function addLine(text) {
      log.textContent += text + "\\n\\n";
      log.scrollTop = log.scrollHeight;
    }

    async function loadHealth() {
      const r = await fetch(apiUrl('health'));
      const j = await r.json();
      meta.textContent = `ok=${j.ok} | device=${j.device} | ckpt=${j.ckpt}`;
    }

    async function sendMessage() {
      const session_id = sessionEl.value || 'demo';
      const message = messageEl.value.trim();
      if (!message) return;
      addLine(`You: ${message}`);
      messageEl.value = '';
      sendBtn.disabled = true;
      try {
        const r = await fetch(apiUrl('chat'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id, message })
        });
        const j = await r.json();
        if (j.ok) {
          addLine(`Assistant: ${j.reply}`);
        } else {
          addLine(`Error: ${j.error || 'chat failed'}`);
        }
      } catch (e) {
        addLine(`Error: ${e}`);
      } finally {
        sendBtn.disabled = false;
        messageEl.focus();
      }
    }

    async function resetSession() {
      const session_id = sessionEl.value || 'demo';
      await fetch(apiUrl('reset'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id })
      });
      addLine('Session reset.');
    }

    sendBtn.addEventListener('click', sendMessage);
    resetBtn.addEventListener('click', resetSession);
    messageEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') sendMessage();
    });

    loadHealth().catch((e) => { meta.textContent = `Health failed: ${e}`; });
  </script>
</body>
</html>
"""


class ChatService:
    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        device_arg: str,
        max_new: int,
        top_k: int,
        temperature: float,
        max_prompt_tokens: int,
        user_prefix: str,
        assistant_prefix: str,
    ) -> None:
        cfg = load_config(resolve_path(config_path))
        mcfg = ModelConfig(**cfg["model"])
        self.tokenizer = load_tokenizer(resolve_path(cfg["data"]["tokenizer_path"]))
        self.device = pick_device(device_arg)
        self.model = build_model(mcfg).to(self.device)

        ckpt_resolved = resolve_path(ckpt_path)
        load_model_source(self.model, ckpt_resolved, map_location=self.device, strict=True)
        self.model.eval()

        self.default_max_new = max_new
        self.default_top_k = top_k
        self.default_temperature = temperature
        self.max_prompt_tokens = max_prompt_tokens
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix

        self.ckpt_name = Path(ckpt_resolved).name
        self.config_path = str(resolve_path(config_path))
        self._sessions: Dict[str, List[Tuple[str, str]]] = {}
        self._sessions_lock = threading.Lock()
        self._gen_lock = threading.Lock()

    def health(self) -> Dict[str, object]:
        return {
            "ok": True,
            "device": str(self.device),
            "ckpt": self.ckpt_name,
            "config": self.config_path,
            "session_count": len(self._sessions),
        }

    def reset(self, session_id: str) -> Dict[str, object]:
        sid = session_id.strip() or "default"
        with self._sessions_lock:
            self._sessions[sid] = []
        return {"ok": True, "session_id": sid}

    def chat(
        self,
        session_id: str,
        message: str,
        reset: bool,
        max_new: int,
        top_k: int,
        temperature: float,
    ) -> Dict[str, object]:
        sid = session_id.strip() or "default"
        msg = message.strip()
        if not msg:
            raise ValueError("`message` must be a non-empty string.")
        if max_new <= 0:
            raise ValueError("`max_new` must be > 0.")

        with self._sessions_lock:
            history = list(self._sessions.get(sid, []))
            if reset:
                history = []

        prompt = build_prompt(
            history=history,
            user_text=msg,
            tokenizer=self.tokenizer,
            user_prefix=self.user_prefix,
            assistant_prefix=self.assistant_prefix,
            max_prompt_tokens=self.max_prompt_tokens,
        )

        with self._gen_lock:
            out = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                prompt=prompt,
                max_new_tokens=max_new,
                top_k=top_k,
                temperature=temperature,
            )

        raw_completion = out[len(prompt) :] if out.startswith(prompt) else out
        assistant_text = postprocess_completion(raw_completion, self.user_prefix)
        if not assistant_text:
            assistant_text = "(empty completion)"

        with self._sessions_lock:
            new_history = list(history)
            new_history.append((msg, assistant_text))
            self._sessions[sid] = new_history
            turns = len(new_history)

        return {
            "ok": True,
            "session_id": sid,
            "reply": assistant_text,
            "turns": turns,
            "device": str(self.device),
            "ckpt": self.ckpt_name,
        }


def make_handler(service: ChatService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def _send_html(self, status: int, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, status: int, payload: Dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict[str, object]:
            length_raw = self.headers.get("Content-Length", "0")
            length = int(length_raw)
            body = self.rfile.read(length) if length > 0 else b"{}"
            if not body:
                return {}
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object.")
            return payload

        def do_OPTIONS(self) -> None:
            self._send_json(200, {"ok": True})

        def do_GET(self) -> None:
            if self.path == "/":
                self._send_html(200, CHAT_UI_HTML)
                return
            if self.path == "/health":
                self._send_json(200, service.health())
                return
            self._send_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            try:
                payload = self._read_json()
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
                return

            if self.path == "/chat":
                try:
                    session_id = str(payload.get("session_id", "default"))
                    message = str(payload.get("message", ""))
                    reset = bool(payload.get("reset", False))
                    max_new = int(payload.get("max_new", service.default_max_new))
                    top_k = int(payload.get("top_k", service.default_top_k))
                    temperature = float(payload.get("temperature", service.default_temperature))
                    result = service.chat(
                        session_id=session_id,
                        message=message,
                        reset=reset,
                        max_new=max_new,
                        top_k=top_k,
                        temperature=temperature,
                    )
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": f"chat_failed: {exc}"})
                    return
                self._send_json(200, result)
                return

            if self.path == "/reset":
                session_id = str(payload.get("session_id", "default"))
                self._send_json(200, service.reset(session_id))
                return

            self._send_json(404, {"ok": False, "error": "not_found"})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="HTTP chat endpoint for Titans checkpoint.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--config", type=str, default="configs/config_baseline_nomem.yaml", help="YAML config path")
    parser.add_argument("--ckpt", type=str, default="ckpt_step_4000.pt", help="Checkpoint path")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--max-new", type=int, default=80, help="Default max new tokens per reply")
    parser.add_argument("--top-k", type=int, default=20, help="Default top-k sampling")
    parser.add_argument("--temperature", type=float, default=0.8, help="Default sampling temperature")
    parser.add_argument("--max-prompt-tokens", type=int, default=512, help="Prompt token budget for each session")
    parser.add_argument("--user-prefix", type=str, default="User:", help="User line prefix")
    parser.add_argument("--assistant-prefix", type=str, default="Assistant:", help="Assistant line prefix")
    args = parser.parse_args()

    service = ChatService(
        config_path=args.config,
        ckpt_path=args.ckpt,
        device_arg=args.device,
        max_new=args.max_new,
        top_k=args.top_k,
        temperature=args.temperature,
        max_prompt_tokens=args.max_prompt_tokens,
        user_prefix=args.user_prefix,
        assistant_prefix=args.assistant_prefix,
    )
    handler_cls = make_handler(service)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)

    print(f"Titans HTTP chat server listening on http://{args.host}:{args.port}")
    print("Endpoints: GET /health, POST /chat, POST /reset")
    print(f"device={service.device} | ckpt={service.ckpt_name}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
