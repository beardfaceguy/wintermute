"""
Code Review API server for the fine-tuned Titans codereview model.

Endpoints:
  GET  /health          — readiness probe
  POST /v1/review       — request a code review
  POST /v1/review/batch — batch review multiple files

The model expects prompts in the SFT training format:
  User: Review the following code change. File: <path> ... Code: <code>
  Assistant: <review comment>

Usage:
  python serve_codereview.py \
    --config configs/config_sft_codereview.yaml \
    --ckpt ckpt_sft_step_3000.pt \
    --port 8020
"""

import argparse
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

import torch

from generate import generate, load_config, load_tokenizer, resolve_path
from model import ModelConfig, build_model, load_model_source


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def normalize_text(value: str) -> str:
    text = str(value or "").strip()
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_review_prompt(
    file_path: str = "",
    language: str = "",
    diff: str = "",
    code: str = "",
    pr_title: str = "",
    line: Optional[int] = None,
) -> str:
    parts = ["Review the following code change."]
    if pr_title:
        parts.append(f"PR: {pr_title}")
    if file_path:
        parts.append(f"File: {file_path}")
    if language:
        parts.append(f"Language: {language}")
    if line is not None:
        parts.append(f"Line: {line}")
    if diff:
        parts.append(f"Diff: {diff}")
    elif code:
        parts.append(f"Code: {code}")
    prompt_body = normalize_text(" ".join(parts))
    return f"User: {prompt_body} Assistant:"


def extract_review(raw_output: str, prompt: str) -> str:
    completion = raw_output[len(prompt):] if raw_output.startswith(prompt) else raw_output
    completion = completion.strip()
    for stop in ("\nUser:", "\nAssistant:"):
        if stop in completion:
            completion = completion.split(stop, 1)[0].strip()
    return completion


class CodeReviewService:
    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        device_arg: str,
        max_new: int,
        top_k: int,
        temperature: float,
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
        self.ckpt_name = Path(ckpt_resolved).name
        self.config_path = str(resolve_path(config_path))
        self._gen_lock = threading.Lock()
        self._request_count = 0
        self._start_time = time.time()

    def health(self) -> Dict[str, object]:
        return {
            "ok": True,
            "model": "codereview-407m",
            "device": str(self.device),
            "ckpt": self.ckpt_name,
            "requests_served": self._request_count,
            "uptime_seconds": int(time.time() - self._start_time),
        }

    def review(
        self,
        file_path: str = "",
        language: str = "",
        diff: str = "",
        code: str = "",
        pr_title: str = "",
        line: Optional[int] = None,
        max_new: Optional[int] = None,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, object]:
        if not diff and not code:
            raise ValueError("Either `diff` or `code` must be provided.")

        prompt = build_review_prompt(
            file_path=file_path,
            language=language,
            diff=diff,
            code=code,
            pr_title=pr_title,
            line=line,
        )

        with self._gen_lock:
            raw = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                prompt=prompt,
                max_new_tokens=max_new or self.default_max_new,
                top_k=top_k if top_k is not None else self.default_top_k,
                temperature=temperature if temperature is not None else self.default_temperature,
                stop_strings=["\nUser:", "\nAssistant:"],
            )

        comment = extract_review(raw, prompt)
        self._request_count += 1

        comment_type = "general"
        type_match = re.match(r"^\[(\w+)\]\s*", comment)
        if type_match:
            comment_type = type_match.group(1).lower()
            comment = comment[type_match.end():].strip()

        return {
            "ok": True,
            "comment": comment,
            "comment_type": comment_type,
            "file_path": file_path,
            "model": "codereview-407m",
        }

    def review_batch(
        self,
        items: List[Dict[str, object]],
        max_new: Optional[int] = None,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, object]:
        results = []
        for item in items:
            try:
                result = self.review(
                    file_path=str(item.get("file_path", "")),
                    language=str(item.get("language", "")),
                    diff=str(item.get("diff", "")),
                    code=str(item.get("code", "")),
                    pr_title=str(item.get("pr_title", "")),
                    line=item.get("line"),
                    max_new=max_new,
                    top_k=top_k,
                    temperature=temperature,
                )
                results.append(result)
            except Exception as exc:
                results.append({
                    "ok": False,
                    "error": str(exc),
                    "file_path": str(item.get("file_path", "")),
                })
        return {"ok": True, "reviews": results, "count": len(results)}


def make_handler(service: CodeReviewService, api_key: Optional[str]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _check_auth(self) -> bool:
            if not api_key:
                return True
            auth = self.headers.get("Authorization", "")
            if auth == f"Bearer {api_key}":
                return True
            self._send_json(401, {"ok": False, "error": "unauthorized"})
            return False

        def _send_json(self, status: int, payload: Dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict[str, object]:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(body.decode("utf-8")) if body else {}

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, service.health())
                return
            self._send_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            if not self._check_auth():
                return
            try:
                payload = self._read_json()
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"ok": False, "error": f"invalid_json: {exc}"})
                return

            if self.path == "/v1/review":
                try:
                    result = service.review(
                        file_path=str(payload.get("file_path", "")),
                        language=str(payload.get("language", "")),
                        diff=str(payload.get("diff", "")),
                        code=str(payload.get("code", "")),
                        pr_title=str(payload.get("pr_title", "")),
                        line=payload.get("line"),
                        max_new=payload.get("max_new"),
                        top_k=payload.get("top_k"),
                        temperature=payload.get("temperature"),
                    )
                    self._send_json(200, result)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": f"review_failed: {exc}"})
                return

            if self.path == "/v1/review/batch":
                items = payload.get("items", [])
                if not isinstance(items, list) or not items:
                    self._send_json(400, {"ok": False, "error": "`items` must be a non-empty array"})
                    return
                try:
                    result = service.review_batch(
                        items=items,
                        max_new=payload.get("max_new"),
                        top_k=payload.get("top_k"),
                        temperature=payload.get("temperature"),
                    )
                    self._send_json(200, result)
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": f"batch_failed: {exc}"})
                return

            self._send_json(404, {"ok": False, "error": "not_found"})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Code Review API server.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--config", type=str, required=True, help="Model YAML config path")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--max-new", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--api-key", type=str, default=None, help="Optional Bearer token for auth")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("CODEREVIEW_API_KEY")

    service = CodeReviewService(
        config_path=args.config,
        ckpt_path=args.ckpt,
        device_arg=args.device,
        max_new=args.max_new,
        top_k=args.top_k,
        temperature=args.temperature,
    )
    handler_cls = make_handler(service, api_key)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)

    print(f"Code Review API listening on http://{args.host}:{args.port}")
    print(f"  POST /v1/review       — single file review")
    print(f"  POST /v1/review/batch — batch review")
    print(f"  GET  /health          — health check")
    print(f"  device={service.device} | ckpt={service.ckpt_name}")
    print(f"  auth={'enabled' if api_key else 'disabled'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
