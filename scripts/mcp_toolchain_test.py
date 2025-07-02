#!/usr/bin/env python3
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()
model_name = os.getenv("MODEL_NAME")

vllm_url = "http://localhost:8000/v1/completions"
mcp_map = {"map_list": "http://localhost:6010/list"}

chat = [
    {"role": "system", "content": "You are a helpful assistant with access to tools."},
    {"role": "user", "content": "List the contents of /home using your tools."},
]

payload = {
    "model": "models/{model_name}",
    "messages": chat,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "map_list",
                "description": "List files in a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory to list, like /home",
                        }
                    },
                    "required": ["path"],
                },
            },
        }
    ],
    "tool_choice": "auto",
    "temperature": 0.3,
    "max_tokens": 256,
}

resp = requests.post(vllm_url, json=payload)
resp.raise_for_status()
resp_data = resp.json()
choices = resp_data.get("choices", [])

for choice in choices:
    tool_calls = choice["message"].get("tool_calls", [])
    if tool_calls:
        for call in tool_calls:
            args = json.loads(call["function"]["arguments"])
            print(
                f"[vLLM ➜ MCP] Calling '{call['function']['name']}' with args: {args}"
            )
            mcp_response = requests.post(mcp_map[call["function"]["name"]], json=args)
            print(f"[MCP ➜ vLLM] Response:\n{mcp_response.text}")
    else:
        print("[vLLM Response]:", choice["message"]["content"])
