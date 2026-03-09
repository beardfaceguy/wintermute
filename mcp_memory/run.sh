#!/bin/bash

CONFIG_PATH="../config/shared_api_config.json"
PORT=$(jq -r '.mcp_memory.port' "$CONFIG_PATH")

if [ -z "$PORT" ]; then
  echo "❌ Failed to read mcp_memory.port from $CONFIG_PATH"
  exit 1
fi

echo "🚀 Starting mcp-memory service on port $PORT"
uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --reload
