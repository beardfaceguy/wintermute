#!/bin/bash

CONFIG_PATH="../config/shared_api_config.json"
PORT=$(jq -r '.mcp_memory.port' "$CONFIG_PATH")

if [ -z "$PORT" ]; then
  echo "❌ Failed to read mcp_memory.port from $CONFIG_PATH"
  exit 1
fi

echo "🚀 Starting MCP Memory service in production mode on port $PORT"

exec gunicorn app.main:app \
  -k uvicorn.workers.UvicornWorker \
  --bind "0.0.0.0:$PORT" \
  --workers 4 \
  --timeout 60 \
  --log-level info
