#!/bin/bash
set -a
source ../.env
set +a

echo "Using model: $MODEL_NAME"
PROMPT="$1"

if [ -z "$PROMPT" ]; then
  echo "Usage: ./run_local_prompt.sh \"Your question here\""
  exit 1
fi

curl http://localhost:8001/v1/completions \
  -s \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_NAME\",
    \"prompt\": \"### System:\\nYou are an expert AI assistant designed for software developers. You provide detailed, technically nuanced, and practical answers,\\n\\n### User:\\n$PROMPT\\n\\n### Assistant:\\n\",
    \"max_tokens\": 512,
    \"temperature\": 0.95,
    \"top_p\": 0.95
  }" | jq -r '.choices[0].text'
