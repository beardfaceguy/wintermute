#!/bin/bash
DEBUG=true

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) DEBUG=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"

export DEBUG=$DEBUG
uvicorn app.main:app --reload --host "$BACKEND_HOST" --port "$BACKEND_PORT"

