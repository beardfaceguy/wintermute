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

# Export DEBUG flag for the app to use
export DEBUG=$DEBUG
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 

