#!/bin/bash
set -e

# Start PostgreSQL via Docker Compose
echo "🔄 Starting PostgreSQL Docker container..."
cd "$(dirname "$0")/infra"
docker compose up -d

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to become available..."
until docker exec "$(docker compose ps -q postgres)" pg_isready -U postgres > /dev/null 2>&1; do
    sleep 1
done
echo "✅ PostgreSQL is ready."

# Start the backend
echo "🚀 Starting FastAPI backend..."
cd ../talkingHead/backend
./run.sh --debug
