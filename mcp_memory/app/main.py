from fastapi import FastAPI
from app.api import memory

app = FastAPI(title="MCP Memory Service")
app.include_router(memory.router, prefix="/api/memory")
