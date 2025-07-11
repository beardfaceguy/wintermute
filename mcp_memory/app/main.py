from fastapi import FastAPI

from app.api.memory import router

app = FastAPI(title="MCP Memory Service")
app.include_router(router, prefix="/api/memory")
