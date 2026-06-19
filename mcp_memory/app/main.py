from app.api.memory import router
from fastapi import FastAPI

app = FastAPI(title="MCP Memory Service")
app.include_router(router, prefix="/api/memory")
