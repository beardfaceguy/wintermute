# mcp_echo.py
import os

import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()

MAP_ECHO_HOST = os.getenv("MAP_ECHO_HOST", "0.0.0.0")
MAP_ECHO_PORT = int(os.getenv("MAP_ECHO_PORT", "6010"))


@app.post("/echo")
async def echo(request: Request):
    body = await request.json()
    return {"echo": body}


if __name__ == "__main__":
    uvicorn.run(app, host=MAP_ECHO_HOST, port=MAP_ECHO_PORT)
