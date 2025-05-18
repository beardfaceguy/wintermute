# mcp_echo.py
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

@app.post("/echo")
async def echo(request: Request):
    body = await request.json()
    return {"echo": body}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6010)

