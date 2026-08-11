import os
import time
import uuid
from fastapi import FastAPI
from pydantic import BaseModel

from src.agent import build_agent

app = FastAPI()
agent = build_agent()

STUDENT_NAME = os.getenv("STUDENT_NAME", "unknown-student")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8000")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}



class ResponseRequest(BaseModel):
    input: str
    model: str | None = None


@app.post("/v1/responses")
async def create_response(request: ResponseRequest):
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": request.input}]
    })

    last_message = result["messages"][-1]
    text = last_message.content if hasattr(last_message, "content") else last_message["content"]

    return {
        "id": f"resp_{uuid.uuid4().hex[:12]}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": request.model or "nvidia/nemotron-3-super-120b-a12b:free",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": text}
                ],
            }
        ],
    }


@app.get("/.well-known/agent-card.json")
async def agent_card():
    return {"todo": True} 