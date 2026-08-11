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
    return {
        "protocolVersion": "1.0",
        "name": f"{STUDENT_NAME}-agent",
        "description": "An AI research assistant that writes structured research briefs and IEEE-style academic abstracts, backed by a Deep Agent with calculator and clock tools.",
        "url": f"{PUBLIC_URL}/v1/responses",
        "version": "0.1.0",
        "capabilities": {"streaming": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": "research-brief",
                "name": "Research Brief",
                "description": "Writes a one-page executive research brief with headline, context, findings, recommendation, and confidence rating.",
                "tags": ["research", "summarization"],
            },
            {
                "id": "ieee-abstract",
                "name": "IEEE Abstract",
                "description": "Writes a formal IEEE-style academic paper abstract from a description of research work.",
                "tags": ["academic", "writing"],
            },
        ],
    }