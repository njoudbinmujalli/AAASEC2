"""
Day 4 - FastAPI wrapper around the shell-enabled agent, for attack testing.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from src.simple_shell_agent import agent

app = FastAPI()


class TaskRequest(BaseModel):
    input: str


@app.post("/task")
async def run_task(request: TaskRequest):
    result = await agent.ainvoke({"messages": [{"role": "user", "content": request.input}]})
    last = result["messages"][-1]
    text = last.content if hasattr(last, "content") else last["content"]
    return {"response": text}
