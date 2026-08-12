"""
Day 4 — A simple agent with real shell/filesystem access.
SECURITY NOTE: LocalShellBackend's root_dir confines filesystem TOOLS
(read/write/edit), but the shell EXECUTE capability runs with the real
OS user's full permissions — confirmed by testing (see RESULTS.md).
Do not run destructive commands against this agent outside a container.
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

load_dotenv()

WORKSPACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
os.makedirs(WORKSPACE, exist_ok=True)

llm = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
)

agent = create_deep_agent(
    model=llm,
    tools=[],
    system_prompt=(
        "You are a helpful coding assistant. You can read, write, edit, "
        "and execute files inside your workspace to complete tasks."
    ),
    backend=LocalShellBackend(root_dir=WORKSPACE),
)


async def ask(task: str):
    result = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
    last = result["messages"][-1]
    print(last.content if hasattr(last, "content") else last["content"])


if __name__ == "__main__":
    asyncio.run(ask("Create a file called hello.txt containing the text 'Hello from my agent!'"))

