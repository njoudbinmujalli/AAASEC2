import ast
import operator
from datetime import datetime

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression: {node}")


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely (no eval)."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: could not evaluate '{expression}' ({e})"


def current_time() -> str:
    """Return the current date and time."""
    return datetime.now().isoformat()



import os
from langchain_openai import ChatOpenAI

USE_FAKE = os.getenv("USE_FAKE") == "1"


class FakeAgent:
    """Deterministic fake agent - same .ainvoke shape as a real Deep Agent."""

    async def ainvoke(self, input: dict) -> dict:
        messages = input["messages"]
        last_user_msg = messages[-1]["content"] if messages else ""
        return {
            "messages": messages + [
                {"role": "assistant", "content": f"[FAKE REPLY] You said: {last_user_msg}"}
            ]
        }


def build_agent():
    """Returns an object with .ainvoke({'messages': [...]}) - real or fake."""
    if USE_FAKE:
        return FakeAgent()

    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    llm = ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
    )

    day3_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    agent = create_deep_agent(
        model=llm,
        tools=[calculate, current_time],
        system_prompt=(
            "You are a helpful assistant with access to a calculator and a clock. "
            "Always use the calculate tool for arithmetic instead of computing it yourself. "
            "Always use the current_time tool when asked about the time or date."
        ),
        backend=FilesystemBackend(root_dir=day3_root, virtual_mode=True),
        skills=["/skills/"],
    )
    return agent




import asyncio

if __name__ == "__main__":
    async def main():
        agent = build_agent()
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": "What is 17 * 23 and what time is it?"}]
        })
        last_message = result["messages"][-1]
        if hasattr(last_message, "content"):
            print(last_message.content)
        else:
            print(last_message["content"])

    asyncio.run(main()) 

