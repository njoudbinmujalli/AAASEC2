from pathlib import Path
from fastmcp import FastMCP
from fastmcp.server.providers.skills import SkillsDirectoryProvider

mcp = FastMCP("Njoud Tools")


@mcp.tool
def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression, e.g. '2 * (3+4) ** 2'."""
    import ast
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def eval_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](eval_node(node.operand))
        raise ValueError(f"Unsupported expression: {node}")

    tree = ast.parse(expression, mode="eval")
    return float(eval_node(tree.body))


@mcp.tool
def word_stats(text: str) -> dict:
    """Return word count, character count, and average word length for a piece of text."""
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    avg_word_length = sum(len(w) for w in words) / word_count if word_count else 0
    return {
        "word_count": word_count,
        "char_count": char_count,
        "avg_word_length": round(avg_word_length, 2),
    }


mcp.add_provider(SkillsDirectoryProvider(roots=Path(__file__).parent.parent / "skills"))


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8001)