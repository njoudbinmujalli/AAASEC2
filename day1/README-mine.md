# Day 1 — Research Agent (Single-Agent Pipeline)

A LangGraph agent that researches a topic, analyzes sources with RAG-backed memory, self-evaluates quality, and retries with a changed query if the score is too low.

## Flow

START -> collect -> store_memory -> analyze -> evaluate
^ |
+-- quality < 7 (max 3 tries) -----+
v quality >= 7
report -> audit -> END


## What it demonstrates

- **State with a reducer** - `execution_logs` uses `Annotated[List[str], operator.add]` so every node appends instead of overwriting.
- **Structured output** - quality scoring uses `llm.with_structured_output(QualityScore)` instead of parsing free text, so a `score: int` is guaranteed valid.
- **A real retry loop with two guardrails** - the query changes every retry (so retrying isn't pointless), and `iteration_count` hard-caps at 3 (so a stuck loop can't run forever / hit `GraphRecursionError`).
- **Basic RAG** - each source is analyzed with related past sources pulled from an in-memory vector store via `similarity_search`.
- **`USE_FAKE=1` fallback** - the search step swaps to deterministic fake results when no Tavily key is set, so the graph structure and retry logic can be verified without any API keys.

## File

`my_agent.py` - the completed implementation (skeleton was `day1_lab_skeleton.py`).

## Running it

```bash
uv sync
cp .env.example .env   # add OPENAI_API_KEY (OpenRouter) + TAVILY_API_KEY
                        # or set USE_FAKE=1 to run with no keys at all
uv run python my_agent.py
```

Prints the generated Mermaid diagram (paste into mermaid.live to see the graph shape), the final report, and the full execution log.
