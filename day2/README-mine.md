# Day 2 — Research Team (Supervisor Pattern, Multi-Agent)

Four LLM personas (researcher, analyst, writer, critic) coordinated by a supervisor node that decides who acts next after every turn - the same `StateGraph` primitives as Day 1, arranged as a hub-and-spoke instead of a fixed chain.

## Flow
         +---------- supervisor -----------+
         |       (LLM decides who's next)  |
+--------+-----------+-----------+---------+
v        v           v           v         v

researcher analyst writer critic FINISH
| | | | v
+--------+-----------+-----------+ END
(every worker reports back to the supervisor)


## What it demonstrates

- **Structured routing** - the supervisor's decision is a `Literal["researcher","analyst","writer","critic","FINISH"]` via `with_structured_output`, so the model can't invent an agent that doesn't exist.
- **The decision moved from the edge into a node** - unlike Day 1's `quality_router` (which contained the actual logic), Day 2's `route_from_supervisor` is one line; all reasoning happens inside `supervisor_node`.
- **Two independent guardrails, layered** - a hard turn cap (`MAX_TURNS`) and a revision cap (`MAX_REVISIONS`) that only applies once a draft exists. Same lesson as Day 1's iteration cap, applied to an LLM-driven router instead of a scripted one: the LLM proposes, the code disposes.
- **Sharp persona boundaries** - every system prompt states what the role does *and* explicitly what it must not do (e.g., the researcher never analyzes), so responsibilities don't blur across agents.
- **A state field designed to be reset** - the writer clears `critique` after acting on it, so the supervisor's next status summary doesn't see stale, already-addressed feedback.
- **Status summaries, not full content** - the supervisor sees `"draft: filled/EMPTY"`, never the draft's actual text, to keep routing decisions cheap and fast.

## File

`my_team_agent.py` - the completed implementation (skeleton was `day2_lab_skeleton.py`).

## Running it

```bash
uv sync
cp .env.example .env   # add OPENAI_API_KEY (OpenRouter) + TAVILY_API_KEY
                        # or set USE_FAKE=1 to run with no keys
uv run python my_team_agent.py
```

Prints the Mermaid diagram (should render as a star, supervisor at the center), the final draft, turn/revision stats, and the full execution log.

## When NOT to use this pattern

A fixed pipeline with a checkable quality bar (Day 1's design) is cheaper, faster, and easier to debug than a multi-agent supervisor. Coordination overhead - more LLM calls, more latency, more failure surface - has to be worth it; it isn't free.
