import os
import operator
from datetime import datetime
from typing import Annotated, List, Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

MAX_REVISIONS = 2
MAX_TURNS = 12

USE_FAKE = os.getenv("USE_FAKE") == "1"


class TeamState(TypedDict):
    task: str
    research_notes: Annotated[List[str], operator.add]
    analysis: str
    draft: str
    critique: str
    revision_count: int
    turn_count: int
    next_agent: str
    execution_logs: Annotated[List[str], operator.add]


class RouterDecision(BaseModel):
    """The supervisor's choice of who acts next."""
    next_agent: Literal["researcher", "analyst", "writer", "critic", "FINISH"]
    reason: str = Field(description="One sentence explaining the choice")


PERSONAS = {
    "researcher": (
        "You are a Researcher on an enterprise research team. "
        "Your ONLY job is to gather and condense factual information "
        "from search results into clear, well-organized notes. "
        "You do NOT analyze, interpret, or draw conclusions - that is "
        "the analyst's job. You do NOT write reports - that is the "
        "writer's job. Stick strictly to summarizing what the sources say."
    ),
    "analyst": (
        "You are an Analyst on an enterprise research team. "
        "Your job is to take raw research notes and extract insights: "
        "patterns, implications, risks, and opportunities. "
        "You do NOT search the web yourself - you only work with the "
        "notes given to you. You do NOT write the final report - that "
        "is the writer's job. Produce a structured analysis, not prose."
    ),
    "writer": (
        "You are a Writer on an enterprise research team. "
        "Your job is to turn the analysis into a clear, professional "
        "report with an executive summary, key findings, and a "
        "conclusion. If you are given a previous draft AND a critique, "
        "revise the draft to address EVERY point in the critique - "
        "do not ignore feedback. You do NOT do research or analysis "
        "yourself - you only write based on what's given to you."
    ),
    "critic": (
        "You are a Critic on an enterprise research team. Your job is "
        "to review a draft report against the research notes and "
        "analysis, and judge whether it is accurate, complete, and "
        "well-written. If the draft is good, respond with exactly: "
        "'APPROVED'. If it needs work, respond with 'REVISE: ' "
        "followed by specific, actionable fixes. You do NOT rewrite "
        "the draft yourself - only critique it. You do NOT search "
        "the web."
    ),
}

llm = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
)

search_tool = TavilySearch(max_results=4)

supervisor_llm = llm.with_structured_output(RouterDecision)


def run_persona(role: str, user_content: str) -> str:
    """Invoke the LLM wearing a specific persona's system prompt."""
    response = llm.invoke([
        SystemMessage(PERSONAS[role]),
        HumanMessage(user_content),
    ])
    return response.content


def fake_search_results(query: str):
    return [
        {"title": f"Source on {query} #1", "content": f"Key findings about {query}, part one.", "url": "https://example.com/1"},
        {"title": f"Source on {query} #2", "content": f"Key findings about {query}, part two.", "url": "https://example.com/2"},
    ]


def supervisor_node(state: TeamState):
    """Decide who acts next, with hard guardrails the LLM can't override."""
    turn_count = state["turn_count"] + 1

    status = f"""Task: {state['task']}

Status:
- research_notes: {'filled (' + str(len(state['research_notes'])) + ' notes)' if state['research_notes'] else 'EMPTY'}
- analysis: {'filled' if state['analysis'] else 'EMPTY'}
- draft: {'filled' if state['draft'] else 'EMPTY'}
- critique: {state['critique'] if state['critique'] else 'none yet'}
- revision_count: {state['revision_count']}/{MAX_REVISIONS}
- turn_count: {turn_count}/{MAX_TURNS}

Decide which agent acts next: researcher, analyst, writer, critic, or FINISH.
Typical order: researcher -> analyst -> writer -> critic -> (writer again if REVISE) -> FINISH."""

    decision = supervisor_llm.invoke(status)
    next_agent = decision.next_agent
    log_line = f"[supervisor] turn {turn_count}: chose {next_agent} - {decision.reason}"

    if turn_count > MAX_TURNS:
        next_agent = "FINISH"
        log_line = f"[supervisor] turn {turn_count}: FORCED FINISH (turn cap reached)"
    elif next_agent in ("writer", "critic") and state["revision_count"] >= MAX_REVISIONS and state["draft"]:
        next_agent = "FINISH"
        log_line = f"[supervisor] turn {turn_count}: FORCED FINISH (revision cap reached)"

    return {
        "next_agent": next_agent,
        "turn_count": turn_count,
        "execution_logs": [log_line],
    }


def researcher_node(state: TeamState):
    """Search the web (ONLY this agent may), condense to notes."""
    if USE_FAKE:
        results = fake_search_results(state["task"])
    else:
        results = search_tool.invoke({"query": state["task"]})["results"]

    raw = "\n\n".join(
        f"Title: {r['title']}\nContent: {r['content']}\nURL: {r.get('url', 'N/A')}"
        for r in results
    )

    notes = run_persona(
        "researcher",
        f"Task: {state['task']}\n\nSearch results:\n{raw}",
    )

    return {
        "research_notes": [notes],
        "execution_logs": [f"[researcher] condensed {len(results)} sources into notes"],
    }


def analyst_node(state: TeamState):
    """Turn raw notes into analysis."""
    notes_text = "\n\n".join(state["research_notes"])

    analysis = run_persona(
        "analyst",
        f"Task: {state['task']}\n\nResearch notes:\n{notes_text}",
    )

    return {
        "analysis": analysis,
        "execution_logs": ["[analyst] produced analysis"],
    }


def writer_node(state: TeamState):
    """Write the draft - or revise it if a critique is present."""
    revising = bool(state["critique"]) and state["critique"].startswith("REVISE")

    if revising:
        prompt = (
            f"Task: {state['task']}\n\n"
            f"Analysis:\n{state['analysis']}\n\n"
            f"Previous draft:\n{state['draft']}\n\n"
            f"Critique to address:\n{state['critique']}\n\n"
            f"Revise the draft to fix every point in the critique."
        )
    else:
        prompt = f"Task: {state['task']}\n\nAnalysis:\n{state['analysis']}\n\nWrite the report."

    draft = run_persona("writer", prompt)

    return {
        "draft": draft,
        "critique": "",
        "revision_count": state["revision_count"] + 1 if revising else state["revision_count"],
        "execution_logs": [f"[writer] {'revised' if revising else 'wrote initial'} draft"],
    }


def critic_node(state: TeamState):
    """Review the draft against the research notes and analysis."""
    notes_text = "\n\n".join(state["research_notes"])

    verdict = run_persona(
        "critic",
        f"Task: {state['task']}\n\nResearch notes:\n{notes_text}\n\n"
        f"Analysis:\n{state['analysis']}\n\nDraft to review:\n{state['draft']}",
    )

    return {
        "critique": verdict,
        "execution_logs": [f"[critic] verdict: {verdict[:60]}..."],
    }


def route_from_supervisor(state: TeamState) -> str:
    """Just reads the supervisor's decision - no logic here."""
    return state["next_agent"]


workflow = StateGraph(TeamState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("writer", writer_node)
workflow.add_node("critic", critic_node)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        "critic": "critic",
        "FINISH": END,
    },
)

for worker in ["researcher", "analyst", "writer", "critic"]:
    workflow.add_edge(worker, "supervisor")


if __name__ == "__main__":
    app = workflow.compile(checkpointer=InMemorySaver())

    print("=== MERMAID DIAGRAM (paste into mermaid.live) ===")
    print(app.get_graph().draw_mermaid())
    print()

    initial_state = {
        "task": "Should our company adopt multi-agent AI systems in 2026?",
        "research_notes": [],
        "analysis": "",
        "draft": "",
        "critique": "",
        "revision_count": 0,
        "turn_count": 0,
        "next_agent": "",
        "execution_logs": [],
    }

    config = {"configurable": {"thread_id": "run-1"}}

    for chunk in app.stream(initial_state, config, stream_mode="values"):
        pass

    final_state = chunk

    print("=== FINAL DRAFT ===")
    print(final_state["draft"])
    print()

    print("=== STATS ===")
    print(f"Turns used: {final_state['turn_count']}/{MAX_TURNS}")
    print(f"Revisions: {final_state['revision_count']}/{MAX_REVISIONS}")
    print()

    print("=== EXECUTION LOGS ===")
    for log in final_state["execution_logs"]:
        print(f"  {log}")
