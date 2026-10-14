"""Lesson 5: pause the graph before a tool that writes, and resume with a human decision.

save_to_shortlist writes to data/shortlist.json, so the graph must stop and ask before
running it. Two variants of the same gate:
  (a) hand-built: the lesson 1 StateGraph whose tools node calls interrupt() before a gated
      tool; the runner resumes with Command(resume={"action": ..., "reason": ...});
  (b) --middleware: create_agent with HumanInTheLoopMiddleware, which interrupts after the
      model node and resumes with Command(resume={"decisions": [{"type": ...}]}).
Both need a checkpointer: an interrupt is a saved state waiting for a resume.

Run: uv run lessons/05_human_in_the_loop.py "your request"
     uv run lessons/05_human_in_the_loop.py --middleware
     uv run lessons/05_human_in_the_loop.py --auto-approve   (or --auto-reject, no stdin)
     uv run lessons/05_human_in_the_loop.py --graph
"""

import sys
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from agentlab.config import SYSTEM, make_model
from agentlab.display import print_updates
from agentlab.tools import TOOLS, save_to_shortlist

DEFAULT_QUESTION = (
    "Cari lowongan backend di SG, pilih satu yang paling cocok buat gua dan simpan ke "
    "shortlist dengan catatan singkat."
)
AGENT_TOOLS = [*TOOLS, save_to_shortlist]
GATED = {save_to_shortlist.name}


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build(
    model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    middleware: bool = False,
):
    model = model or make_model()
    checkpointer = checkpointer or InMemorySaver()
    if middleware:
        return create_agent(
            model,
            tools=AGENT_TOOLS,
            system_prompt=SYSTEM,
            middleware=[HumanInTheLoopMiddleware(interrupt_on={save_to_shortlist.name: True})],
            checkpointer=checkpointer,
        )
    return build_hand_made(model, checkpointer)


def build_hand_made(model: BaseChatModel, checkpointer: BaseCheckpointSaver):
    bound = model.bind_tools(AGENT_TOOLS)
    tools_by_name = {t.name: t for t in AGENT_TOOLS}

    def call_model(state: State) -> dict:
        return {"messages": [bound.invoke([SystemMessage(SYSTEM), *state["messages"]])]}

    def run_tools(state: State) -> dict:
        calls = state["messages"][-1].tool_calls
        # A node that interrupts is re-run from its first line on resume, so decide every
        # gated call before running any tool. Otherwise the ungated ones would run twice.
        decisions = {
            c["id"]: interrupt({"tool": c["name"], "args": c["args"]})
            for c in calls
            if c["name"] in GATED
        }
        results = []
        for call in calls:
            decision = decisions.get(call["id"], {"action": "approve"})
            if decision.get("action") == "approve":
                output = tools_by_name[call["name"]].invoke(call["args"])
                results.append(ToolMessage(output, tool_call_id=call["id"], name=call["name"]))
                continue
            reason = decision.get("reason") or "no reason given"
            results.append(
                ToolMessage(
                    f"Vincent declined this {call['name']} call: {reason}. Do not retry it.",
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )
            )
        return {"messages": results}

    def route(state: State) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    builder = StateGraph(State)
    builder.add_node("model", call_model)
    builder.add_node("tools", run_tools)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route, ["tools", END])
    builder.add_edge("tools", "model")
    return builder.compile(checkpointer=checkpointer)


def resume_payload(approve: bool, reason: str, middleware: bool) -> dict:
    """The two variants read different shapes out of Command(resume=...)."""
    if middleware:
        decision = {"type": "approve"} if approve else {"type": "reject", "message": reason}
        return {"decisions": [decision]}
    return {"action": "approve"} if approve else {"action": "reject", "reason": reason}


def pending_call(value: dict, middleware: bool) -> tuple[str, dict]:
    """Tool name and args out of an interrupt value, whichever variant raised it."""
    if middleware:
        request = value["action_requests"][0]
        return request["name"], request["args"]
    return value["tool"], value["args"]


def decide(args: list[str]) -> tuple[bool, str]:
    if "--auto-approve" in args:
        return True, ""
    if "--auto-reject" in args:
        return False, "sample run, not a real pick"
    answer = input("approve? [y/N] ").strip().lower()
    if answer == "y":
        return True, ""
    return False, input("reason: ").strip()


if __name__ == "__main__":
    use_middleware = "--middleware" in sys.argv
    agent = build(middleware=use_middleware)
    if "--graph" in sys.argv:
        print(agent.get_graph().draw_mermaid())
        sys.exit()
    words = [a for a in sys.argv[1:] if not a.startswith("--")]
    question = " ".join(words) or DEFAULT_QUESTION
    config = {"configurable": {"thread_id": "shortlist-demo"}}

    print_updates(agent.stream({"messages": [HumanMessage(question)]}, config))
    while interrupts := agent.get_state(config).interrupts:
        name, call_args = pending_call(interrupts[0].value, use_middleware)
        print(f"\n[paused] {name}({call_args})")
        approve, reason = decide(sys.argv)
        print(f"[resume] {'approve' if approve else f'reject: {reason}'}\n")
        payload = resume_payload(approve, reason, use_middleware)
        print_updates(agent.stream(Command(resume=payload), config))
