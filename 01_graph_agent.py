"""Lesson 1: a tool-calling agent built by hand as a LangGraph StateGraph.

Run: uv run 01_graph_agent.py "your question"
     uv run 01_graph_agent.py --graph   (print the graph as Mermaid)
"""

import sys
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from common import SYSTEM, TOOLS, make_model, print_updates


class State(TypedDict):
    # add_messages is a reducer: nodes return new messages and LangGraph appends them
    # to the list instead of replacing it.
    messages: Annotated[list[AnyMessage], add_messages]


model = make_model().bind_tools(TOOLS)
tools_by_name = {t.name: t for t in TOOLS}


def call_model(state: State) -> dict:
    reply = model.invoke([SystemMessage(SYSTEM), *state["messages"]])
    return {"messages": [reply]}


def run_tools(state: State) -> dict:
    results = []
    for call in state["messages"][-1].tool_calls:
        output = tools_by_name[call["name"]].invoke(call["args"])
        results.append(ToolMessage(content=output, tool_call_id=call["id"], name=call["name"]))
    return {"messages": results}


def route(state: State) -> str:
    return "tools" if state["messages"][-1].tool_calls else END


builder = StateGraph(State)
builder.add_node("model", call_model)
builder.add_node("tools", run_tools)
builder.add_edge(START, "model")
builder.add_conditional_edges("model", route, ["tools", END])
builder.add_edge("tools", "model")
agent = builder.compile()


if __name__ == "__main__":
    if "--graph" in sys.argv:
        print(agent.get_graph().draw_mermaid())
        sys.exit()
    question = " ".join(sys.argv[1:]) or "Ada lowongan engineer di Singapore yang nyebutin gaji? Rupiahin gajinya."
    print_updates(agent.stream({"messages": [HumanMessage(question)]}, stream_mode="updates"))
