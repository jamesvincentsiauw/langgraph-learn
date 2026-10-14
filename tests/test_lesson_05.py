import importlib

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agentlab import data
from agentlab.fakes import ScriptedChatModel, tool_call

lesson = importlib.import_module("05_human_in_the_loop")

KEY = "smartrecruiters:Grab/744000151028388"
SCRIPT = [tool_call("save_to_shortlist", {"key": KEY, "note": "backend, SG"}), "Oke."]
CONFIG = {"configurable": {"thread_id": "t1"}}


@pytest.fixture(params=[False, True], ids=["hand_built", "middleware"])
def variant(request):
    return request.param


def start(middleware: bool):
    agent = lesson.build(ScriptedChatModel(SCRIPT), middleware=middleware)
    result = agent.invoke({"messages": [HumanMessage("simpan Grab")]}, CONFIG)
    return agent, result


def test_graph_stops_with_an_interrupt_before_writing(variant):
    agent, result = start(variant)
    assert "__interrupt__" in result
    assert lesson.pending_call(result["__interrupt__"][0].value, variant) == (
        "save_to_shortlist",
        {"key": KEY, "note": "backend, SG"},
    )
    state = agent.get_state(CONFIG)
    assert state.next and state.interrupts
    assert [m.type for m in state.values["messages"]] == ["human", "ai"]
    assert data.read_shortlist() == []


def test_approve_runs_the_tool(variant):
    agent, _ = start(variant)
    payload = lesson.resume_payload(True, "", variant)
    result = agent.invoke(Command(resume=payload), CONFIG)
    assert "__interrupt__" not in result
    assert [m.type for m in result["messages"]] == ["human", "ai", "tool", "ai"]
    assert result["messages"][2].content.startswith(f"Saved {KEY}")
    assert [e["key"] for e in data.read_shortlist()] == [KEY]


def test_reject_skips_the_tool_and_tells_the_model(variant):
    agent, _ = start(variant)
    payload = lesson.resume_payload(False, "gaji nggak disebut", variant)
    result = agent.invoke(Command(resume=payload), CONFIG)
    tool_msg = result["messages"][2]
    assert tool_msg.type == "tool" and tool_msg.status == "error"
    assert "gaji nggak disebut" in tool_msg.content
    assert result["messages"][-1].content == "Oke."
    assert data.read_shortlist() == []


def test_ungated_tools_run_without_asking(variant):
    script = [tool_call("search_openings", {"keyword": "backend", "market": "SG"}), "Ada."]
    agent = lesson.build(ScriptedChatModel(script), middleware=variant)
    result = agent.invoke({"messages": [HumanMessage("cari")]}, CONFIG)
    assert "__interrupt__" not in result
    assert [m.type for m in result["messages"]] == ["human", "ai", "tool", "ai"]


def test_hand_built_decides_gated_calls_before_running_any_tool():
    # Both calls arrive in one model turn. On resume the tools node re-runs from the top, so
    # the search must not have executed before the interrupt, or it would run twice.
    ai = tool_call("search_openings", {"keyword": "backend"})
    ai.tool_calls.append(
        {"name": "save_to_shortlist", "args": {"key": KEY, "note": "n"}, "id": "c2"}
    )
    agent = lesson.build(ScriptedChatModel([ai, "Oke."]))
    first = agent.invoke({"messages": [HumanMessage("x")]}, CONFIG)
    assert "__interrupt__" in first
    result = agent.invoke(Command(resume={"action": "approve"}), CONFIG)
    assert [m.type for m in result["messages"]] == ["human", "ai", "tool", "tool", "ai"]
