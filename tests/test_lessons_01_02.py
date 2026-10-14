import importlib

from langchain_core.messages import HumanMessage

from agentlab.fakes import ScriptedChatModel, tool_call

SCRIPT = [tool_call("search_openings", {"keyword": "engineer", "market": "SG"}), "Ada beberapa."]


def run(lesson: str):
    agent = importlib.import_module(lesson).build(ScriptedChatModel(SCRIPT))
    return agent.invoke({"messages": [HumanMessage("lowongan SG?")]})["messages"]


def test_hand_built_agent_loops_through_tools():
    msgs = run("01_hand_built_agent")
    assert [m.type for m in msgs] == ["human", "ai", "tool", "ai"]
    assert msgs[-1].content == "Ada beberapa."


def test_create_agent_matches_hand_built():
    msgs = run("02_create_agent")
    assert [m.type for m in msgs] == ["human", "ai", "tool", "ai"]
