import asyncio
import importlib

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentlab import tools
from agentlab.fakes import ScriptedChatModel, tool_call

lesson = importlib.import_module("10_middleware_and_streaming")

SEARCH = tool_call("search_openings", {"keyword": "engineer", "market": "SG"})


class FakeRates:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"rates": {"IDR": 12000.0}, "date": "2026-10-14"}


@pytest.fixture(autouse=True)
def offline_rates(monkeypatch):
    monkeypatch.setattr(tools.httpx, "get", lambda *a, **k: FakeRates())


def build(script, summary_script=("ringkasan",)):
    timer = lesson.ToolTimer()
    agent = lesson.build(
        ScriptedChatModel(script), summary_model=ScriptedChatModel(summary_script), timer=timer
    )
    return agent, timer


def test_tool_timer_records_each_call(capsys):
    agent, timer = build([SEARCH, "Ada beberapa."])
    agent.invoke({"messages": [HumanMessage("lowongan SG?")]})
    assert [c["tool"] for c in timer.calls] == ["search_openings"]
    assert timer.calls[0]["seconds"] >= 0
    assert "[timer] search_openings took" in capsys.readouterr().out


def test_tool_timer_works_under_astream_events():
    agent, timer = build([SEARCH, "Ada beberapa."])

    async def collect():
        return [
            e["event"]
            async for e in agent.astream_events(
                {"messages": [HumanMessage("lowongan SG?")]}, version="v2"
            )
        ]

    events = asyncio.run(collect())
    assert "on_tool_end" in events and "on_chat_model_stream" in events
    assert [c["tool"] for c in timer.calls] == ["search_openings"]


def test_convert_limit_ends_the_run():
    # The model would convert forever; the thread limit stops it after CONVERT_LIMIT calls
    # and, because exit_behavior is "end", the scripted final answer is never requested.
    converts = [
        tool_call("convert_to_idr", {"amount": 1, "currency": "SGD"}, f"c{i}") for i in range(6)
    ]
    agent, timer = build([*converts, "never reached"])
    msgs = agent.invoke({"messages": [HumanMessage("rupiahin")]})["messages"]
    assert len(timer.calls) == lesson.CONVERT_LIMIT
    blocked = [m for m in msgs if isinstance(m, ToolMessage) and m.status == "error"]
    assert len(blocked) == 1 and "limit" in blocked[0].content
    assert isinstance(msgs[-1], AIMessage) and "limit" in msgs[-1].content
    assert "never reached" not in msgs[-1].content


def test_summarization_fires_on_small_history():
    searches = [tool_call("search_openings", {"keyword": "go"}, f"s{i}") for i in range(3)]
    agent, _ = build([*searches, "Udah dirangkum."], summary_script=["ringkasan singkat"])
    msgs = agent.invoke({"messages": [HumanMessage("cari go")]})["messages"]
    summaries = [m for m in msgs if m.additional_kwargs.get("lc_source") == "summarization"]
    assert len(summaries) == 1 and "ringkasan singkat" in summaries[0].content
    # The original question was folded into the summary; the last tool pair survived.
    assert msgs[0] is not None and msgs[0].type == "human" and "summary" in msgs[0].content
    assert [m.type for m in msgs[1:]] == ["ai", "tool", "ai"]


def test_updates_stream_yields_one_update_per_node():
    agent, _ = build([SEARCH, "Ada beberapa."])
    updates = list(
        agent.stream({"messages": [HumanMessage("lowongan SG?")]}, stream_mode="updates")
    )
    assert all(len(u) == 1 for u in updates)
    nodes = [next(iter(u)) for u in updates]
    assert nodes.count("model") == 2 and nodes.count("tools") == 1
    assert "SummarizationMiddleware.before_model" in nodes
    assert "ToolCallLimitMiddleware[convert_to_idr].after_model" in nodes


def test_messages_stream_yields_text_tokens():
    agent, _ = build([SEARCH, "Ada beberapa lowongan."])
    tokens = [
        chunk.text
        for chunk, _ in agent.stream(
            {"messages": [HumanMessage("lowongan SG?")]}, stream_mode="messages"
        )
        if chunk.type == "AIMessageChunk" and chunk.text
    ]
    assert tokens == ["Ada", " ", "beberapa", " ", "lowongan."]
