import importlib

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agentlab.checkpoints import open_sqlite_saver
from agentlab.fakes import ScriptedChatModel, tool_call

lesson = importlib.import_module("04_memory")

SCRIPT = [
    tool_call("search_openings", {"keyword": "backend", "market": "SG"}),
    "Ada Delivery Hero dan Grab.",
    "Delivery Hero: link apply ada di tool output tadi.",
    "Yang pertama apa ya? Belum ada konteks.",
]


def test_scripted_model_records_what_it_was_sent():
    model = ScriptedChatModel(["hi"])
    model.invoke([HumanMessage("a")])
    assert [m.content for m in model.seen[0]] == ["a"]
    assert ScriptedChatModel(["x"]).seen == []


def test_same_thread_carries_history_into_the_next_model_call():
    model = ScriptedChatModel(SCRIPT)
    agent = lesson.build(model)
    cfg = lesson.thread("t1")
    agent.invoke({"messages": [HumanMessage(lesson.FIRST)]}, cfg)
    result = agent.invoke({"messages": [HumanMessage(lesson.FOLLOW_UP)]}, cfg)

    assert [m.type for m in result["messages"]] == ["human", "ai", "tool", "ai", "human", "ai"]
    # The third model call is the follow-up; it must see the whole earlier turn, not just
    # the new question. The system prompt is prepended, so skip it.
    third_prompt = [m for m in model.seen[2] if m.type != "system"]
    assert [m.type for m in third_prompt] == ["human", "ai", "tool", "ai", "human"]
    assert third_prompt[-1].content == lesson.FOLLOW_UP


def test_other_thread_starts_empty():
    model = ScriptedChatModel(SCRIPT)
    agent = lesson.build(model)
    agent.invoke({"messages": [HumanMessage(lesson.FIRST)]}, lesson.thread("t1"))
    agent.invoke({"messages": [HumanMessage(lesson.FOLLOW_UP)]}, lesson.thread("t1"))
    result = agent.invoke({"messages": [HumanMessage(lesson.FOLLOW_UP)]}, lesson.thread("t2"))

    assert [m.type for m in result["messages"]] == ["human", "ai"]
    assert [m.type for m in model.seen[3] if m.type != "system"] == ["human"]


def test_get_state_exposes_the_checkpoint():
    agent = lesson.build(ScriptedChatModel(SCRIPT))
    cfg = lesson.thread("t1")
    assert agent.get_state(cfg).values == {}
    agent.invoke({"messages": [HumanMessage(lesson.FIRST)]}, cfg)
    state = agent.get_state(cfg)
    assert len(state.values["messages"]) == 4
    assert state.next == ()
    assert sum(1 for _ in agent.get_state_history(cfg)) > 1


def test_sqlite_saver_keeps_the_thread_across_agents(tmp_path):
    path = tmp_path / "threads.sqlite"
    saver = open_sqlite_saver(path)
    assert isinstance(saver, SqliteSaver)
    cfg = lesson.thread("t1")
    lesson.build(ScriptedChatModel(SCRIPT), saver).invoke(
        {"messages": [HumanMessage(lesson.FIRST)]}, cfg
    )
    saver.conn.close()

    # A new process would do exactly this: new saver over the same file, same thread_id.
    again = lesson.build(ScriptedChatModel(SCRIPT[2:]), open_sqlite_saver(path))
    assert len(again.get_state(cfg).values["messages"]) == 4
    result = again.invoke({"messages": [HumanMessage(lesson.FOLLOW_UP)]}, cfg)
    assert len(result["messages"]) == 6
