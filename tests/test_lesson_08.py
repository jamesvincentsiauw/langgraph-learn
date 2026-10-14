import importlib

from langchain_core.messages import AIMessage, HumanMessage

from agentlab.fakes import ScriptedChatModel, tool_call

lesson = importlib.import_module("08_supervisor")


def run(graph, question="lowongan golang remote?"):
    """Node names in order, and the shared message history as the nodes produced it."""
    nodes, messages = [], []
    for update in graph.stream({"messages": [HumanMessage(question)]}, stream_mode="updates"):
        for node, output in update.items():
            nodes.append(node)
            messages.extend(output["messages"])
    return nodes, messages


def test_supervisor_hands_off_and_gets_control_back():
    supervisor = ScriptedChatModel(
        [tool_call("transfer_to_job_scout", {"task": "find golang openings"}), "Ada dua."]
    )
    scout = ScriptedChatModel(
        [tool_call("search_openings", {"keyword": "golang"}), "Two openings: a, b."]
    )
    graph = lesson.build(supervisor, scout, ScriptedChatModel([]))
    nodes, messages = run(graph)
    assert nodes == ["supervisor", "job_scout", "supervisor"]
    handoff, answer, final = messages
    assert handoff.tool_calls[0]["name"] == "transfer_to_job_scout"
    assert answer.type == "tool" and answer.tool_call_id == handoff.tool_calls[0]["id"]
    assert answer.content == "Two openings: a, b."
    assert final.content == "Ada dua."
    # The specialist's own search call stays inside its subgraph.
    assert not any(getattr(m, "name", None) == "search_openings" for m in messages)


def test_supervisor_finishes_on_plain_text():
    graph = lesson.build(ScriptedChatModel(["Ga perlu cari."]), ScriptedChatModel([]), None)
    nodes, messages = run(graph)
    assert nodes == ["supervisor"]
    assert messages[-1].type == "ai" and messages[-1].content == "Ga perlu cari."


def test_two_handoffs_in_one_reply_run_both_specialists():
    both = AIMessage(
        content="",
        tool_calls=[
            {"name": "transfer_to_job_scout", "args": {"task": "search"}, "id": "c1"},
            {"name": "transfer_to_salary_analyst", "args": {"task": "convert"}, "id": "c2"},
        ],
    )
    graph = lesson.build(
        ScriptedChatModel([both, "Selesai."]),
        ScriptedChatModel(["scout reply"]),
        ScriptedChatModel(["analyst reply"]),
    )
    nodes, messages = run(graph)
    assert nodes[0] == "supervisor" and nodes[-1] == "supervisor"
    assert set(nodes[1:3]) == {"job_scout", "salary_analyst"}
    replies = {m.tool_call_id: m.content for m in messages if m.type == "tool"}
    assert replies == {"c1": "scout reply", "c2": "analyst reply"}


def test_job_scout_runs_alone():
    agent = lesson.build_job_scout(
        ScriptedChatModel([tool_call("search_openings", {"keyword": "golang"}), "done"])
    )
    msgs = agent.invoke({"messages": [HumanMessage("golang?")]})["messages"]
    assert [m.type for m in msgs] == ["human", "ai", "tool", "ai"]
    assert "backend-engineer-golang-python" in msgs[2].content


def test_salary_analyst_runs_alone():
    call = tool_call("monthly_salary", {"amount": 120000, "currency": "sgd", "period": "year"})
    agent = lesson.build_salary_analyst(ScriptedChatModel([call, "done"]))
    msgs = agent.invoke({"messages": [HumanMessage("per bulan?")]})["messages"]
    assert [m.type for m in msgs] == ["human", "ai", "tool", "ai"]
    assert "10,000 SGD per month" in msgs[2].content
