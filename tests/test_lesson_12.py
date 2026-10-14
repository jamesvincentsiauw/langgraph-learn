import importlib
import json

import pytest
from fastapi.testclient import TestClient

from agentlab.fakes import ScriptedChatModel, tool_call

lesson = importlib.import_module("12_serve")

SCRIPT = [
    tool_call("search_openings", {"keyword": "golang", "market": "REMOTE"}),
    "Ada dua: Startup Talents dan YLD.",
    "Yang pertama remote.",
]


@pytest.fixture
def served(tmp_path):
    model = ScriptedChatModel(SCRIPT)
    app = lesson.build_app(model, db_path=tmp_path / "threads.sqlite")
    with TestClient(app) as client:
        yield client, model


def post(client: TestClient, thread_id: str, text: str) -> list[tuple[str, dict]]:
    """Collect (event, data) pairs from one SSE response."""
    events, name = [], None
    with client.stream("POST", f"/threads/{thread_id}/messages", json={"text": text}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        for line in resp.iter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and name:
                events.append((name, json.loads(line.split(":", 1)[1].strip())))
                name = None
    return events


def test_post_streams_tool_tokens_then_done(served):
    client, _ = served
    events = post(client, "t1", "Ada lowongan golang remote?")
    names = [e for e, _ in events]
    assert names[0] == "tool" and names[-1] == "done"
    assert events[0][1] == {
        "name": "search_openings",
        "args": {"keyword": "golang", "market": "REMOTE"},
    }
    assert events[1][1]["name"] == "search_openings" and "Startup Talents" in events[1][1]["result"]
    tokens = "".join(d["text"] for e, d in events if e == "token")
    assert tokens == "Ada dua: Startup Talents dan YLD."
    assert events[-1][1] == {"text": "Ada dua: Startup Talents dan YLD."}


def test_second_post_on_same_thread_sees_history(served):
    client, model = served
    post(client, "t1", "Ada lowongan golang remote?")
    events = post(client, "t1", "Yang pertama remote gak?")
    assert events[-1] == ("done", {"text": "Yang pertama remote."})
    # The third model call received the whole thread, not just the new question.
    history = [m.type for m in model.seen[-1]]
    assert history == ["system", "human", "ai", "tool", "ai", "human"]


def test_get_thread_returns_checkpointed_messages(served):
    client, _ = served
    assert client.get("/threads/empty").json() == {"thread_id": "empty", "messages": []}
    post(client, "t1", "Ada lowongan golang remote?")
    body = client.get("/threads/t1").json()
    assert [m["type"] for m in body["messages"]] == ["human", "ai", "tool", "ai"]
    assert body["messages"][1]["tool_calls"][0]["name"] == "search_openings"
    assert body["messages"][2]["name"] == "search_openings"
    assert body["messages"][-1]["text"] == "Ada dua: Startup Talents dan YLD."


def test_threads_survive_a_new_app_on_the_same_db(tmp_path):
    db = tmp_path / "threads.sqlite"
    with TestClient(lesson.build_app(ScriptedChatModel(SCRIPT), db_path=db)) as client:
        post(client, "t1", "Ada lowongan golang remote?")
    with TestClient(lesson.build_app(ScriptedChatModel(["lagi"]), db_path=db)) as client:
        assert len(client.get("/threads/t1").json()["messages"]) == 4
