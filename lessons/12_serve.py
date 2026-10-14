"""Lesson 12: the agent behind a FastAPI endpoint with SSE streaming and SQLite threads.

POST /threads/{thread_id}/messages takes {"text": ...} and streams the reply as server-sent
events: "tool" when the model calls a tool and when the tool answers, "token" for each text
chunk, "done" with the final text. GET /threads/{thread_id} returns the history from the
SqliteSaver checkpoint, so threads survive a server restart. The agent is the lesson-4
agent; the only new code is turning one graph stream into HTTP events.

Run: uv run lessons/12_serve.py                 (serves on http://127.0.0.1:8000)
     uv run lessons/12_serve.py --graph
     curl -N -X POST localhost:8000/threads/t1/messages \\
          -H 'content-type: application/json' -d '{"text": "Ada lowongan golang remote?"}'
     curl localhost:8000/threads/t1
"""

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agentlab.checkpoints import THREADS_PATH, open_sqlite_saver
from agentlab.config import SYSTEM, make_model
from agentlab.tools import TOOLS


class MessageIn(BaseModel):
    text: str


def build(model: BaseChatModel | None = None, checkpointer: BaseCheckpointSaver | None = None):
    return create_agent(
        model or make_model(),
        tools=TOOLS,
        system_prompt=SYSTEM,
        checkpointer=checkpointer or open_sqlite_saver(),
    )


def thread(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def serialize(msg: BaseMessage) -> dict:
    out = {"type": msg.type, "text": msg.text}
    if isinstance(msg, AIMessage) and msg.tool_calls:
        out["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in msg.tool_calls]
    if isinstance(msg, ToolMessage):
        out["name"] = msg.name
    return out


def events(agent, thread_id: str, text: str) -> Iterator[dict]:
    """One graph stream in two modes: "messages" gives tokens, "updates" gives whole messages."""
    final = ""
    stream = agent.stream(
        {"messages": [HumanMessage(text)]}, thread(thread_id), stream_mode=["updates", "messages"]
    )
    for mode, payload in stream:
        if mode == "messages":
            chunk, _metadata = payload
            if isinstance(chunk, AIMessageChunk) and chunk.text:
                yield {"event": "token", "data": json.dumps({"text": chunk.text})}
            continue
        for update in payload.values():
            for msg in (update or {}).get("messages", []):
                if isinstance(msg, ToolMessage):
                    data = {"name": msg.name, "result": str(msg.content)[:200]}
                    yield {"event": "tool", "data": json.dumps(data)}
                elif isinstance(msg, AIMessage) and msg.tool_calls:
                    for call in msg.tool_calls:
                        data = {"name": call["name"], "args": call["args"]}
                        yield {"event": "tool", "data": json.dumps(data)}
                elif isinstance(msg, AIMessage):
                    final = msg.text
    yield {"event": "done", "data": json.dumps({"text": final})}


def build_app(model: BaseChatModel | None = None, db_path: Path = THREADS_PATH) -> FastAPI:
    agent = build(model, checkpointer=open_sqlite_saver(db_path))
    app = FastAPI(title="langgraph-learn lesson 12")

    @app.post("/threads/{thread_id}/messages")
    def post_message(thread_id: str, body: MessageIn) -> EventSourceResponse:
        # A sync generator is fine here: sse-starlette iterates it on a worker thread.
        return EventSourceResponse(events(agent, thread_id, body.text))

    @app.get("/threads/{thread_id}")
    def get_thread(thread_id: str) -> dict:
        messages = agent.get_state(thread(thread_id)).values.get("messages", [])
        return {"thread_id": thread_id, "messages": [serialize(m) for m in messages]}

    return app


if __name__ == "__main__":
    if "--graph" in sys.argv:
        print(build().get_graph().draw_mermaid())
        sys.exit()
    print(f"threads stored in {THREADS_PATH}")
    uvicorn.run(build_app(), host="127.0.0.1", port=8000)
