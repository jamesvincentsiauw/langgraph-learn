"""Lesson 4: checkpointers and thread_id, so a follow-up question has the earlier turns.

Without a checkpointer every invoke starts from an empty state. With one, LangGraph saves
the state after every step under config["configurable"]["thread_id"], and the next invoke
on the same thread continues from it. InMemorySaver lives for one process; SqliteSaver
keeps threads across runs in data/threads.sqlite.

Run: uv run lessons/04_memory.py
     uv run lessons/04_memory.py --sqlite      (run it twice: the thread keeps growing)
     uv run lessons/04_memory.py --graph
"""

import sys

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from agentlab.checkpoints import THREADS_PATH, open_sqlite_saver
from agentlab.config import SYSTEM, make_model
from agentlab.display import print_updates
from agentlab.tools import TOOLS

FIRST = "Ada lowongan backend di SG? Sebutin 2 aja, singkat."
FOLLOW_UP = "Yang pertama apply-nya di mana?"


def build(model: BaseChatModel | None = None, checkpointer: BaseCheckpointSaver | None = None):
    return create_agent(
        model or make_model(),
        tools=TOOLS,
        system_prompt=SYSTEM,
        checkpointer=checkpointer or InMemorySaver(),
    )


def thread(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def ask(agent, thread_id: str, question: str) -> None:
    print(f"\n[{thread_id}] > {question}")
    print_updates(agent.stream({"messages": [HumanMessage(question)]}, thread(thread_id)))


def describe_state(agent, thread_id: str) -> None:
    """What the checkpoint holds for a thread: the state values plus a few metadata fields."""
    state = agent.get_state(thread(thread_id))
    n_messages = len(state.values.get("messages", []))
    n_checkpoints = sum(1 for _ in agent.get_state_history(thread(thread_id)))
    print(f"[{thread_id}] {n_messages} messages, {n_checkpoints} checkpoints, next={state.next}")


if __name__ == "__main__":
    use_sqlite = "--sqlite" in sys.argv
    agent = build(checkpointer=open_sqlite_saver() if use_sqlite else None)
    if "--graph" in sys.argv:
        print(agent.get_graph().draw_mermaid())
        sys.exit()
    if use_sqlite:
        print(f"threads stored in {THREADS_PATH}")
    describe_state(agent, "demo-1")
    ask(agent, "demo-1", FIRST)
    ask(agent, "demo-1", FOLLOW_UP)
    describe_state(agent, "demo-1")
    # A fresh thread shares nothing with demo-1, so the same follow-up has no context.
    ask(agent, "demo-2", FOLLOW_UP)
    describe_state(agent, "demo-2")
