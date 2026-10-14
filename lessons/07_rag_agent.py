"""Lesson 7: agentic RAG over the job descriptions.

The graph retrieves chunks with BM25, asks the model whether they are enough to answer, and
if not rewrites the query with the grader's suggestion and retrieves again. A cap on attempts
keeps the loop finite. The answer must cite company and title and say what the chunks lack.

Run: uv run lessons/07_rag_agent.py "Lowongan mana yang nyebutin Kafka atau event-driven?"
     uv run lessons/07_rag_agent.py --graph
"""

import sys
from collections.abc import Callable
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentlab import retrieval
from agentlab.config import make_model

DEFAULT_QUESTION = "Lowongan mana yang nyebutin Kafka atau event-driven?"
MAX_ATTEMPTS = 2
CHUNKS_PER_QUERY = 4

ANSWER_SYSTEM = (
    "You answer questions about job openings using only the chunks given. Name the company "
    "and title for every claim. When the chunks do not mention what the question asks for, "
    "say so plainly instead of guessing. Answer in casual Indonesian, briefly."
)


class Grade(BaseModel):
    enough: bool = Field(description="True when the chunks contain what the question needs")
    better_query: str = Field(
        description="Short English keyword query that would find better chunks"
    )


class State(TypedDict):
    question: str
    query: str
    docs: list[Document]
    grade: Grade | None
    answer: str
    attempts: int


Retriever = Callable[[str, int], list[Document]]


def render(docs: list[Document]) -> str:
    if not docs:
        return "(no chunks matched)"
    return "\n\n".join(
        f"[{d.metadata['company']}: {d.metadata['title']}]\n{d.page_content}" for d in docs
    )


def build(model: BaseChatModel | None = None, retrieve: Retriever = retrieval.retrieve):
    model = model or make_model()
    grader = model.with_structured_output(Grade)

    def retrieve_chunks(state: State) -> dict:
        query = state.get("query") or state["question"]
        docs = retrieve(query, CHUNKS_PER_QUERY)
        return {"query": query, "docs": docs, "attempts": state.get("attempts", 0) + 1}

    def grade(state: State) -> dict:
        prompt = (
            f"Question: {state['question']}\n\nChunks:\n{render(state['docs'])}\n\n"
            "Are these chunks enough to answer the question? The descriptions are in English, "
            "so the better query must be English keywords."
        )
        return {"grade": grader.invoke(prompt)}

    def route(state: State) -> str:
        if state["grade"].enough or state["attempts"] >= MAX_ATTEMPTS:
            return "answer"
        return "rewrite"

    def rewrite(state: State) -> dict:
        return {"query": state["grade"].better_query}

    def answer(state: State) -> dict:
        reply = model.invoke(
            [
                SystemMessage(ANSWER_SYSTEM),
                HumanMessage(f"Question: {state['question']}\n\nChunks:\n{render(state['docs'])}"),
            ]
        )
        return {"answer": str(reply.text)}

    builder = StateGraph(State)
    builder.add_node("retrieve", retrieve_chunks)
    builder.add_node("grade", grade)
    builder.add_node("rewrite", rewrite)
    builder.add_node("answer", answer)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", route, ["answer", "rewrite"])
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("answer", END)
    return builder.compile()


if __name__ == "__main__":
    graph = build()
    if "--graph" in sys.argv:
        print(graph.get_graph().draw_mermaid())
        sys.exit()
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    for update in graph.stream({"question": question}, stream_mode="updates"):
        for node, out in update.items():
            if node == "retrieve":
                hits = ", ".join(d.metadata["company"] for d in out["docs"]) or "nothing"
                print(f"[retrieve] attempt {out['attempts']} {out['query']!r} -> {hits}")
            elif node == "grade":
                g = out["grade"]
                print(f"[grade] enough={g.enough} better_query={g.better_query!r}")
            elif node == "rewrite":
                print(f"[rewrite] query -> {out['query']!r}")
            elif node == "answer":
                print(f"[answer]\n{out['answer']}")
