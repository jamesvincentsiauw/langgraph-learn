import importlib

from langchain_core.documents import Document

from agentlab import retrieval
from agentlab.fakes import ScriptedChatModel

lesson = importlib.import_module("07_rag_agent")

NO = '{"enough": false, "better_query": "event streams processing"}'
YES = '{"enough": true, "better_query": ""}'


def fake_retrieve(query: str, k: int) -> list[Document]:
    meta = {"key": "k1", "company": "Airship", "title": "Backend Java Engineer"}
    return [Document(page_content=f"chunk for {query}", metadata=meta)]


def run(script: list[str], retrieve=fake_retrieve):
    graph = lesson.build(ScriptedChatModel(script), retrieve=retrieve)
    nodes, state = [], {}
    for update in graph.stream({"question": "Kafka?"}, stream_mode="updates"):
        for node, out in update.items():
            nodes.append(node)
            state.update(out)
    return nodes, state


def test_retriever_returns_chunks_mentioning_the_term():
    docs = retrieval.retrieve("event streams", k=3)
    assert docs
    assert all("event" in d.page_content.lower() for d in docs)
    assert {"key", "company", "title"} <= docs[0].metadata.keys()


def test_retriever_returns_nothing_when_no_chunk_matches():
    assert retrieval.retrieve("kafka", k=3) == []


def test_chunks_respect_size_and_keep_words_whole():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = retrieval.chunk_text(text, size=100)
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks) == text


def test_graph_answers_right_away_when_grade_says_yes():
    nodes, state = run([YES, "Airship."])
    assert nodes == ["retrieve", "grade", "answer"]
    assert state["attempts"] == 1
    assert state["answer"] == "Airship."


def test_graph_loops_once_when_grade_says_no():
    nodes, state = run([NO, YES, "Airship nyebutin event streams."])
    assert nodes == ["retrieve", "grade", "rewrite", "retrieve", "grade", "answer"]
    assert state["query"] == "event streams processing"
    assert state["docs"][0].page_content == "chunk for event streams processing"
    assert state["attempts"] == 2


def test_loop_cap_stops_after_two_attempts():
    nodes, state = run([NO, NO, "Nggak ada yang nyebutin Kafka."])
    assert nodes.count("retrieve") == 2
    assert nodes[-1] == "answer"
    assert state["attempts"] == 2
    assert state["grade"].enough is False


def test_real_retriever_feeds_the_graph():
    nodes, state = run([YES, "ok"], retrieve=retrieval.retrieve)
    assert nodes == ["retrieve", "grade", "answer"]
    assert state["docs"] == []
