"""Keyword retrieval over the sample openings, chunked, for the RAG lessons.

BM25 needs no embedding API, so retrieval works offline and in tests. Swapping in a vector
store means replacing `retrieve`; the graph only ever sees that function.
"""

import re
from functools import lru_cache

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from agentlab import data

CHUNK_CHARS = 500
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Pack whole words into chunks of at most `size` characters, so no word is cut."""
    chunks: list[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > size:
            chunks.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        chunks.append(current)
    return chunks


def build_documents(rows: list[dict]) -> list[Document]:
    docs = []
    for row in rows:
        for i, piece in enumerate(chunk_text(row["description"])):
            meta = {"key": row["key"], "company": row["company"], "title": row["title"], "chunk": i}
            docs.append(Document(page_content=piece, metadata=meta))
    return docs


class BM25Retriever:
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.index = BM25Okapi([tokenize(d.page_content) for d in docs])

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        scores = self.index.get_scores(tokenize(query))
        order = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        # A zero score means no query term appears in the chunk; handing it to the grader
        # would only make it look like the corpus has something it does not.
        return [self.docs[i] for i in order[:k] if scores[i] > 0]


@lru_cache(maxsize=1)
def sample_retriever() -> BM25Retriever:
    return BM25Retriever(build_documents(data.load_sample()))


def retrieve(query: str, k: int = 4) -> list[Document]:
    """The swap point: with a vector store this body becomes `store.similarity_search(query, k)`."""
    return sample_retriever().retrieve(query, k)
