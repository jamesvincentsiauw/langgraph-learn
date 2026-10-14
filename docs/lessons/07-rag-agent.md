# Lesson 7: agentic RAG over the job descriptions

File: `lessons/07_rag_agent.py`. Retriever: `agentlab/retrieval.py`. Tests:
`tests/test_lesson_07.py`.

## What this teaches

- Plain RAG is retrieve then answer. Agentic RAG adds a judgment step: after retrieving, the
  model grades whether the chunks can answer the question and, if not, proposes a better
  query. The graph loops back to `retrieve` with that query.
- The loop is bounded by code, not by the model. `attempts` is counted in `retrieve` and the
  `route` edge sends the graph to `answer` once `MAX_ATTEMPTS` is reached, whatever the grade
  said.
- The state is explicit (`question`, `query`, `docs`, `grade`, `answer`, `attempts`), so every
  step's input and output is visible in `stream_mode="updates"`.
- The answer prompt requires citing company and title and saying when the chunks do not
  contain what was asked. The sample question asks for Kafka, which no opening mentions, and
  the run shows the model saying so.
- Retrieval is BM25 over 500-character chunks of the sample descriptions. No embedding API,
  so the retriever runs offline and in tests, and the swap to a vector store is one function.

## How it is built

- `agentlab/retrieval.py`: `chunk_text` packs whole words into chunks of at most
  `CHUNK_CHARS`; `build_documents` turns each opening into `Document`s with `key`, `company`,
  `title` and `chunk` in the metadata; `BM25Retriever.retrieve(query, k)` scores with
  `rank_bm25.BM25Okapi` and drops zero-score chunks, so an unmatched query returns an empty
  list instead of random text.
- `retrieve(query, k)` at module level is the swap point. A vector store version is
  `return store.similarity_search(query, k=k)` with the same `Document` return type, and the
  graph does not change. `build(model, retrieve=...)` takes the function as a keyword argument
  so tests pass a fake and you can pass a vector store.
- `Grade` is the structured output of the grader: `enough: bool` and `better_query: str`.
  `rewrite` is plain code that copies `better_query` into `query`; there is no second model
  call for the rewrite because the grader already produced it.
- `route(state)` returns `"answer"` when `grade.enough` or `attempts >= MAX_ATTEMPTS`,
  otherwise `"rewrite"`. The conditional edge lists both targets so `--graph` draws them.
- BM25 was chosen because the corpus is 28 short descriptions and the questions are keyword
  shaped (a technology name, a city). Keyword matching answers "which opening says Kafka"
  exactly, and it needs no key, so the tests exercise the real retriever.

## What to notice when you run it

- Attempt 1 uses the raw Indonesian question as the query. Only `event` matches anything, so
  the grader says not enough and proposes an English keyword query.
- Attempt 2 with the rewritten query finds a similar set. The grader again says not enough
  (there is no Kafka chunk to find), and `route` goes to `answer` anyway because
  `attempts` hit 2. No third retrieval happens.
- The answer states that no chunk mentions Kafka, then lists the openings that mention event
  processing or event streams with company and title, and flags the one chunk that mentions
  neither. That is the grounding rule in `ANSWER_SYSTEM` doing its job on a question the
  corpus cannot fully answer.
- Try a question the corpus can answer ("Lowongan mana yang nyebutin Google Cloud?") and
  watch whether the grade comes back `enough=True` on the first attempt, skipping `rewrite`.

## Sample run

```
$ uv run lessons/07_rag_agent.py
[retrieve] attempt 1 'Lowongan mana yang nyebutin Kafka atau event-driven?' -> Overture Rede,
           Bright Vision Technologies, Airship, Marigold
[grade] enough=False better_query='Kafka event-driven architecture job posting'
[rewrite] query -> 'Kafka event-driven architecture job posting'
[retrieve] attempt 2 'Kafka event-driven architecture job posting' -> Overture Rede,
           Bright Vision Technologies, General Dynamics Information Technology, Airship
[grade] enough=False better_query='Kafka event-driven architecture backend engineer job'
[answer]
Dari chunk yang ada, **tidak ada lowongan yang nyebut Kafka** secara eksplisit.

Untuk **event-driven**, belum ada yang nyebut langsung, tapi ada yang paling mirip:

- **Bright Vision Technologies: Backend Software Engineer** menyebut *"durable event
  processing"* dan *"asynchronous job execution"*.
- **Airship: Backend Java Engineer** menyebut layanan yang *"process continuous event
  streams"*.
- **Overture Rede: C# Developer with Strong OOPS Concepts** menyebut *Event Hubs* (layanan
  streaming event di Azure). Ini bukan Kafka, dan chunk-nya tidak pakai istilah event-driven.

Lowongan **General Dynamics Information Technology: AWS Technical Lead/Sr. Java/Angular
Developer** tidak menyebut keduanya di chunk ini.
```

Long lines are wrapped here; the real output prints each on one line.

## Exercises

1. Keep the chunks from both attempts instead of replacing them: give `docs` a reducer that
   dedupes on `(key, chunk)` and check the answer can cite chunks from attempt 1.
2. Swap the retriever: build an `InMemoryVectorStore` over `build_documents(load_sample())`
   with an embedding model and pass `retrieve=lambda q, k: store.similarity_search(q, k=k)`
   to `build`. The tests for the graph should still pass with the fake retriever.
3. Raise `MAX_ATTEMPTS` to 3 and ask the Kafka question again. Does the third query find
   anything new, or does it cost a model call for nothing?
4. Add a `sources` field to the answer with structured output (`list[str]` of keys) and check
   each key appears in `docs`, like the grounding check in lesson 3.
