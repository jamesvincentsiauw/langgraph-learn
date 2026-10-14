# langgraph-learn

Learning LangChain 1.x and LangGraph 1.x by building agents over real data: open job
openings collected by [jobscout](https://github.com/jamesvincentsiauw) (28 of them ship in
`data/sample_openings.json`), plus live SGD/JPY/NZD/USD to IDR rates from
[Frankfurter](https://frankfurter.dev).

Each lesson is one runnable file. They go from a hand-written agent loop to multi-agent
systems, evals and a served endpoint. The same tools and data are reused throughout, so the
only thing that changes between lessons is the orchestration idea being taught.

## Setup

```sh
uv sync
cp .env.example .env   # fill in ANTHROPIC_API_KEY or OPENROUTER_API_KEY
uv run lessons/01_hand_built_agent.py "Ada lowongan golang remote?"
uv run lessons/01_hand_built_agent.py --graph   # Mermaid, paste into mermaid.live
```

`MODEL` overrides the model id (default `claude-haiku-5-5`). Tests and lint run without a key:

```sh
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

## Lessons

| # | File | What it teaches |
| --- | --- | --- |
| 1 | `01_hand_built_agent.py` | The agent loop written by hand: `StateGraph`, a `messages` state with the `add_messages` reducer, a model node, a tools node, and a conditional edge that ends the run when the model stops calling tools |
| 2 | `02_create_agent.py` | The same agent from `create_agent` in one call. Compare `--graph` output: same nodes, same edges |
| 3 | `03_structured_output.py` | Typed results with Pydantic: `with_structured_output` on a model and `response_format` on an agent, then checking the result against what the tools returned |
| 4 | `04_memory.py` | Checkpointers and `thread_id`: `InMemorySaver`, then `SqliteSaver`, so "yang nomor 2 apply-nya di mana?" works across runs |
| 5 | `05_human_in_the_loop.py` | `interrupt()` before a tool that writes (the shortlist), resuming with `Command(resume=...)`, and the same thing via `HumanInTheLoopMiddleware` |
| 6 | `06_pipeline_graph.py` | A non-chat graph: fixed nodes that extract, filter and score openings, `Send` to fan out scoring in parallel, a reducer to collect results, an LLM only where judgment is needed |
| 7 | `07_rag_agent.py` | Agentic RAG over the job descriptions: retrieve (BM25), grade relevance, rewrite the query, answer, with a loop cap |
| 8 | `08_supervisor.py` | Multi-agent with a supervisor: specialist agents as subgraphs, `Command` handoffs, and why the supervisor should not see every tool |
| 9 | `09_application_crew.py` | Capstone pipeline: researcher, writer and reviewer agents drafting an application for one opening, with a revise loop and a max revisions guard |
| 10 | `10_middleware_and_streaming.py` | `create_agent` middleware (tool call limits, summarization, a custom hook), stream modes, `astream_events`, and token usage tracking |
| 11 | `11_evals.py` | A small eval harness: a dataset of questions, mechanical checks, an LLM judge with structured output, and a results table |
| 12 | `12_serve.py` | The agent behind a FastAPI endpoint with SSE streaming and SQLite thread persistence |

Lesson notes with what to look for in the output and exercises are in `docs/lessons/`.
The shape every lesson follows is in `docs/conventions.md`.

## Layout

```
agentlab/     shared code: model factory, tools, data access, printing, test fakes
lessons/      one file per lesson
docs/         lesson notes and conventions
data/         sample openings (real, exported from jobscout) and the shortlist file
tests/        one test file per lesson, no API key needed
```
