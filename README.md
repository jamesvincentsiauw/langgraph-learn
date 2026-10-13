# langgraph-learn

Learning LangGraph by building a small agent over real data: open job openings from the
[jobscout](../jobscout) SQLite database, plus live SGD/JPY/NZD/USD to IDR rates from
[Frankfurter](https://frankfurter.dev).

## Setup

```sh
uv sync
cp .env.example .env   # fill in ANTHROPIC_API_KEY or OPENROUTER_API_KEY
```

`common.py` uses Anthropic when `ANTHROPIC_API_KEY` is set, otherwise OpenRouter. `MODEL`
overrides the model id.

## Lessons

| File | What it shows |
| --- | --- |
| `01_graph_agent.py` | The agent loop written by hand: `StateGraph`, a `messages` state with the `add_messages` reducer, a `model` node, a `tools` node, and a conditional edge that ends the run when the model stops calling tools |
| `02_create_agent.py` | The same agent from LangChain's `create_agent` in one call. Run both with `--graph` and compare: same nodes, same edges |

```sh
uv run 01_graph_agent.py "Ada lowongan golang remote?"
uv run 01_graph_agent.py --graph   # Mermaid, paste into mermaid.live
```

Each step prints as `[node] ...`, so you can watch the model call tools, get results, and
call the model again.

## Next

3. Memory: compile with a checkpointer (`InMemorySaver`, then SQLite) and pass a `thread_id`,
   so a follow-up like "yang nomor 2 apply-nya di mana?" works.
4. Human in the loop: `interrupt()` before a tool that writes something, resume with
   `Command(resume=...)`.
5. Structured output: return a typed result (Pydantic) instead of free text, then check it
   against the tool output.
6. A non-chat graph: extract, filter, score as fixed nodes with an LLM only where it is needed.
