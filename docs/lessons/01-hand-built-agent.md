# Lesson 1: the agent loop by hand

## What this teaches

An "agent" in LangGraph is a graph with two nodes and one decision. The model node sends the
conversation to the LLM. If the reply contains tool calls, the tools node runs them and
appends the results. The conditional edge after the model node is the only logic: tool calls
present means go to tools, otherwise stop. Everything else (memory, approvals, multiple
agents) is built by adding nodes and edges to this shape.

Three LangGraph ideas appear here:

- State is a TypedDict. Nodes receive the whole state and return a partial update.
- A reducer decides how an update merges into the state. `add_messages` appends instead of
  replacing, which is what a conversation needs.
- `add_conditional_edges` takes a function that returns the name of the next node.

## How it is built

`build()` in `lessons/01_hand_built_agent.py` binds the tools to the model
(`model.bind_tools(TOOLS)`), so the LLM sees the tool schemas and can ask for calls. The
`run_tools` node looks each call up by name and wraps the output in a `ToolMessage` carrying
the `tool_call_id`, which is how the model matches results to its requests. `route` reads the
last message and returns `"tools"` or `END`.

The graph is compiled once and streamed with `stream_mode="updates"`, so each node's output
prints as it happens.

## What to notice when you run it

The model may issue several tool calls in one turn (two `convert_to_idr` calls below). The
hand-built tools node runs them in order. Lesson 2's `create_agent` does the same with a
`ToolNode` that can run them concurrently.

## Sample run

```
$ uv run lessons/01_hand_built_agent.py "Ada lowongan golang remote? sebutin 2 aja, singkat"
[model] calls search_openings({'keyword': 'golang', 'market': 'REMOTE'})
[tools] search_openings -> [{"key": "himalayas:backend-engineer-golang-python", ...
[model] calls convert_to_idr({'amount': 80000, 'currency': 'USD'}); calls convert_to_idr({'amount': 100000, 'currency': 'USD'})
[tools] convert_to_idr -> 80,000 USD = Rp1,430,480,000 (rate 17881, 2026-10-13)
[tools] convert_to_idr -> 100,000 USD = Rp1,788,100,000 (rate 17881, 2026-10-13)
[model] answer:
Ada 5 lowongan golang remote. Dua yang paling oke:

1. Backend Engineer (Golang - Python) dari Startup Talents, remote US. Gajinya USD 80.000
   sampai 100.000 per tahun, kira-kira Rp1,43 miliar sampai Rp1,79 miliar (kurs 17.881).
2. Contract Golang Software Engineer dari YLD.com, remote Europe. Gaji nggak dicantumkan.
```

## Exercises

1. Add a `max_steps` counter to the state and make `route` return `END` once it is reached.
   What happens to the conversation when the model is cut off mid tool call?
2. Make `run_tools` catch exceptions from a tool and return the error text as the
   `ToolMessage` content. Watch how the model reacts to a failed currency lookup.
3. Replace `add_messages` with a plain `list` annotation and run again. Explain the output.
