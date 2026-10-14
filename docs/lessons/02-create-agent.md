# Lesson 2: the same agent from create_agent

## What this teaches

`create_agent` from `langchain.agents` builds lesson 1 for you: a model node, a tools node
(`ToolNode`), and the conditional edge. It also accepts a `system_prompt`, a `checkpointer`
(lesson 4), `middleware` (lesson 10) and `response_format` (lesson 3), so most of the later
lessons start from this one call instead of a hand-written graph.

The lesson is mostly a comparison. Run both files with `--graph` and diff the Mermaid: the
node names differ but the shape is the same.

## How it is built

`build()` is one line. The returned object is a compiled `StateGraph`, so `.stream()`,
`.invoke()`, `.get_graph()` and `.get_state()` all work exactly as in lesson 1.

## What to notice when you run it

Parallel tool calls run concurrently inside `ToolNode`, so two `convert_to_idr` results may
print in either order. The state after the run has the same message sequence as lesson 1:
human, ai (with tool calls), tool, ..., ai (final answer). `tests/test_lessons_01_02.py`
asserts that sequence for both lessons with a scripted model.

## Sample run

```
$ uv run lessons/02_create_agent.py --graph
graph TD;
	__start__([<p>__start__</p>]):::first
	model(model)
	tools(tools)
	__end__([<p>__end__</p>]):::last
	__start__ --> model;
	model -.-> __end__;
	model -.-> tools;
	tools -.-> model;
```

The question run matches lesson 1's sample, with `[model]` and `[tools]` as node names.

## Exercises

1. Pass `debug=True` to `create_agent` and read what it logs per step.
2. Give the agent a tool it should never need (for example `save_to_shortlist`) and ask a
   question that tempts it. Does the system prompt stop it? That gap is what lesson 5 closes.
3. Remove `system_prompt` and ask the same question. Compare language and salary handling.
