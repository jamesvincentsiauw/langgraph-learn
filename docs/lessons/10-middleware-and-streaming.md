# Lesson 10: middleware, stream modes and token usage

## What this teaches

`create_agent` accepts a `middleware` list. Each middleware hooks into the loop at a fixed
point (`before_model`, `after_model`, `wrap_tool_call`, ...) and shows up as its own node in
the graph, so you can watch it in the `updates` stream. Three are used here:

- `ToolCallLimitMiddleware(tool_name="convert_to_idr", thread_limit=3, exit_behavior="end")`
  counts calls in `after_model` and, past the limit, writes an error `ToolMessage` for the
  blocked call plus an explanatory `AIMessage`, then jumps to the end of the graph.
- `SummarizationMiddleware(model, trigger=("messages", 6), keep=("messages", 2))` runs in
  `before_model`: once the history holds six messages it asks the model for a summary,
  replaces the old messages with one `HumanMessage` holding it, and keeps the last two.
- `ToolTimer`, our own `AgentMiddleware`, wraps every tool call and logs its duration.

The second half streams the same run three ways: `stream_mode="updates"` (one dict per node),
`stream_mode="messages"` (text chunks as they arrive) and `astream_events` (every callback
event, here filtered to `on_chat_model_stream` and `on_tool_end`). A
`UsageMetadataCallbackHandler` passed in `config["callbacks"]` sums token usage per model.

## How it is built

`build(model, summary_model, timer)` assembles the middleware list. `summary_model` exists so
tests can script the summary, and `timer` so tests can read `ToolTimer.calls`.

`ToolTimer` is a class rather than the `@wrap_tool_call` decorator: the decorator on a sync
function installs only the sync hook, and `astream_events` drives the agent asynchronously,
which raises `NotImplementedError: Asynchronous implementation of awrap_tool_call is not
available`. The class implements `wrap_tool_call` and `awrap_tool_call` with one shared
`_record` helper. (`@before_model` on a sync function does run under async, but it cannot
time a tool call.)

`stream_messages` prints `AIMessageChunk.text` with `end=""` and one line per tool result.
`stream_events` reads `event["data"]["chunk"]` for model tokens and `event["data"]["output"]`
for tool results. `--mode` picks one; `print_usage` runs after all three.

The thread limit is really a run limit here: without a checkpointer every run starts from
an empty state, so `thread_tool_call_count` resets. Add the lesson-4 `SqliteSaver` and a
`thread_id` and the count carries over between runs.

## What to notice when you run it

In the sample below the model issues three `convert_to_idr` calls in one message. That
makes the history seven messages long at the next model call, so the summarization node
fires: `removes earlier messages`, then a `HumanMessage` that starts with "Here is a summary
of the conversation to date", then the preserved tool call and its three results. The final
answer is written from the summary plus those results. The summary is long because the
default `summary_prompt` asks for four sections; the exercise below shortens it.

The `[timer]` lines come from `ToolTimer`. Under `--mode events` they interleave with the
streamed tokens because the tool runs while the output is being printed.

Token usage under `--mode events` printed the model name twice
(`anthropic/claude-haiku-5.5anthropic/claude-haiku-5.5`). Every streamed chunk carries
`response_metadata["model_name"]` and chunk merging concatenates strings, so the handler
keys on the doubled name. The totals are right; the key is not. The non-streaming `updates`
mode prints it once.

`tests/test_lesson_10.py` scripts a model that calls `convert_to_idr` six times and asserts
that `ToolTimer` saw exactly three, that one error `ToolMessage` was injected, and that the
scripted final answer was never requested (the run jumped to the end). Another test asserts
that the `updates` stream names the middleware nodes.

## Sample run

```
$ uv run lessons/10_middleware_and_streaming.py "Rupiahin gaji minimum dari 3 lowongan engineer SG yang nyebutin gaji, satu per satu."
[model] calls search_openings({'keyword': 'engineer', 'market': 'SG', 'salary_only': True})
[timer] search_openings took 523 ms
[tools] search_openings -> [{"key": "ashby:openai/b71906c2-...", "company": "OpenAI", ...
[model] calls convert_to_idr({'amount': 171000, 'currency': 'SGD'}); calls convert_to_idr(...); calls convert_to_idr({'amount': 295000, 'currency': 'SGD'})
[timer] convert_to_idr took 152 ms
[tools] convert_to_idr -> 295,000 SGD = Rp4,121,445,000 (rate 13971, 2026-10-13)
[timer] convert_to_idr took 157 ms
[tools] convert_to_idr -> 171,000 SGD = Rp2,389,041,000 (rate 13971, 2026-10-13)
[timer] convert_to_idr took 185 ms
[tools] convert_to_idr -> 171,000 SGD = Rp2,389,041,000 (rate 13971, 2026-10-13)
[SummarizationMiddleware.before_model] removes earlier messages
[SummarizationMiddleware.before_model] answer:
Here is a summary of the conversation to date:

## SESSION INTENT

The user wants the minimum salary converted from SGD to Indonesian Rupiah (IDR) for 3 Singapore engineer job openings that list salaries, presented one at a time.
...
[SummarizationMiddleware.before_model] calls convert_to_idr({'amount': 171000, 'currency': 'SGD'}); ...
[SummarizationMiddleware.before_model] convert_to_idr -> 171,000 SGD = Rp2,389,041,000 (rate 13971, 2026-10-13)
[SummarizationMiddleware.before_model] convert_to_idr -> 171,000 SGD = Rp2,389,041,000 (rate 13971, 2026-10-13)
[SummarizationMiddleware.before_model] convert_to_idr -> 295,000 SGD = Rp4,121,445,000 (rate 13971, 2026-10-13)
[model] answer:
Oke, ini hasil konversi minimum gaji untuk 3 lowongan pertama dari hasil pencarian (kurs Rp13.971 per SGD, per 13 Oktober 2026):

**1. OpenAI, Quality Engineer, Rack Infrastructure & Site Operations** (remote, SG)
- Minimum: 171.000 SGD per tahun
- ≈ **Rp2.389.041.000** per tahun
...
[usage] anthropic/claude-haiku-5.5: 7236 in, 2509 out, 9745 total

$ uv run lessons/10_middleware_and_streaming.py --mode events "Gaji 200000 SGD setahun itu berapa rupiah?"
[on_tool_end] convert_to_idr -> 200,000 SGD = Rp2,794,200,000 (rate 13971, 2026-10-13)
[timer] convert_to_idr took 111 ms
Gaji 200.000 SGD setahun kira-kira **Rp2.794.200.000** (kurs 1 SGD ≈ Rp13.971, data tanggal 13 Oktober 2026).
[usage] anthropic/claude-haiku-5.5anthropic/claude-haiku-5.5: 1772 in, 266 out, 2038 total
```

## Exercises

1. Change `exit_behavior` to `"continue"` and rerun the first question. The blocked call now
   gets an error `ToolMessage` and the model answers anyway. Which behaviour do you want for
   a paid API like the rate lookup?
2. Pass a short `summary_prompt` to `SummarizationMiddleware` (three lines: goal, facts
   found, what is left) and compare the token usage line with the sample.
3. Add `ModelCallLimitMiddleware(run_limit=3)` and script a test the same way as
   `test_convert_limit_ends_the_run`. Check in the source which hook it uses.
4. In `stream_events`, also print `on_tool_start` with its input. Then filter events by
   `event["metadata"]["langgraph_node"]` to show only what the `tools` node emits.
