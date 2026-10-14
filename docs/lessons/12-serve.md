# Lesson 12: the agent behind FastAPI with SSE

## What this teaches

Serving the agent is mostly a question of streaming. A compiled graph with a checkpointer
already gives you threads (`thread_id`), history (`get_state`) and a stream of events; the
endpoint turns that stream into server-sent events so a client can show tool calls and text
as they happen instead of waiting for the final answer. Two routes:

- `POST /threads/{thread_id}/messages` with `{"text": ...}` streams `tool` events (a call
  with its args, then its result), `token` events for every text chunk, and one `done` event
  with the final text.
- `GET /threads/{thread_id}` returns the message history from the `SqliteSaver` checkpoint.

## How it is built

`build()` is the lesson-4 agent: `create_agent` plus `open_sqlite_saver()` from
`agentlab.checkpoints`, which opens `data/threads.sqlite` with `check_same_thread=False`
because the request runs on a worker thread.

`events()` is the whole streaming story. It calls `agent.stream(..., stream_mode=["updates",
"messages"])`, which yields `(mode, payload)` tuples from one run. The `messages` mode gives
`AIMessageChunk`s, each becomes a `token` event. The `updates` mode gives whole messages per
node: an `AIMessage` with `tool_calls` becomes one `tool` event per call, a `ToolMessage`
becomes a `tool` event with the result (cut to 200 characters), and the final `AIMessage`
is kept for the `done` event. Event data is JSON so whitespace in tokens survives the SSE
framing.

`build_app(model=None, db_path=THREADS_PATH)` builds the agent and the FastAPI app around
it. `EventSourceResponse` from sse-starlette takes the sync generator and iterates it on a
worker thread. Tests call `build_app` with a scripted model and a temporary sqlite path and
drive it with `fastapi.testclient.TestClient`.

## What to notice when you run it

The second POST on the same thread does not repeat the search: the model has the thread
history from the checkpoint, so "Yang pertama apply-nya di mana?" goes straight to
`get_opening_details` for the first result. The GET afterwards shows the full sequence
(human, ai with tool_calls, tool, ai, human, ai with tool_calls, tool, ai).

Token events arrive in chunks of several words, not single tokens, because the provider
batches them; with a scripted model in the tests they are single words. The `done` event
repeats the full text so a client that only wants the answer can ignore the tokens.

This run used the local jobscout database (five golang openings instead of the two in the
sample): as `docs/conventions.md` says, the tools prefer the jobscout DB when it exists.

`tests/test_lesson_12.py` parses the SSE lines back into `(event, data)` pairs and asserts
the order (tool, tool, tokens, done), that the second post sees the thread history (via
`ScriptedChatModel.seen`), that GET returns the checkpointed messages, and that a new app on
the same sqlite file still sees the thread.

## Sample run

```
$ uv run lessons/12_serve.py &
threads stored in /Users/.../langgraph-learn/data/threads.sqlite
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)

$ curl -s -N -X POST localhost:8000/threads/demo/messages \
    -H 'content-type: application/json' -d '{"text": "Ada lowongan golang remote? Singkat aja."}'
event: tool
data: {"name": "search_openings", "args": {"keyword": "golang", "market": "REMOTE"}}

event: tool
data: {"name": "search_openings", "result": "[{\"key\": \"himalayas:backend-engineer-golang-python\", \"company\": \"Startup Talents\", ...

event: token
data: {"text": "Ada 5 lowongan golang remote:\n\n1"}

event: token
data: {"text": ". **Backend Engineer (Golang - Python)**, Startup Talents. Remote, US. Gaji USD 80.000 ...
...
event: done
data: {"text": "Ada 5 lowongan golang remote:\n\n1. **Backend Engineer (Golang - Python)**, Startup Talents. ...

$ curl -s -N -X POST localhost:8000/threads/demo/messages \
    -H 'content-type: application/json' -d '{"text": "Yang pertama apply-nya di mana?"}'
event: tool
data: {"name": "get_opening_details", "args": {"key": "himalayas:backend-engineer-golang-python"}}

event: tool
data: {"name": "get_opening_details", "result": "{\"key\": \"himalayas:backend-engineer-golang-python\", \"company\": \"Startup Talents\", ...

event: token
data: {"text": "Apply-nya l"}

event: token
data: {"text": "ewat Himalayas: https://himalayas.app/companies/startup-talents/jobs/backend-engineer-golang-python\n\nCatatan: ...

event: done
data: {"text": "Apply-nya lewat Himalayas: https://himalayas.app/companies/startup-talents/jobs/backend-engineer-golang-python ...

$ curl -s localhost:8000/threads/demo
{"thread_id":"demo","messages":[{"type":"human","text":"Ada lowongan golang remote? Singkat aja."},{"type":"ai","text":"","tool_calls":[{"name":"search_openings","args":{"keyword":"golang","market":"REMOTE"}}]},{"type":"tool","text":"[{\"key\": ...","name":"search_openings"}, ...
```

## Exercises

1. Add `DELETE /threads/{thread_id}` using `agent.checkpointer.delete_thread(thread_id)` and
   a test that GET returns an empty list afterwards.
2. Emit an `error` event instead of a 500 when a tool raises (make `convert_to_idr` fail by
   passing a fake currency). Where in `events()` does the exception surface?
3. Replace the sync generator with an `async def` using `agent.astream` and the lesson-10
   `ToolTimer`. What has to change for the middleware to work under `astream`?
4. Write a tiny client with `httpx` that prints tokens as they arrive and compare the time
   to first token with the time to `done`.
