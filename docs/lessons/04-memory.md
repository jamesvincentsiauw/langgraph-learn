# Lesson 4: checkpointers and thread_id

## What this teaches

A compiled graph has no memory of its own. Each `invoke` starts from the input you give it,
runs to the end, and forgets. A checkpointer changes that: after every step LangGraph writes
the state to the checkpointer under the `thread_id` in `config["configurable"]`, and the next
`invoke` with the same `thread_id` loads that state first, so the new `HumanMessage` is
appended to the old conversation instead of starting a new one.

- `InMemorySaver` keeps threads in a dict: fine for tests and one process.
- `SqliteSaver` writes them to a file, so a thread survives restarts.
- `agent.get_state(config)` returns the latest checkpoint: `values` (the state),
  `next` (nodes still to run, empty when done), and `config` with the checkpoint id.
- `agent.get_state_history(config)` iterates every checkpoint of the thread, newest first.
  There is one per step, which is why a two-question thread has more checkpoints than
  messages.

Lessons 5 and 12 build on this: interrupts are checkpoints waiting for a resume, and the
served endpoint keys conversations by `thread_id`.

## How it is built

`build()` in `lessons/04_memory.py` is lesson 2's `create_agent` with one extra argument,
`checkpointer`. The default is `InMemorySaver()`; `--sqlite` passes
`agentlab.checkpoints.open_sqlite_saver()`, which opens `data/threads.sqlite` with
`check_same_thread=False` because LangGraph may run nodes on worker threads.

`ask()` streams a question into a thread and `describe_state()` prints what the checkpoint
holds. The runner asks `FIRST` and `FOLLOW_UP` on thread `demo-1`, then only `FOLLOW_UP` on
`demo-2`.

## What to notice when you run it

The follow-up on `demo-1` is answered without a tool call: the search result from the first
question is already in the thread, so the model reads the apply URL from it. The same
follow-up on `demo-2` has nothing to refer to and the model says so.

Counts: `demo-1` ends with 6 messages (human, ai, tool, ai, human, ai) and 8 checkpoints.
Run `--sqlite` twice and the first line changes from `0 messages` to `6 messages`: the second
process picked the thread up from the file, and the first question is then answered from
history, again without a search.

The test for this lesson checks the mechanism, not the prose: `ScriptedChatModel.seen` records
every prompt, and the third model call must contain the whole first turn.

## Sample run

```
$ uv run lessons/04_memory.py
[demo-1] 0 messages, 0 checkpoints, next=()

[demo-1] > Ada lowongan backend di SG? Sebutin 2 aja, singkat.
[model] calls search_openings({'keyword': 'backend', 'market': 'SG'})
[tools] search_openings -> [{"key": "smartrecruiters:Deliveryhero/744000136767869", ...
[model] answer:
Ada 5 lowongan backend di SG. Dua yang bisa langsung kamu cek:

1. **Senior Backend Software Engineer** di Delivery Hero (onsite, Singapura)
2. **Lead Software Engineer, Backend** di Grab (onsite, Singapura)

Dua-duanya belum mencantumkan gaji, jadi belum bisa aku konversi ke rupiah.

[demo-1] > Yang pertama apply-nya di mana?
[model] answer:
Apply-nya lewat halaman SmartRecruiters Delivery Hero:
https://jobs.smartrecruiters.com/DeliveryHero/744000151330608-senior-backend-software-engineer-
[demo-1] 6 messages, 8 checkpoints, next=()

[demo-2] > Yang pertama apply-nya di mana?
[model] answer:
Maaf, aku belum punya daftar lowongan sebelumnya di percakapan ini, jadi aku nggak tahu
mana yang kamu maksud sebagai "yang pertama". ...
[demo-2] 2 messages, 3 checkpoints, next=()

$ uv run lessons/04_memory.py --sqlite        # second run on the same file
threads stored in .../data/threads.sqlite
[demo-1] 6 messages, 8 checkpoints, next=()
...
[demo-1] 10 messages, 14 checkpoints, next=()
```

## Exercises

1. Print `agent.get_state(config).config["configurable"]["checkpoint_id"]` after each question,
   then call `agent.get_state(config_with_that_checkpoint_id)` to read an older state. Which
   messages are missing?
2. Ask ten questions on one thread with `--sqlite` and watch the prompt size grow (count the
   messages the model receives). Then add `SummarizationMiddleware` from
   `langchain.agents.middleware` and compare. Lesson 10 does this properly.
3. Invoke the same thread from two Python processes at once. What does `SqliteSaver` do, and
   where would you put a lock?
4. Replace the thread id with Vincent's email plus a date, so each day starts a fresh thread
   but the same day resumes. Where does that logic belong: the lesson, `agentlab`, or the
   caller?
