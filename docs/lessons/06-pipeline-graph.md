# Lesson 6: a pipeline graph, with the model only where judgment is needed

File: `lessons/06_pipeline_graph.py`. Tests: `tests/test_lesson_06.py`.

## What this teaches

- A graph does not need a `messages` list. The state here is a `TypedDict` with named fields
  (`profile`, `openings`, `scores`, `shortlist`) and each node reads and writes only the
  fields it owns.
- `load`, `filter` and `rank` are plain Python. They are tested with dicts, no model involved.
- `score` is the one node that calls the model, through `with_structured_output(Verdict)` so
  the result is a validated `fit` and `reason`, not prose to parse.
- The `fan_out` conditional edge returns a list of `Send("score", {...})`, one per opening.
  LangGraph runs those as parallel tasks, each with its own small input, and waits for all of
  them before moving on to `rank`.
- `scores: Annotated[list[Score], operator.add]` is a reducer. Each score run returns a
  one-item list and LangGraph appends it. Without the reducer the parallel writes would
  overwrite each other and only one score would survive.

## How it is built

- `State` holds the pipeline fields. `ScoreInput` is a second, smaller `TypedDict`: it is what
  a `Send` carries, and because `score(state: ScoreInput)` is annotated with it, LangGraph
  infers it as that node's input schema.
- `keep_wanted(openings, markets)` drops openings whose `regions` do not overlap the wanted
  markets, then dedupes by lowercased company + title. `pick_top(openings, scores, n)` sorts
  by `fit` and joins the opening fields back in. Both are module-level functions so the tests
  call them directly.
- `fan_out(state)` returns `"rank"` when nothing survived the filter, otherwise the list of
  `Send` objects. `add_conditional_edges("filter", fan_out, ["score", "rank"])` declares both
  destinations so `--graph` can draw them.
- `Verdict` is what the model is asked for (`fit` 0 to 10 with `Field(ge=0, le=10)`, and a
  one-line `reason`). `Score` extends it with `key`, which the `score` node attaches from the
  opening itself. The model never types an identifier.
- The profile text comes from `agentlab.profile.load_profile()`: `PROFILE_PATH` in `.env`, or
  `data/profile.example.md`, a labeled placeholder profile.
- Tests use `agentlab.fakes.KeyedChatModel`, which picks its reply by which opening title
  appears in the prompt. The parallel `score` runs finish in any order, so a reply script that
  depends on call order would be flaky.

## What to notice when you run it

- `[score]` lines arrive in an order that changes between runs. That is the parallel fan-out.
- The run below uses the bundled sample (`JOBSCOUT_DB` pointed at a missing file). With the
  jobscout DB present, `load` reads from it instead and you will see different openings.
- `load` found 24 openings with "engineer" in the title; `filter` kept 13 for SG and NZ. That
  cut happened in Python before any model call, which is the cheap part of the pipeline.
- The one real backend role (Delivery Hero) scored 7 and tops the shortlist. Mobile and sales
  roles scored 1 or 2 with the exclusion from the profile named in the reason, which is what a
  structured verdict buys you: you can read why.
- `--graph` shows dotted edges from `filter` to both `score` and `rank`, and a solid edge
  `score -> rank`. The fan-out itself is not visible in the drawing; it is decided at runtime.

## Sample run

```
$ JOBSCOUT_DB=/nonexistent uv run lessons/06_pipeline_graph.py engineer --markets SG,NZ
[load] 24 openings with 'engineer' in the title
[filter] 13 left for markets ['SG', 'NZ'] after dedupe
[score]  1  ashby:airwallex/0248185e-...: The opening is an iOS mobile engineer role, which
            the candidate explicitly excludes, though it is onsite in Singapore, ...
[score]  1  smartrecruiters:Wise/744000145649689: It is an Android mobile role, which the
            candidate explicitly does not want, ...
[score]  2  ashby:sierra/ea3d33e7-...: Enterprise Sales Engineer is a customer-facing sales
            role, which the candidate explicitly excludes, ...
[score]  1  ashby:halter/05147ed1-...: This is a React Native (mobile) lead role onsite in
            Auckland, and the candidate explicitly excludes mobile roles, ...
... (six more scores between 1 and 3)
[score]  7  smartrecruiters:Deliveryhero/744000136767869: Backend role in Singapore, which
            matches the target location, and it centers on owning backend systems. ...
[score]  3  ashby:halter/20940ac0-...: Location fits (Auckland, NZ) and the work is likely
            backend or platform adjacent, but it is an engineering manager role, ...
[score]  6  ashby:sierra/79953d72-...: Location matches (Singapore onsite) and the work is
            backend-leaning software for an AI platform, but it is a new-grad role ...
[rank] 1. Delivery Hero: Principal Backend Software Engineer (fit 7)
[rank] 2. Sierra: Software Engineer, Agent (New Grad 2027) (fit 6)
[rank] 3. Sierra: Software Engineer, Agent (Cantonese Speaking) (fit 4)
```

Reasons are cut with `...` here; the real output prints them in full on one line.

## Exercises

1. Replace `operator.add` with a custom reducer that keeps only the highest `fit` per key, then
   send the same opening twice from `fan_out` and check that `scores` has one entry for it.
2. Add a `salary_only` field to `State` and pass it through `load` to `query_openings`. The
   filter and rank code should not need to change.
3. Give `score` a `retry_policy` (`add_node(..., retry_policy=RetryPolicy(...))`) and make the
   fake raise once for one key to see that only that `Send` run is retried.
4. Write your own profile, point `PROFILE_PATH` at it in `.env`, and run with the jobscout DB
   present. Compare the top 3 against what you would have picked by hand.
