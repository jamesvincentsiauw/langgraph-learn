# Lesson 5: pause before a tool that writes

## What this teaches

Reading tools are safe to run on the model's say-so. `save_to_shortlist` writes a file, so
the graph should stop, show the call, and wait for a decision. LangGraph does this with
`interrupt()` and a checkpointer:

- `interrupt(value)` inside a node suspends the run. `invoke` returns with an `__interrupt__`
  key holding the value, and the checkpoint records which node is pending.
- The run resumes with `agent.invoke(Command(resume=payload), config)` on the same thread.
  Inside the node, `interrupt()` now returns `payload` and execution continues.
- The node that interrupted is re-run from its first line on resume. Anything before the
  `interrupt()` call runs twice, which decides where the call has to go.
- A checkpointer is required: without a saved state there is nothing to resume.

`create_agent` packages the same thing as `HumanInTheLoopMiddleware`, with its own request
and decision shapes (approve, edit, reject, respond).

## How it is built

`lessons/05_human_in_the_loop.py` has both variants behind one `build(model, checkpointer,
middleware=False)`.

Hand-built (`build_hand_made`): lesson 1's graph with `save_to_shortlist` added to the tools
and the tools node changed. It first resolves every gated call with
`interrupt({"tool": name, "args": args})`, then runs the tools. Resolving first matters
because of the re-run rule: if the search ran before the interrupt, it would run again after
the resume (`test_hand_built_decides_gated_calls_before_running_any_tool` pins this). A
rejection becomes a `ToolMessage` with `status="error"` that carries the reason, so the model
can explain instead of retrying.

Middleware: `HumanInTheLoopMiddleware(interrupt_on={"save_to_shortlist": True})`. `True`
allows all four decisions; an `InterruptOnConfig` dict narrows them
(`{"allowed_decisions": ["approve", "reject"]}`) and can add a `when` predicate. The
middleware interrupts in an `after_model` node, before the tools node, so nothing runs twice.
The interrupt value is an `HITLRequest` (`action_requests` + `review_configs`) and the resume
payload is `{"decisions": [{"type": "approve"}]}` or
`{"type": "reject", "message": reason}`, one decision per gated call. These shapes come from
`langchain/agents/middleware/human_in_the_loop.py`, which is worth reading in full.

`resume_payload()` and `pending_call()` hide the shape difference so the runner loop is the
same for both: stream, then while `agent.get_state(config).interrupts` is non-empty, ask and
resume. `--auto-approve` and `--auto-reject` skip stdin.

## What to notice when you run it

In the middleware run the pause shows up as a node named
`HumanInTheLoopMiddleware.after_model`; `--graph --middleware` shows it between `model` and
`tools`. After a reject that node emits the model's message again (with the gated call
removed) plus the rejection `ToolMessage`, and the model's next answer says the save was
declined and does not retry.

In the hand-built run the approve resume continues inside `tools`, which prints the
`save_to_shortlist` result as if nothing had happened. Check `data/shortlist.json` after an
approve and after a reject.

## Sample run

```
$ uv run lessons/05_human_in_the_loop.py --auto-approve
[model] calls search_openings({'keyword': 'backend', 'market': 'SG'})
[tools] search_openings -> [{"key": "smartrecruiters:Deliveryhero/744000136767869", ...
[model] calls get_opening_details({'key': 'smartrecruiters:Deliveryhero/744000151330608'}); ...
[tools] get_opening_details -> {"key": "smartrecruiters:Deliveryhero/744000151330608", ...
[tools] get_opening_details -> {"key": "smartrecruiters:Grab/744000151028388", ...
[tools] get_opening_details -> {"key": "smartrecruiters:Grab/744000147831203", ...
[model] calls save_to_shortlist({'key': 'smartrecruiters:Grab/744000151028388', 'note': ...})

[paused] save_to_shortlist({'key': 'smartrecruiters:Grab/744000151028388', 'note': 'Senior
Software Engineer, Backend di Grab (FundsCore Recon, Singapore onsite). Minta 4+ tahun ...'})
[resume] approve

[tools] save_to_shortlist -> Saved smartrecruiters:Grab/744000151028388 to the shortlist (1 entries).
[model] answer:
Dari 5 lowongan backend di SG, gua pilih **Senior Software Engineer, Backend di Grab** ...
Udah gua simpan ke shortlist.

$ uv run lessons/05_human_in_the_loop.py --middleware --auto-reject
...
[model] calls save_to_shortlist({'key': 'smartrecruiters:Grab/744000151028388', 'note': ...})

[paused] save_to_shortlist({'key': 'smartrecruiters:Grab/744000151028388', ...})
[resume] reject: sample run, not a real pick

[HumanInTheLoopMiddleware.after_model] calls save_to_shortlist({...})
[HumanInTheLoopMiddleware.after_model] save_to_shortlist -> User rejected the tool call for
`save_to_shortlist` with reason: sample run, not a real pick
[model] answer:
Gua udah nyari dan baca 3 lowongan backend di SG yang paling relevan, tapi shortlist belum
gua simpan. Penyimpanannya ditolak dengan alasan "sample run, not a real pick", jadi gua
nggak coba ulang. ...
```

## Exercises

1. Add an "edit" path to the hand-built variant: the resume payload carries new `args` and the
   tool runs with those. Then do the same with the middleware's `edit` decision and compare
   what the model is told afterwards (`edit_notice`).
2. Use `InterruptOnConfig` with a `when` predicate so saving is auto-approved when the note
   mentions a salary, and gated otherwise.
3. Gate `convert_to_idr` as well and ask a question that triggers both tools in one turn. How
   many interrupts do you get, and how many decisions does the resume need?
4. Run with `--sqlite`-style persistence (reuse `agentlab.checkpoints.open_sqlite_saver`), stop
   the process at the pause, and resume from a second process with only the thread id.
