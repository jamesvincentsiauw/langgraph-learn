# Lesson 11: a small eval harness

## What this teaches

An eval is a dataset plus checks you can rerun after every prompt or model change. The
dataset here (`data/evals.json`) is eight questions in casual Indonesian. Each case lists
mechanical checks that hold for the sample data: the right tool was called with the right
arguments (`called_tool`), a company the tool returned is named in the answer
(`mentions_returned_company`), no salary figure appears when the returned openings have
none (`no_salary_unless_returned`), and so on. One LLM judge then scores what a regex
cannot: is the answer in casual Indonesian, and does every salary it states come from the
tool outputs. The run prints a table and exits 1 when any mechanical check fails.

## How it is built

`agentlab/evals.py` holds the checks. Each is a function over the final message list that
returns a `CheckResult(name, ok, detail)`; `CHECKS` maps the names used in the JSON to the
functions and `run_checks` applies a list of specs. The checks read the transcript, not the
prose: `returned_openings` parses the `search_openings` `ToolMessage` back to rows, which is
how `mentions_returned_company` knows which companies count and how
`no_salary_unless_returned` knows whether a salary was available at all.

`lessons/11_evals.py` has the judge: `Verdict(score, reason)` as a Pydantic model,
`judge(messages, judge_model)` builds a prompt from the tool outputs and the final answer
and calls `judge_model.with_structured_output(Verdict)`. Search results go in slimmed to
company, title and salary, so the judge sees every row instead of a truncated JSON dump.
`run_case` times the agent and runs both layers; `print_report` prints the table and the
summary line; `exit_code` looks only at the mechanical checks.

Mechanical checks come first for three reasons. They are deterministic, so a failure is a
real regression and not judge noise. They are free, so they run on every change. And they
catch the failures a judge is bad at: whether `market="SG"` was actually passed is a fact in
the transcript, not an opinion. The judge is for the fuzzy part. It can be trusted to spot
register (formal vs casual Indonesian) and to compare figures in the answer against the
tool outputs it was shown. It cannot be trusted with arithmetic (it will not recompute an
IDR conversion), with anything it was not shown (so the prompt must include the full tool
outputs), or with consistency across runs (the same answer can get a 4 one day and a 5 the
next). That is why the judge score is reported next to the checks and never decides the
exit code.

## What to notice when you run it

The sample below is a clean run: 8/8 cases passed, mean judge 4.8. In an earlier run the
judge gave `sg_salary_to_idr` a 3 with the reason that two openings "cannot be verified
because the tool output is truncated". The prompt was cutting tool outputs at 1500
characters, so the judge was right and the harness was wrong. Slimming the rows fixed it.
Read the judge's reason before trusting its score; it is often the more useful of the two.

Case names tell you what each one guards. `sg_no_match` asks for a role that does not exist
in the data, so the only way to fail `no_salary_unless_returned` is to invent one.
`plain_conversion` checks that a pure arithmetic question does not trigger a search.

`tests/test_lesson_11.py` runs every check on hand-built message lists (including a
made-up NZD salary that must fail), then runs `run_all` with a scripted agent model and a
scripted judge and asserts the aggregated table and the exit code.

## Sample run

```
$ uv run lessons/11_evals.py
case                   checks  judge  seconds
sg_engineer            3/3     5/5      7.2
sg_salary_to_idr       4/4     5/5     11.3
jp_robotics_no_salary  4/4     5/5     13.6
nz_mobile              4/4     4/5      7.4
golang_remote          3/3     5/5     13.5
plain_conversion       4/4     5/5      7.0
sg_no_match            3/3     5/5      8.9
sg_ios_salary          4/4     4/5     10.7

8/8 cases passed every check, mean judge 4.8/5, 80s
```

A failed check prints under its row, for example
`FAIL called_tool: search_openings wanted {'market': 'SG'}, saw [{'keyword': 'engineer'}]`,
and a judge score under 4 prints its reason.

## Exercises

1. Add a case that asks for a salary in JPY. Which check would catch the model converting
   with a made-up rate instead of calling `convert_to_idr`?
2. Write a `tool_calls_at_most(messages, n)` check, register it, and put it on
   `plain_conversion`. Then make the system prompt vaguer and watch which cases break.
3. Run the harness three times and compare judge scores per case. How much do they move
   when the mechanical results do not?
4. Swap `make_model()` for a smaller or cheaper model in `build()` only (keep the judge) and
   read the table. Which failures are mechanical and which only the judge noticed?
