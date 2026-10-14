# Lesson 8: a supervisor and two specialists

## What this teaches

One agent with every tool works until the tool list and the prompt grow past what one
model call can follow. The usual split is a supervisor that decides who works next and
specialists that each own a small set of tools. Here `job_scout` has `search_openings` and
`get_opening_details`, `salary_analyst` has `convert_to_idr` and `monthly_salary`, and the
supervisor has neither: it only has two handoff tools, `transfer_to_job_scout` and
`transfer_to_salary_analyst`.

Three LangGraph ideas appear here:

- `Command(goto=..., update=...)` returned from a node both writes state and picks the next
  node, so the supervisor needs no conditional edges. The `Command[Literal[...]]` return
  annotation is what tells `draw_mermaid` which edges exist.
- A compiled graph (here each `create_agent` result) can be invoked inside a node of another
  graph. With `stream(subgraphs=True)` its inner steps show up with a namespace.
- Several `goto` targets in one `Command` run as parallel nodes in the same step.

## How it is built

`build()` in `lessons/08_supervisor.py` takes three optional models (supervisor and one per
specialist) so a test can script each role on its own. `build_job_scout` and
`build_salary_analyst` are plain `create_agent` calls with a short system prompt and a
two-tool list; they are importable and testable without the supervisor.

The supervisor node binds the model to the two handoff tools. A reply with no tool calls is
the final answer and routes to `END`. A reply with handoff calls routes to the named
specialists; the tool call stays open in the history. The specialist node finds its own call
in the last message, runs its agent on the history up to that call plus a `Task from
supervisor:` line, and hands back a `ToolMessage` whose content is the agent's final text,
with `Command(goto="supervisor")`.

Only that final text crosses back. The searches and conversions the specialist ran stay in
its subgraph. The tradeoff: the supervisor never sees raw tool output, so it has to trust the
specialist's summary, and a specialist that drops a fact (the opening key, the rate date)
loses it for everyone. The gain is that the shared history stays one tool-call pair per
handoff, so a run with five handoffs does not drag thirty tool messages into every supervisor
prompt. If you want the opposite, hand back `result["messages"][len(history):]` instead and
watch the prompt size grow.

Why the supervisor does not see every tool: a model picking from two handoffs makes fewer
wrong calls than one picking from five tools and reasoning about currency and search at the
same time; a failure is attributable (the analyst converted the wrong number, the scout
picked the wrong opening); and each specialist can be invoked and tested alone, which
`tests/test_lesson_08.py` does.

## What to notice when you run it

The first live run of this lesson had `job_scout` converting USD to IDR with a made-up rate
because the user asked for rupiah and the scout had no conversion tool. The analyst then
"corrected its earlier answer", because with a shared history every specialist reads the
other's reply as its own. Two prompt lines fixed it (`Do not convert currencies or guess
exchange rates; salary_analyst does that` and `without commenting on earlier messages`), and
the point stands: a shared history is convenient, but each specialist needs to be told what
is not its job.

Specialists answered in Indonesian although their prompts ask for English: the shared
history outweighs a one-line instruction. The supervisor's final answer only repeats numbers
that appeared in a `ToolMessage`.

## Sample run

```
$ uv run lessons/08_supervisor.py "Ada lowongan golang remote? Ambil satu yang ada gajinya, rupiahin gaji per bulannya. Singkat."
[supervisor] calls transfer_to_job_scout({'task': 'Cari lowongan backend engineer Golang yang remote. Pilih satu lowongan yang mencantumkan gaji, ...'})
[job_scout/model] calls search_openings({'keyword': 'golang', 'market': 'REMOTE', 'salary_only': True})
[job_scout/tools] search_openings -> [{"key": "himalayas:backend-engineer-golang-python", "company": "Startup Talents", ...
[job_scout/model] calls get_opening_details({'key': 'himalayas:backend-engineer-golang-python'})
[job_scout/tools] get_opening_details -> {"key": "himalayas:backend-engineer-golang-python", ...
[job_scout/model] answer:
Saya tidak mengonversi ke rupiah, karena konversi mata uang ditangani oleh salary_analyst. Berikut lowongan yang dipilih:
- **Key:** himalayas:backend-engineer-golang-python
- **Perusahaan:** Startup Talents
- **Posisi:** Backend Engineer (Golang - Python)
- **Gaji:** 80.000–100.000 USD per tahun
[job_scout] transfer_to_job_scout -> Saya tidak mengonversi ke rupiah, ...
[supervisor] calls transfer_to_salary_analyst({'task': 'Konversi gaji berikut ke bulanan dan ke IDR. Gaji: 80.000–100.000 USD per tahun ...'})
[salary_analyst/model] calls monthly_salary({'amount': 80000, 'currency': 'USD', 'period': 'year'}); calls monthly_salary({'amount': 100000, 'currency': 'USD', 'period': 'year'})
[salary_analyst/tools] monthly_salary -> 80,000 USD per year = 6,667 USD per month (gross)
[salary_analyst/tools] monthly_salary -> 100,000 USD per year = 8,333 USD per month (gross)
[salary_analyst/model] calls convert_to_idr({'amount': 6667, 'currency': 'USD'}); calls convert_to_idr({'amount': 8333, 'currency': 'USD'})
[salary_analyst/tools] convert_to_idr -> 6,667 USD = Rp119,212,627 (rate 17881, 2026-10-13)
[salary_analyst/tools] convert_to_idr -> 8,333 USD = Rp149,002,373 (rate 17881, 2026-10-13)
[salary_analyst/model] answer:
**Gaji per bulan (kotor):**
- 80.000 USD/tahun = **6.667 USD/bulan** ≈ **Rp119.212.627**
- 100.000 USD/tahun = **8.333 USD/bulan** ≈ **Rp149.002.373**
Kurs: 1 USD = Rp17.881 (per 13 Oktober 2026).
[salary_analyst] transfer_to_salary_analyst -> **Gaji per bulan (kotor):** ...
[supervisor] answer:
Ketemu satu lowongan: **Backend Engineer (Golang - Python)** di Startup Talents, remote.
- **Gaji:** 80.000–100.000 USD per tahun
- **Per bulan (kotor):** sekitar 6.667–8.333 USD, atau sekitar **Rp119 juta – Rp149 juta**
- **Kurs:** 1 USD = Rp17.881 (13 Oktober 2026)
Catatan: lokasinya tercantum United States, tapi deskripsinya minta remote Eropa/zona waktu EU.
```

## Exercises

1. Hand back the specialist's full message trail instead of only its final text (see "How it
   is built"). Print `len(state["messages"])` in the supervisor node and compare the two
   versions on the same question.
2. Ask a question that needs both specialists at once ("cari lowongan golang remote dan
   rupiahin 120000 SGD per tahun"). Does the supervisor issue two handoffs in one reply? The
   graph supports it (`test_two_handoffs_in_one_reply_run_both_specialists`); the model may
   still choose to go one at a time.
3. Add a third specialist, `shortlister`, that owns `save_to_shortlist`, and put lesson 5's
   `interrupt()` in front of it. Where does the interrupt surface: in the subgraph or in the
   supervisor graph?
4. Give the supervisor `search_openings` as well as the handoffs and ask the same question.
   Count how often it searches itself instead of delegating, and whether the answers differ.
