# Lesson 9: researcher, writer, reviewer

## What this teaches

The capstone is not a chat agent. It is a fixed pipeline with explicit state: `opening`,
`profile`, `research_notes`, `draft`, `review`, `revisions`. Each role is a node that reads
some fields and writes others, and the only decision in the graph is the edge after the
reviewer: approved, or two revisions spent, ends the run; anything else goes back to the
writer with the issues attached.

Ideas from earlier lessons come together:

- Typed state (lesson 6) instead of a message list: a node cannot accidentally depend on
  something another node said in prose.
- Structured output (lesson 3) for the verdict: `approved` and `issues` are fields, so the
  routing function reads a bool, not a sentence.
- A loop with a cap (lesson 7): `revisions >= MAX_REVISIONS` ends the run even when the
  reviewer never approves.
- A mechanical check in front of the model (`agentlab/checks.py`): the rules that can be a
  regex are a regex, and their findings are merged into the model's verdict.

## How it is built

`build()` in `lessons/09_application_crew.py` takes one model; the reviewer uses
`model.with_structured_output(Review)` on the same instance, so a test scripts all three
roles in call order with one `ScriptedChatModel`.

`researcher` gets the opening (title, company, locations, salary, description) and the
profile text, and returns bullet notes under `Meets`, `Gaps` and `Ask`. It also resets
`revisions` to 0. `writer` gets the notes and the profile; when a draft already exists it
appends the previous draft and the review's issues to its prompt and bumps `revisions`.
`reviewer` first runs `cover_note_issues` (em dash regex, a buzzword list, the word limit),
then asks the model, then returns `Review(approved=verdict.approved and not mechanical,
issues=mechanical + verdict.issues)`. The model can add findings; it cannot remove the
mechanical ones. `route` reads `review.approved` and `revisions`.

The opening comes from `agentlab.data.get_opening` and the profile from
`agentlab.profile.load_profile` (`PROFILE_PATH` in `.env`, falling back to
`data/profile.example.md`). Both are loaded in `__main__` and passed in as the initial
state, so the graph itself touches no files and a test can pass a two-line profile.

## What to notice when you run it

The researcher's gap list is long because the opening's description is long; most of it is
stacks the profile simply does not mention. The writer is told to pick one real gap and did
(AWS versus GCP). The reviewer rejected the first draft for a single line, "I work
remotely", which the profile does not say: it says the candidate is looking for remote roles.
That is exactly the kind of claim the reviewer exists for, and the writer's second draft
drops it. The mechanical check found nothing in either draft; the test
`test_mechanical_check_overrides_a_model_approval` shows what happens when it does.

The default opening (first sample row with a salary) is a manufacturing engineer role, which
makes the researcher's gap list almost the whole description. Pass a key, as below, for a
run worth reading.

## Sample run

```
$ uv run lessons/09_application_crew.py himalayas:backend-engineer-golang-python
Opening: Backend Engineer (Golang - Python) at Startup Talents

[researcher]
**Meets:**
- Writes production Go and Python, which matches the core language requirements.
- Owns services end to end (design, tests, deploy, on-call), which fits the design and build responsibilities.
- Runs Postgres in production, which matches one of the listed databases.
**Gaps:**
- The profile does not state 5+ years of distributed backend experience.
- No mention of Docker or Kubernetes.
- No AWS experience; the profile runs on GCP.
- Location conflict: the profile seeks Singapore, Japan, NZ, or fully remote roles, while the opening lists the United States and asks for someone based in Europe or the EU timezone.
  (seven more gaps trimmed)
**Ask:**
- The listing shows United States as the location but says the role can be based in Europe or the EU timezone. Is the role open to candidates elsewhere?
- Does the team run on AWS, and would GCP experience transfer?
  (six more questions trimmed)

[writer] revision 0
  (first draft trimmed; it contained "I work remotely and am looking for fully remote roles")

[reviewer] {
 "approved": false,
 "issues": [
  "The claim 'I work remotely' is not supported by the profile, which only says the candidate is looking for fully remote roles."
 ]
}

[writer] revision 1
I am applying for the Backend Engineer (Golang - Python) role at Startup Talents.

I write production Go and Python and build services around Kafka and event-driven pipelines. I run them on GCP with Postgres, and I own each service end to end: design, tests, deploy, and on-call. That matches the core language and database requirements and the build responsibilities in the listing. I am looking for fully remote roles, so the 100% remote setup suits my search.

The gap I should be direct about: I have no AWS experience. I run on GCP, so if your team is on AWS, I would need to ramp up there.

The listing shows a United States location but asks for someone in Europe or the EU timezone. Is the role open to candidates outside Europe?

[reviewer] {
 "approved": true,
 "issues": []
}
```

## Exercises

1. Add the reviewer's rejected drafts to the state (a list with an `operator.add` reducer)
   and print them at the end. How often does the writer reintroduce an issue it already
   fixed?
2. Give the reviewer its own, cheaper or stricter model via a `reviewer_model` keyword on
   `build()`, following the pattern in lesson 8. Script it separately in a test.
3. Extend `cover_note_issues` with a check that the draft names the company from the
   opening. Which other reviewer rules can become mechanical, and which cannot?
4. Replace the writer's "fix these issues" prompt with a tool call: give the writer a
   `get_profile_fact(topic)` tool and see whether a draft with a rejected claim gets fixed by
   checking the profile instead of by deleting the sentence.
