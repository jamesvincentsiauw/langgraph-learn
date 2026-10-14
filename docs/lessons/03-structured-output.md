# Lesson 3: typed results and a check against the tool output

## What this teaches

Prose is hard to test. A Pydantic object is not. This lesson gets a typed result out of the
model in two ways and then checks it mechanically:

- `model.with_structured_output(Schema)` turns one model call into a parser: send text in,
  get a `Schema` instance back. Good for extraction steps with no tool loop.
- `create_agent(..., response_format=Schema)` keeps the whole agent loop and adds a last
  step: the final answer is parsed into `Schema` and stored in
  `result["structured_response"]` next to `result["messages"]`.
- The system prompt says "only state salaries the tools returned". With a typed answer that
  sentence becomes code: every `key` must appear in a `ToolMessage`, and every salary must
  equal the one the tool returned for that key.

## How it is built

`Opening`, `Openings` and `Answer` in `lessons/03_structured_output.py` are plain Pydantic
models. Salary fields default to `None`, and the field description tells the model to leave
them null when the opening states no salary, which is what makes the check meaningful.

`extract()` is way (a): one `with_structured_output` call over the raw `search_openings`
string. `build()` is way (b): lesson 2's `create_agent` plus `response_format=Answer`.

How `response_format` works depends on the model. If LangChain knows the model supports
provider-native structured output, it asks for JSON directly (`ProviderStrategy`). Otherwise
it registers the schema as a tool named after the class (`ToolStrategy`): the model "calls"
`Answer(...)`, LangChain parses the arguments and adds a `ToolMessage` saying
`Returning structured response: ...`. The sample run below took the tool route because the
OpenRouter model id has no profile; the test scripts that same tool call by name.

`agentlab/grounding.py` holds the check. `facts_from_messages()` rebuilds a `{key: row}` map
from every `ToolMessage` whose content parses as JSON, and `check_grounding()` returns a list
of problems (empty means grounded). It takes plain dicts so lesson 11's evals can reuse it.

## What to notice when you run it

Way (a) returns five openings because the raw tool output had five; the model did not get to
pick. Way (b) returns three because the question said "sebutin 3 aja": the agent decides what
goes into the typed answer, the schema only fixes its shape. The `summary` field is where the
casual Indonesian ends up; the openings list stays machine-readable.

The check passes here. Make it fail on purpose: ask for salaries in IDR and the model may put
the converted amount into `salary_min`, which the tool output never contained.

## Sample run

```
$ uv run lessons/03_structured_output.py
== (a) with_structured_output over one search_openings call
  ashby:openai/b71906c2-51a9-4ad7-a744-2390ccdc1ad8: 171000.0-345000.0 SGD
  ashby:openai/a547274b-246c-41a0-8590-a874fcc6f28b: 171000.0-345000.0 SGD
  ashby:sierra/b37a6f9e-53c7-4c47-beb5-1ab4451f3001: 295000.0-495000.0 SGD
  ashby:sierra/79953d72-60d4-43e0-8c8c-6eccda422dce: 200000.0-240000.0 SGD
  ashby:sierra/ea3d33e7-d5cb-480a-b542-535d70fe2736: 200000.0-340000.0 SGD
== (b) create_agent with response_format=Answer
  calls search_openings({'keyword': 'engineer', 'market': 'SG', 'salary_only': True})
  search_openings -> [{"key": "ashby:openai/b71906c2-51a9-4ad7-a744-2390ccdc1ad8", ...
  calls Answer({'openings': [{'key': 'ashby:openai/b71906c2-...', 'company': 'OpenAI', ...
  Answer -> Returning structured response: openings=[Opening(key='ashby:openai/b71906c2-...
  typed: Answer(openings=[Opening(key='ashby:openai/b71906c2-51a9-4ad7-a744-2390ccdc1ad8',
    company='OpenAI', title='Quality Engineer, Rack Infrastructure & Site Operations',
    salary_min=171000.0, salary_max=345000.0, currency='SGD'), Opening(key='ashby:openai/
    a547274b-...', company='OpenAI', title='PCBA Manufacturing Engineer', salary_min=171000.0,
    salary_max=345000.0, currency='SGD'), Opening(key='ashby:sierra/b37a6f9e-...',
    company='Sierra', title='Software Engineer, Agent (Cantonese Speaking)',
    salary_min=295000.0, salary_max=495000.0, currency='SGD')],
    summary='Ini 3 lowongan engineer di Singapore yang nyantumin gaji: ... belum aku
    konversi ke IDR.')
== (c) mechanical check against the tool output
  ok, every key and salary came from a tool
```

## Exercises

1. Add `period: str | None` to `Opening` and extend `check_grounding` so a yearly salary
   reported as monthly is flagged.
2. Pass `ToolStrategy(Answer)` and `ProviderStrategy(Answer)` from
   `langchain.agents.structured_output` explicitly instead of the bare class. Compare the
   message lists; which one has the extra `ToolMessage`?
3. Make the schema a union (`Answer | NoMatch`) where `NoMatch` carries only a reason, and ask
   for "lowongan COBOL di NZ". Check which type comes back.
4. Feed `check_grounding` the messages from lesson 1's run. What breaks, and what does that say
   about relying on prose?
