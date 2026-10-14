# Lesson conventions

Every lesson is one file in `lessons/` and follows the same shape, so the pieces compose and
the tests can run without an API key.

- `build(model: BaseChatModel | None = None)` returns the compiled graph. The default model
  comes from `agentlab.config.make_model()`. Tests pass `agentlab.fakes.ScriptedChatModel`.
- `if __name__ == "__main__":` runs the lesson from the command line. `--graph` prints the
  graph as Mermaid (paste into mermaid.live).
- Tools, data access and printing live in `agentlab/`, not in the lesson file.
- The module docstring says what the lesson teaches and how to run it. The longer note with
  what to look for in the output and exercises is in `docs/lessons/`.
- One test file per lesson in `tests/`. Tests script the model with `ScriptedChatModel` and
  assert on the graph's behaviour (which nodes ran, what state came out), not on prose.
- Data: `data/sample_openings.json` holds 28 real openings exported from jobscout, so the
  repo runs anywhere. When the jobscout DB exists locally the tools use it instead.
- Comments explain a reason or a constraint, never what the next line does.
