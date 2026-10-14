"""Lesson 3: typed results with Pydantic, and a mechanical check against the tool output.

Two ways to get a typed object instead of prose:
  (a) model.with_structured_output(Schema) on a plain model call, here to extract openings
      from the raw search_openings output;
  (b) create_agent(..., response_format=Schema), so the agent's final answer lands in
      result["structured_response"] as a Schema instance.
Then agentlab.grounding.check_grounding compares the typed answer with what the tools
actually returned, which turns the "only state salaries the tools returned" prompt rule
into something a test can fail.

Run: uv run lessons/03_structured_output.py "your question"
     uv run lessons/03_structured_output.py --graph
"""

import sys

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentlab.config import SYSTEM, make_model
from agentlab.display import describe
from agentlab.grounding import check_grounding
from agentlab.tools import TOOLS, search_openings

DEFAULT_QUESTION = "Lowongan engineer di Singapore yang nyebutin gaji? Sebutin 3 aja."

EXTRACT_PROMPT = (
    "Extract every opening in this tool output. Fill the salary fields only when the opening "
    "states a salary, otherwise leave them null."
)


class Opening(BaseModel):
    key: str = Field(description="the opening key exactly as the tool returned it")
    company: str
    title: str
    salary_min: float | None = None
    salary_max: float | None = None
    currency: str | None = Field(default=None, description="ISO code such as SGD or USD")


class Openings(BaseModel):
    openings: list[Opening]


class Answer(BaseModel):
    """The agent's final answer: the openings it is talking about plus a one-line summary."""

    openings: list[Opening]
    summary: str = Field(description="one or two sentences in casual Indonesian")


def extract(model: BaseChatModel, tool_output: str) -> Openings:
    """Way (a): a single structured call over raw tool output, no agent loop."""
    extractor = model.with_structured_output(Openings)
    return extractor.invoke([SystemMessage(EXTRACT_PROMPT), HumanMessage(tool_output)])


def build(model: BaseChatModel | None = None):
    """Way (b): the lesson 2 agent, with its final answer forced into Answer."""
    return create_agent(
        model or make_model(), tools=TOOLS, system_prompt=SYSTEM, response_format=Answer
    )


def check_answer(answer: Answer, messages: list) -> list[str]:
    return check_grounding([o.model_dump() for o in answer.openings], messages)


if __name__ == "__main__":
    agent = build()
    if "--graph" in sys.argv:
        print(agent.get_graph().draw_mermaid())
        sys.exit()
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION

    print("== (a) with_structured_output over one search_openings call")
    raw = search_openings.invoke({"keyword": "engineer", "market": "SG", "salary_only": True})
    for opening in extract(make_model(), raw).openings:
        print(f"  {opening.key}: {opening.salary_min}-{opening.salary_max} {opening.currency}")

    print("== (b) create_agent with response_format=Answer")
    result = agent.invoke({"messages": [HumanMessage(question)]})
    for msg in result["messages"][1:]:
        print(f"  {describe(msg)}")
    answer: Answer = result["structured_response"]
    print(f"  typed: {answer!r}")

    print("== (c) mechanical check against the tool output")
    problems = check_answer(answer, result["messages"])
    print("  ok, every key and salary came from a tool" if not problems else "\n  ".join(problems))
