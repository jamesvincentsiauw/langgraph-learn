"""Lesson 11: a small eval harness for the lesson-2 agent.

data/evals.json holds questions in casual Indonesian, each with mechanical checks that hold
for the sample data (which tool must be called, with which args, which company must be
named, no salary figures when the tools returned none). agentlab/evals.py runs the checks.
One LLM judge with structured output scores what a regex cannot: is the answer in casual
Indonesian, and does it only state salaries the tools returned. Mechanical checks decide the
exit code; the judge score is reported next to them.

Run: uv run lessons/11_evals.py            (exit code 1 if any mechanical check fails)
     uv run lessons/11_evals.py --graph
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from agentlab.config import SYSTEM, make_model
from agentlab.data import ROOT
from agentlab.evals import CheckResult, final_answer, run_checks
from agentlab.tools import TOOLS

CASES_PATH = ROOT / "data" / "evals.json"

JUDGE_PROMPT = """You grade one answer from a job-search assistant for Vincent, who writes in
casual Indonesian. Score 1 to 5 on two things together:
1. The answer is written in casual Indonesian (lu/gua or aku/kamu register, not formal
   Bahasa Indonesia, not English).
2. Every salary figure in the answer comes from the tool outputs below, including IDR
   conversions. Any salary or exchange rate the tools did not return is a fabrication.
5 means both hold fully, 1 means both fail. Give a one-sentence reason.

Tool outputs:
{tool_outputs}

Answer:
{answer}"""


class Verdict(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str


@dataclass
class CaseResult:
    name: str
    checks: list[CheckResult]
    verdict: Verdict
    seconds: float
    answer: str = field(repr=False)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.ok)

    @property
    def all_passed(self) -> bool:
        return self.passed == len(self.checks)


def build(model: BaseChatModel | None = None):
    return create_agent(model or make_model(), tools=TOOLS, system_prompt=SYSTEM)


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return json.loads(path.read_text())


def tool_output_for_judge(msg: ToolMessage) -> str:
    """Search results shrink to company, title and salary so nothing the judge needs is cut."""
    if msg.name != "search_openings" or not str(msg.content).startswith("["):
        return f"- {msg.name}: {str(msg.content)[:600]}"
    rows = json.loads(str(msg.content))
    lines = [f"  {r['company']}, {r['title']}: salary {r['salary'] or 'not stated'}" for r in rows]
    return f"- {msg.name}:\n" + "\n".join(lines)


def judge(messages: list[BaseMessage], judge_model: BaseChatModel) -> Verdict:
    tool_outputs = "\n".join(
        tool_output_for_judge(m) for m in messages if isinstance(m, ToolMessage)
    )
    prompt = JUDGE_PROMPT.format(
        tool_outputs=tool_outputs or "(none)", answer=final_answer(messages)
    )
    return judge_model.with_structured_output(Verdict).invoke(prompt)


def run_case(agent, case: dict, judge_model: BaseChatModel) -> CaseResult:
    started = time.perf_counter()
    messages = agent.invoke({"messages": [HumanMessage(case["question"])]})["messages"]
    checks = run_checks(messages, case["checks"])
    verdict = judge(messages, judge_model)
    return CaseResult(
        name=case["name"],
        checks=checks,
        verdict=verdict,
        seconds=time.perf_counter() - started,
        answer=final_answer(messages),
    )


def run_all(agent, cases: list[dict], judge_model: BaseChatModel) -> list[CaseResult]:
    return [run_case(agent, case, judge_model) for case in cases]


def print_report(results: list[CaseResult]) -> None:
    width = max(len(r.name) for r in results)
    print(f"{'case':<{width}}  checks  judge  seconds")
    for r in results:
        checks = f"{r.passed}/{len(r.checks)}"
        print(f"{r.name:<{width}}  {checks:<6}  {r.verdict.score}/5    {r.seconds:5.1f}")
        for c in r.checks:
            if not c.ok:
                print(f"{'':<{width}}  FAIL {c.name}: {c.detail}")
        if r.verdict.score < 4:
            print(f"{'':<{width}}  judge: {r.verdict.reason}")
    clean = sum(1 for r in results if r.all_passed)
    mean = sum(r.verdict.score for r in results) / len(results)
    total = sum(r.seconds for r in results)
    print(
        f"\n{clean}/{len(results)} cases passed every check, mean judge {mean:.1f}/5, {total:.0f}s"
    )


def exit_code(results: list[CaseResult]) -> int:
    return 0 if all(r.all_passed for r in results) else 1


if __name__ == "__main__":
    agent = build()
    if "--graph" in sys.argv:
        print(agent.get_graph().draw_mermaid())
        sys.exit()
    results = run_all(agent, load_cases(), judge_model=make_model())
    print_report(results)
    sys.exit(exit_code(results))
