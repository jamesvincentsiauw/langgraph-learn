"""Mechanical checks over an agent's final message list, used by lesson 11.

A check takes the messages plus keyword arguments from the dataset and returns a CheckResult.
CHECKS maps the names used in data/evals.json to the functions, so adding a check means
writing one function and registering it.
"""

import json
import re
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

# A currency code or Rp next to a number, or a number followed by an Indonesian amount word.
SALARY_RE = re.compile(
    r"\b(rp|sgd|usd|jpy|nzd|idr|cad)\.?\s*\d[\d.,]*|\d[\d.,]*\s*(juta|miliar|ribu|/tahun|/bulan)",
    re.IGNORECASE,
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def final_answer(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return msg.text
    return ""


def tool_calls(messages: list[BaseMessage]) -> list[dict]:
    return [call for m in messages if isinstance(m, AIMessage) for call in m.tool_calls]


def returned_openings(messages: list[BaseMessage]) -> list[dict]:
    """Openings the search tool actually returned, parsed back from its JSON output."""
    rows = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "search_openings":
            try:
                rows.extend(json.loads(str(msg.content)))
            except json.JSONDecodeError:
                continue
    return rows


def _args_match(actual: dict, expected: dict | None) -> bool:
    if not expected:
        return True
    return all(str(actual.get(k, "")).lower() == str(v).lower() for k, v in expected.items())


def called_tool(messages: list[BaseMessage], name: str, args: dict | None = None) -> CheckResult:
    calls = [c for c in tool_calls(messages) if c["name"] == name]
    hit = any(_args_match(c["args"], args) for c in calls)
    seen = [c["args"] for c in calls] or "no call"
    return CheckResult("called_tool", hit, f"{name} wanted {args or 'any args'}, saw {seen}")


def not_called_tool(messages: list[BaseMessage], name: str) -> CheckResult:
    count = sum(1 for c in tool_calls(messages) if c["name"] == name)
    return CheckResult("not_called_tool", count == 0, f"{name} called {count} times")


def final_is_text(messages: list[BaseMessage]) -> CheckResult:
    last = messages[-1] if messages else None
    ok = isinstance(last, AIMessage) and not last.tool_calls and bool(last.text.strip())
    return CheckResult("final_is_text", ok, f"last message type: {getattr(last, 'type', None)}")


def mentions_returned_company(messages: list[BaseMessage]) -> CheckResult:
    companies = {r["company"] for r in returned_openings(messages)}
    answer = final_answer(messages).lower()
    named = sorted(c for c in companies if c.lower() in answer)
    return CheckResult(
        "mentions_returned_company",
        bool(named),
        f"named {named or 'none'} of {sorted(companies) or 'no results'}",
    )


def no_salary_unless_returned(messages: list[BaseMessage]) -> CheckResult:
    """If none of the returned openings state a salary, the answer must not state one."""
    with_salary = [r for r in returned_openings(messages) if r.get("salary")]
    figures = [m.group(0) for m in SALARY_RE.finditer(final_answer(messages))]
    if with_salary:
        return CheckResult("no_salary_unless_returned", True, "tools returned salaries")
    return CheckResult(
        "no_salary_unless_returned", not figures, f"figures in answer: {figures or 'none'}"
    )


def answer_mentions(messages: list[BaseMessage], text: str) -> CheckResult:
    ok = text.lower() in final_answer(messages).lower()
    return CheckResult("answer_mentions", ok, f"{text!r} {'found' if ok else 'missing'}")


CHECKS = {
    "called_tool": called_tool,
    "not_called_tool": not_called_tool,
    "final_is_text": final_is_text,
    "mentions_returned_company": mentions_returned_company,
    "no_salary_unless_returned": no_salary_unless_returned,
    "answer_mentions": answer_mentions,
}


def run_checks(messages: list[BaseMessage], specs: list[dict]) -> list[CheckResult]:
    """Each spec is {"check": name, ...kwargs}, as written in data/evals.json."""
    results = []
    for spec in specs:
        kwargs = {k: v for k, v in spec.items() if k != "check"}
        results.append(CHECKS[spec["check"]](messages, **kwargs))
    return results
