"""Mechanical check that a structured answer only contains facts the tools returned.

The system prompt says "only state salaries the tools returned". A prompt is a wish; this
module makes it a test. Facts are rebuilt from the ToolMessages in the run, then each claimed
opening is compared against them.
"""

import json

from langchain_core.messages import BaseMessage


def facts_from_messages(messages: list[BaseMessage]) -> dict[str, dict]:
    """Openings the tools returned in this run, keyed by opening key."""
    facts: dict[str, dict] = {}
    for msg in messages:
        if msg.type != "tool":
            continue
        try:
            payload = json.loads(str(msg.content))
        except json.JSONDecodeError:
            continue
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if isinstance(row, dict) and "key" in row:
                facts[row["key"]] = row
    return facts


def salary_of(row: dict) -> dict | None:
    """The salary dict stored on a row, or None when the opening states none."""
    raw = row.get("salary")
    if not raw:
        return None
    return json.loads(raw) if isinstance(raw, str) else raw


def check_grounding(claims: list[dict], messages: list[BaseMessage]) -> list[str]:
    """Problems with the claimed openings; an empty list means every claim is grounded.

    Each claim is a dict with key, salary_min, salary_max and currency (None when not stated).
    """
    facts = facts_from_messages(messages)
    problems = []
    for claim in claims:
        key = claim.get("key")
        if key not in facts:
            problems.append(f"{key}: not in any tool output")
            continue
        claimed = {
            "min": claim.get("salary_min"),
            "max": claim.get("salary_max"),
            "currency": claim.get("currency"),
        }
        if all(v is None for v in claimed.values()):
            continue
        stated = salary_of(facts[key])
        if stated is None:
            problems.append(f"{key}: claims a salary but the tool output has none")
            continue
        for field, value in claimed.items():
            if value is not None and value != stated.get(field):
                problems.append(f"{key}: {field} {value!r} but the tool said {stated.get(field)!r}")
    return problems
