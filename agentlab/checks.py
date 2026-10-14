"""Mechanical checks on drafted text, run before an LLM reviewer sees it.

A reviewer prompt that says "reject em dashes and buzzwords" is a wish. A regex is a check.
Lesson 9 merges these findings into the reviewer's verdict so the loop cannot approve a draft
that fails them, whatever the model says.
"""

import re

# chr() keeps the character out of the source; the repo bans em dashes in any text.
EM_DASH = re.compile(chr(0x2014))

BUZZWORDS = (
    "passionate",
    "seamless",
    "cutting-edge",
    "synergy",
    "leverage",
    "rockstar",
    "ninja",
    "world-class",
    "game-changing",
    "best-in-class",
    "proven track record",
    "results-driven",
    "thrilled",
)
BUZZWORD_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in BUZZWORDS) + r")\b", re.I)


def find_em_dashes(text: str) -> list[str]:
    count = len(EM_DASH.findall(text))
    return [f"contains {count} em dash(es); use a comma, period or colon"] if count else []


def find_buzzwords(text: str) -> list[str]:
    found = sorted({m.group(0).lower() for m in BUZZWORD_RE.finditer(text)})
    return [f"buzzword: {w}" for w in found]


def cover_note_issues(text: str, max_words: int = 180) -> list[str]:
    """Everything a cover note fails mechanically: em dashes, buzzwords, length."""
    issues = find_em_dashes(text) + find_buzzwords(text)
    words = len(text.split())
    if words >= max_words:
        issues.append(f"{words} words, must be under {max_words}")
    return issues
