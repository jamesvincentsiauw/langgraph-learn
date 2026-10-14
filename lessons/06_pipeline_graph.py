"""Lesson 6: a pipeline graph where the model only does the judging.

`load`, `filter` and `rank` are plain Python and testable without a model. Only `score` calls
the model, once per opening, and those runs happen in parallel because the `fan_out` edge
returns one `Send` per opening. A reducer on `scores` collects the results.

Run: uv run lessons/06_pipeline_graph.py engineer --markets SG,NZ
     uv run lessons/06_pipeline_graph.py --graph
"""

import json
import operator
import sys
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from agentlab import data
from agentlab.config import make_model
from agentlab.profile import load_profile

MARKETS = ["SG", "JP", "NZ", "REMOTE"]
TOP_N = 3


class Verdict(BaseModel):
    fit: int = Field(ge=0, le=10, description="How well the opening fits the profile, 0 to 10")
    reason: str = Field(description="One line on why")


class Score(Verdict):
    # The key is attached by code after the model answers: identifiers are never typed by a model.
    key: str


class State(TypedDict):
    profile: str
    keyword: str
    markets: list[str]
    top_n: int
    openings: list[dict]
    # Each parallel score run returns a one-item list; operator.add appends instead of replacing.
    scores: Annotated[list[Score], operator.add]
    shortlist: list[dict]


class ScoreInput(TypedDict):
    profile: str
    opening: dict


def keep_wanted(openings: list[dict], markets: list[str]) -> list[dict]:
    """Drop openings outside the wanted markets and repeats of the same company + title."""
    wanted = {m.upper() for m in markets}
    seen: set[tuple[str, str]] = set()
    kept = []
    for opening in openings:
        regions = set(json.loads(opening["regions"]))
        ident = (opening["company"].strip().lower(), opening["title"].strip().lower())
        if regions & wanted and ident not in seen:
            seen.add(ident)
            kept.append(opening)
    return kept


def pick_top(openings: list[dict], scores: list[Score], n: int) -> list[dict]:
    by_key = {o["key"]: o for o in openings}
    ranked = sorted(scores, key=lambda s: s.fit, reverse=True)[:n]
    return [
        {
            "key": s.key,
            "company": by_key[s.key]["company"],
            "title": by_key[s.key]["title"],
            "fit": s.fit,
            "reason": s.reason,
        }
        for s in ranked
    ]


def fan_out(state: State) -> str | list[Send]:
    if not state["openings"]:
        return "rank"
    return [Send("score", {"profile": state["profile"], "opening": o}) for o in state["openings"]]


def build(model: BaseChatModel | None = None):
    judge = (model or make_model()).with_structured_output(Verdict)

    def load(state: State) -> dict:
        return {"openings": data.query_openings(state["keyword"], limit=30)}

    def filter_openings(state: State) -> dict:
        return {"openings": keep_wanted(state["openings"], state["markets"])}

    def score(state: ScoreInput) -> dict:
        o = state["opening"]
        prompt = (
            f"Candidate profile:\n{state['profile']}\n\n"
            f"Opening: {o['company']}: {o['title']} ({o['locations']}, {o['remote']})\n"
            f"{o['description'][:800]}\n\n"
            "Rate how well this opening fits the candidate."
        )
        verdict = judge.invoke(prompt)
        return {"scores": [Score(key=o["key"], **verdict.model_dump())]}

    def rank(state: State) -> dict:
        return {"shortlist": pick_top(state["openings"], state["scores"], state["top_n"])}

    builder = StateGraph(State)
    builder.add_node("load", load)
    builder.add_node("filter", filter_openings)
    builder.add_node("score", score)
    builder.add_node("rank", rank)
    builder.add_edge(START, "load")
    builder.add_edge("load", "filter")
    builder.add_conditional_edges("filter", fan_out, ["score", "rank"])
    builder.add_edge("score", "rank")
    builder.add_edge("rank", END)
    return builder.compile()


def initial_state(keyword: str, markets: list[str], top_n: int = TOP_N) -> State:
    return {
        "profile": load_profile(),
        "keyword": keyword,
        "markets": markets,
        "top_n": top_n,
        "openings": [],
        "scores": [],
        "shortlist": [],
    }


if __name__ == "__main__":
    graph = build()
    if "--graph" in sys.argv:
        print(graph.get_graph().draw_mermaid())
        sys.exit()
    args = sys.argv[1:]
    markets = MARKETS
    if "--markets" in args:
        i = args.index("--markets")
        markets = args[i + 1].split(",")
        del args[i : i + 2]
    keyword = " ".join(args) or "engineer"
    for update in graph.stream(initial_state(keyword, markets), stream_mode="updates"):
        for node, out in update.items():
            if node == "load":
                print(f"[load] {len(out['openings'])} openings with {keyword!r} in the title")
            elif node == "filter":
                print(f"[filter] {len(out['openings'])} left for markets {markets} after dedupe")
            elif node == "score":
                s = out["scores"][0]
                print(f"[score] {s.fit:>2}  {s.key}: {s.reason}")
            elif node == "rank":
                for i, row in enumerate(out["shortlist"], 1):
                    print(f"[rank] {i}. {row['company']}: {row['title']} (fit {row['fit']})")
