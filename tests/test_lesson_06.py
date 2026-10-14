import importlib
import json

from langgraph.types import Send
from pydantic import BaseModel

from agentlab.fakes import KeyedChatModel

lesson = importlib.import_module("06_pipeline_graph")


def opening(key: str, company: str, title: str, regions: list[str]) -> dict:
    return {
        "key": key,
        "company": company,
        "title": title,
        "regions": json.dumps(regions),
        "locations": "x",
        "remote": "remote",
        "description": "d",
    }


def test_filter_drops_other_markets_and_dedupes():
    rows = [
        opening("a", "Halter", "Mobile Engineer", ["NZ"]),
        opening("b", "halter", "mobile engineer", ["NZ"]),
        opening("c", "Wise", "Android Engineer", ["SG"]),
        opening("d", "Codekeeper", "Back End Engineer", ["REMOTE"]),
    ]
    kept = lesson.keep_wanted(rows, ["nz", "SG"])
    assert [o["key"] for o in kept] == ["a", "c"]


def test_fan_out_sends_one_per_opening():
    rows = [opening("a", "A", "t", ["SG"]), opening("b", "B", "t", ["SG"])]
    sends = lesson.fan_out({"profile": "p", "openings": rows})
    assert [s.node for s in sends] == ["score", "score"]
    assert [s.arg["opening"]["key"] for s in sends] == ["a", "b"]
    assert sends[0].arg["profile"] == "p"
    assert lesson.fan_out({"profile": "p", "openings": []}) == "rank"


def test_rank_picks_top_n_by_fit():
    rows = [opening(k, k.upper(), "t", ["SG"]) for k in "abc"]
    scores = [
        lesson.Score(key="a", fit=3, reason="meh"),
        lesson.Score(key="b", fit=9, reason="yes"),
        lesson.Score(key="c", fit=6, reason="ok"),
    ]
    top = lesson.pick_top(rows, scores, 2)
    assert [(r["key"], r["fit"]) for r in top] == [("b", 9), ("c", 6)]
    assert top[0]["company"] == "B"


def test_graph_scores_each_opening_in_its_own_run():
    # The sample has four NZ openings with "engineer" in the title: micro1 and three at Halter.
    # Send runs finish in any order, so replies are keyed by the title found in the prompt.
    replies = {
        "Junior Software Engineer": '{"fit": 2, "reason": "junior"}',
        "Lead React Native Engineer": '{"fit": 1, "reason": "mobile"}',
        "Senior Mobile Engineer": '{"fit": 3, "reason": "mobile"}',
        "Software Engineering Manager": '{"fit": 5, "reason": "management"}',
    }
    graph = lesson.build(KeyedChatModel(replies=replies))
    state = lesson.initial_state("engineer", ["NZ"], top_n=2)
    state["profile"] = "backend"
    score_runs = 0
    for update in graph.stream(state, stream_mode="updates"):
        score_runs += "score" in update
        final = update
    assert score_runs == 4
    shortlist = final["rank"]["shortlist"]
    assert [(r["company"], r["fit"]) for r in shortlist] == [("Halter", 5), ("Halter", 3)]


def test_keyed_fake_picks_reply_by_prompt_content():
    class V(BaseModel):
        fit: int

    fake = KeyedChatModel(replies={"Halter": '{"fit": 7}', "micro1": '{"fit": 2}'})
    judge = fake.with_structured_output(V)
    assert judge.invoke("Opening: micro1: Junior").fit == 2
    assert judge.invoke("Opening: Halter: Senior").fit == 7
    assert isinstance(Send("score", {}), Send)
