import importlib
import json

from agentlab import data
from agentlab.checks import cover_note_issues
from agentlab.fakes import ScriptedChatModel

lesson = importlib.import_module("09_application_crew")

APPROVE = json.dumps({"approved": True, "issues": []})
REJECT = json.dumps({"approved": False, "issues": ["claims Kubernetes, profile does not"]})


def run(script):
    graph = lesson.build(ScriptedChatModel(script))
    opening = data.get_opening("himalayas:backend-engineer-golang-python")
    nodes, reviews, state = [], [], {}
    for update in graph.stream({"opening": opening, "profile": "Go and Kafka."}):
        for node, output in update.items():
            nodes.append(node)
            state.update(output)
            if node == "reviewer":
                reviews.append(output["review"])
    return nodes, reviews, state


def test_loop_ends_on_approval():
    nodes, reviews, state = run(["notes", "Clean draft.", APPROVE])
    assert nodes == ["researcher", "writer", "reviewer"]
    assert state["review"].approved and state["revisions"] == 0
    assert state["draft"] == "Clean draft."


def test_loop_ends_after_two_revisions():
    script = ["notes", "d1", REJECT, "d2", REJECT, "d3", REJECT]
    nodes, reviews, state = run(script)
    assert nodes == ["researcher", "writer", "reviewer"] + ["writer", "reviewer"] * 2
    assert state["revisions"] == 2 and not state["review"].approved
    assert state["draft"] == "d3"


def test_mechanical_check_overrides_a_model_approval():
    script = [
        "notes",
        "I am thrilled " + chr(0x2014) + " really.",
        APPROVE,
        "Fixed draft.",
        APPROVE,
    ]
    nodes, reviews, state = run(script)
    assert nodes == ["researcher", "writer", "reviewer", "writer", "reviewer"]
    first = reviews[0]
    assert not first.approved
    assert any("em dash" in i for i in first.issues)
    assert any("thrilled" in i for i in first.issues)
    assert state["review"].approved and state["revisions"] == 1


def test_cover_note_issues_flags_length():
    assert cover_note_issues("word " * 200) == ["200 words, must be under 180"]
    assert cover_note_issues("Short and plain.") == []


def test_default_opening_has_a_salary():
    assert data.get_opening(lesson.default_opening_key())["salary"]
