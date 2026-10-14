"""Lesson 9: a researcher, a writer and a reviewer drafting one application.

Three roles in one StateGraph with explicit state. The researcher compares the opening with
the profile, the writer drafts a short cover note, the reviewer returns a structured verdict.
A conditional edge sends a rejected draft back to the writer with the issues, at most twice.
Before the reviewer model is asked, a regex pass catches em dashes, buzzwords and length, so
the verdict carries those findings even when the model misses them.

Run: uv run lessons/09_application_crew.py [opening key]
     uv run lessons/09_application_crew.py --graph
"""

import json
import sys
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentlab import data
from agentlab.checks import cover_note_issues
from agentlab.config import make_model
from agentlab.profile import load_profile

MAX_REVISIONS = 2
MAX_WORDS = 180

RESEARCHER = (
    "You compare a job opening with a candidate profile. Reply with three bullet lists: "
    "'Meets:' requirements the profile clearly covers, 'Gaps:' requirements it does not cover "
    "or does not mention, 'Ask:' questions to raise in the application. Only use facts "
    "from the opening and the profile. Be brief."
)
WRITER = (
    f"You write a cover note in English, under {MAX_WORDS} words, plain and specific. "
    "Name the role and company. Mention what the candidate has done that matches, name one "
    "real gap honestly, and end with one of the questions from the notes. No greeting line "
    "to a named person, no signature, no claims that are not in the profile, no buzzwords, "
    "no em dashes."
)
REVIEWER = (
    "You review a cover note against the candidate profile and the opening. Reject any claim "
    "the profile does not support, any em dash, and buzzwords such as passionate, seamless, "
    "cutting-edge. List each issue in one line. Approve only when there are no issues."
)


class Review(BaseModel):
    approved: bool
    issues: list[str] = Field(default_factory=list)


class State(TypedDict, total=False):
    opening: dict
    profile: str
    research_notes: str
    draft: str
    review: Review
    revisions: int


def opening_summary(opening: dict) -> str:
    salary = opening.get("salary") or "not stated"
    return (
        f"Title: {opening['title']}\nCompany: {opening['company']}\n"
        f"Locations: {opening['locations']}\nSalary: {salary}\n\n{opening['description']}"
    )


def default_opening_key() -> str:
    return next(r["key"] for r in data.load_sample() if r["salary"])


def build(model: BaseChatModel | None = None):
    model = model or make_model()
    reviewer_model = model.with_structured_output(Review)

    def researcher(state: State) -> dict:
        prompt = f"Opening:\n{opening_summary(state['opening'])}\n\nProfile:\n{state['profile']}"
        reply = model.invoke([SystemMessage(RESEARCHER), HumanMessage(prompt)])
        return {"research_notes": reply.text, "revisions": 0}

    def writer(state: State) -> dict:
        opening = state["opening"]
        prompt = (
            f"Role: {opening['title']} at {opening['company']}\n\n"
            f"Profile:\n{state['profile']}\n\nResearch notes:\n{state['research_notes']}"
        )
        revisions = state.get("revisions", 0)
        if state.get("draft"):
            revisions += 1
            issues = "\n".join(f"- {i}" for i in state["review"].issues)
            prompt += f"\n\nPrevious draft:\n{state['draft']}\n\nFix these issues:\n{issues}"
        reply = model.invoke([SystemMessage(WRITER), HumanMessage(prompt)])
        return {"draft": reply.text, "revisions": revisions}

    def reviewer(state: State) -> dict:
        mechanical = cover_note_issues(state["draft"], MAX_WORDS)
        prompt = (
            f"Opening:\n{opening_summary(state['opening'])}\n\nProfile:\n{state['profile']}"
            f"\n\nCover note:\n{state['draft']}"
        )
        verdict = reviewer_model.invoke([SystemMessage(REVIEWER), HumanMessage(prompt)])
        # Mechanical findings always count; the model cannot approve past them.
        review = Review(
            approved=verdict.approved and not mechanical, issues=mechanical + verdict.issues
        )
        return {"review": review}

    def route(state: State) -> str:
        if state["review"].approved or state["revisions"] >= MAX_REVISIONS:
            return END
        return "writer"

    builder = StateGraph(State)
    builder.add_node("researcher", researcher)
    builder.add_node("writer", writer)
    builder.add_node("reviewer", reviewer)
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "reviewer")
    builder.add_conditional_edges("reviewer", route, ["writer", END])
    return builder.compile()


def print_steps(stream) -> None:
    for update in stream:
        for node, output in update.items():
            if node == "researcher":
                print(f"[researcher]\n{output['research_notes']}\n")
            elif node == "writer":
                print(f"[writer] revision {output['revisions']}\n{output['draft']}\n")
            else:
                review = output["review"]
                print(f"[reviewer] {json.dumps(review.model_dump(), indent=1)}\n")


if __name__ == "__main__":
    graph = build()
    if "--graph" in sys.argv:
        print(graph.get_graph().draw_mermaid())
        sys.exit()
    key = sys.argv[1] if len(sys.argv) > 1 else default_opening_key()
    opening = data.get_opening(key)
    if not opening:
        sys.exit(f"No opening with key {key}")
    print(f"Opening: {opening['title']} at {opening['company']}\n")
    print_steps(
        graph.stream({"opening": opening, "profile": load_profile()}, stream_mode="updates")
    )
