import importlib
import json

from langchain_core.messages import HumanMessage, ToolMessage

from agentlab.fakes import ScriptedChatModel, tool_call
from agentlab.grounding import check_grounding
from agentlab.tools import search_openings

lesson = importlib.import_module("03_structured_output")

SIERRA = "ashby:sierra/79953d72-60d4-43e0-8c8c-6eccda422dce"
WISE = "smartrecruiters:Wise/744000145649689"


def search_message(**kwargs) -> ToolMessage:
    return ToolMessage(
        content=search_openings.invoke(kwargs), tool_call_id="call_1", name="search_openings"
    )


def test_extract_parses_structured_output_into_schema():
    payload = {
        "openings": [
            {
                "key": SIERRA,
                "company": "Sierra",
                "title": "SWE",
                "salary_min": 200000,
                "salary_max": 240000,
                "currency": "SGD",
            },
            {"key": WISE, "company": "Wise", "title": "Android Engineer - Spend"},
        ]
    }
    result = lesson.extract(ScriptedChatModel([json.dumps(payload)]), "raw tool output")
    assert isinstance(result, lesson.Openings)
    assert result.openings[0].salary_min == 200000
    assert result.openings[1].salary_min is None and result.openings[1].currency is None


def test_agent_puts_typed_answer_in_structured_response():
    answer = {
        "openings": [{"key": SIERRA, "company": "Sierra", "title": "SWE"}],
        "summary": "Ada satu.",
    }
    script = [
        tool_call("search_openings", {"keyword": "engineer", "market": "SG"}),
        # With a model that has no provider-native structured output, create_agent registers
        # the schema as a tool named after the class and parses the call it makes.
        tool_call("Answer", answer, "call_2"),
    ]
    result = lesson.build(ScriptedChatModel(script)).invoke({"messages": [HumanMessage("q")]})
    typed = result["structured_response"]
    assert isinstance(typed, lesson.Answer)
    assert typed.openings[0].key == SIERRA
    assert [m.type for m in result["messages"]] == ["human", "ai", "tool", "ai", "tool"]


def test_check_passes_when_keys_and_salaries_match_tool_output():
    msgs = [search_message(keyword="engineer", market="SG", salary_only=True)]
    claims = [
        {"key": SIERRA, "salary_min": 200000, "salary_max": 240000, "currency": "SGD"},
        {"key": SIERRA, "salary_min": None, "salary_max": None, "currency": None},
    ]
    assert check_grounding(claims, msgs) == []


def test_check_flags_unknown_key_and_invented_salary():
    # search_openings returns at most 5 rows, so the Wise row needs its own search.
    msgs = [
        search_message(keyword="engineer", market="SG"),
        search_message(keyword="android", market="SG"),
    ]
    claims = [
        {"key": "ashby:nope/123", "salary_min": None, "salary_max": None, "currency": None},
        {"key": WISE, "salary_min": 100000, "salary_max": None, "currency": "SGD"},
        {"key": SIERRA, "salary_min": 200000, "salary_max": 999999, "currency": "SGD"},
    ]
    problems = check_grounding(claims, msgs)
    assert len(problems) == 3
    assert "not in any tool output" in problems[0]
    assert "tool output has none" in problems[1]
    assert "max 999999" in problems[2]


def test_check_answer_uses_the_agent_messages():
    answer = lesson.Answer(
        openings=[lesson.Opening(key=WISE, company="Wise", title="Android", currency="SGD")],
        summary="x",
    )
    msgs = [search_message(keyword="android", market="SG")]
    assert lesson.check_answer(answer, msgs) == [
        f"{WISE}: claims a salary but the tool output has none"
    ]
