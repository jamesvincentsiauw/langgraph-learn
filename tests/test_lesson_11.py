import importlib
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentlab import evals
from agentlab.fakes import ScriptedChatModel, tool_call

lesson = importlib.import_module("11_evals")

SG_ROWS = json.dumps(
    [
        {"key": "a", "company": "Sierra", "title": "SWE", "salary": '{"min": 200000}'},
        {"key": "b", "company": "Wise", "title": "iOS", "salary": ""},
    ]
)
NZ_ROWS = json.dumps([{"key": "c", "company": "Halter", "title": "Mobile", "salary": ""}])


def transcript(call: AIMessage, result: str, answer: str) -> list:
    return [
        HumanMessage("q"),
        call,
        ToolMessage(result, tool_call_id=call.tool_calls[0]["id"], name=call.tool_calls[0]["name"]),
        AIMessage(answer),
    ]


SG = transcript(
    tool_call("search_openings", {"keyword": "engineer", "market": "SG"}),
    SG_ROWS,
    "Ada di Sierra, gajinya SGD 200.000 per tahun.",
)
NZ_CLEAN = transcript(
    tool_call("search_openings", {"keyword": "mobile", "market": "NZ"}),
    NZ_ROWS,
    "Halter lagi buka.",
)
NZ_MADE_UP = transcript(
    tool_call("search_openings", {"keyword": "mobile", "market": "NZ"}),
    NZ_ROWS,
    "Halter lagi buka, gajinya sekitar NZD 120.000.",
)


def test_called_tool_matches_args_case_insensitively():
    assert evals.called_tool(SG, "search_openings", {"market": "sg"}).ok
    assert not evals.called_tool(SG, "search_openings", {"market": "JP"}).ok
    assert not evals.called_tool(SG, "convert_to_idr").ok
    assert evals.not_called_tool(SG, "convert_to_idr").ok


def test_mentions_returned_company_uses_tool_output_not_prose():
    assert evals.mentions_returned_company(SG).detail == "named ['Sierra'] of ['Sierra', 'Wise']"
    no_results = transcript(
        tool_call("search_openings", {"keyword": "x"}), "No openings found.", "Sierra ada."
    )
    assert not evals.mentions_returned_company(no_results).ok


def test_no_salary_unless_returned():
    assert evals.no_salary_unless_returned(SG).ok
    assert evals.no_salary_unless_returned(NZ_CLEAN).ok
    result = evals.no_salary_unless_returned(NZ_MADE_UP)
    assert not result.ok and "NZD 120" in result.detail


def test_final_is_text_and_answer_mentions():
    assert evals.final_is_text(SG).ok
    assert not evals.final_is_text(SG[:-1]).ok
    assert evals.answer_mentions(SG, "sierra").ok
    assert not evals.answer_mentions(SG, "Halter").ok


def test_run_checks_reads_specs_from_json_shape():
    specs = [
        {"check": "called_tool", "name": "search_openings", "args": {"market": "SG"}},
        {"check": "answer_mentions", "text": "Wise"},
    ]
    assert [r.ok for r in evals.run_checks(SG, specs)] == [True, False]


def test_dataset_checks_are_registered():
    for case in lesson.load_cases():
        assert all(spec["check"] in evals.CHECKS for spec in case["checks"]), case["name"]


def test_runner_aggregates_scripted_agent_and_judge():
    cases = [
        {
            "name": "sg",
            "question": "lowongan SG?",
            "checks": [
                {"check": "called_tool", "name": "search_openings", "args": {"market": "SG"}},
                {"check": "mentions_returned_company"},
            ],
        },
        {
            "name": "nz",
            "question": "lowongan NZ?",
            "checks": [
                {"check": "called_tool", "name": "search_openings", "args": {"market": "NZ"}}
            ],
        },
    ]
    agent_model = ScriptedChatModel(
        [
            tool_call("search_openings", {"keyword": "engineer", "market": "SG"}),
            "Sierra lagi buka.",
            tool_call("search_openings", {"keyword": "mobile", "market": "JP"}),
            "Gak ada.",
        ]
    )
    judge_model = ScriptedChatModel(
        ['{"score": 5, "reason": "santai"}', '{"score": 2, "reason": "formal"}']
    )
    results = lesson.run_all(lesson.build(agent_model), cases, judge_model=judge_model)
    assert [(r.name, r.passed, len(r.checks)) for r in results] == [("sg", 2, 2), ("nz", 0, 1)]
    assert [r.verdict.score for r in results] == [5, 2]
    assert results[1].verdict.reason == "formal"
    assert all(r.seconds >= 0 for r in results)
    assert lesson.exit_code(results) == 1
    assert lesson.exit_code(results[:1]) == 0


def test_report_prints_table_and_summary(capsys):
    checks = [
        evals.CheckResult("called_tool", True),
        evals.CheckResult("final_is_text", False, "x"),
    ]
    results = [
        lesson.CaseResult("sg", checks, lesson.Verdict(score=3, reason="agak formal"), 1.2, "a")
    ]
    lesson.print_report(results)
    out = capsys.readouterr().out
    assert "sg" in out and "1/2" in out and "3/5" in out
    assert "FAIL final_is_text" in out and "agak formal" in out
    assert "0/1 cases passed every check, mean judge 3.0/5" in out


def test_judge_sees_slim_search_rows_and_scripted_verdict():
    judge_model = ScriptedChatModel(['{"score": 4, "reason": "ada angka"}'])
    verdict = lesson.judge(SG, judge_model)
    assert (verdict.score, verdict.reason) == (4, "ada angka")
    prompt = judge_model.seen[-1][-1].text
    assert "Sierra, " in prompt and "Wise, " in prompt and "not stated" in prompt
    assert "Ada di Sierra" in prompt
