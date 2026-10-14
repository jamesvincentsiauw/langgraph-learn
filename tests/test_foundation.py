import json

from langchain_core.messages import HumanMessage

from agentlab import data
from agentlab.fakes import ScriptedChatModel, tool_call
from agentlab.tools import save_to_shortlist, search_openings


def test_sample_data_used_when_db_missing():
    rows = data.query_openings("engineer", market="SG")
    assert rows and all("SG" in json.loads(r["regions"]) for r in rows)


def test_search_tool_strips_description():
    out = json.loads(search_openings.invoke({"keyword": "engineer"}))
    assert out and "description" not in out[0]


def test_shortlist_round_trip():
    assert "Saved" in save_to_shortlist.invoke({"key": "x", "note": "fits"})
    assert "already" in save_to_shortlist.invoke({"key": "x", "note": "fits"})
    assert data.read_shortlist()[0]["key"] == "x"


def test_scripted_model_runs_in_order():
    model = ScriptedChatModel([tool_call("search_openings", {"keyword": "go"}), "done"])
    first = model.bind_tools([search_openings]).invoke([HumanMessage("hi")])
    assert first.tool_calls[0]["name"] == "search_openings"
    assert model.invoke([HumanMessage("hi")]).content == "done"


def test_scripted_model_streams_text_and_tool_calls():
    model = ScriptedChatModel([tool_call("search_openings", {"keyword": "go"}), "dua kata"])
    call_chunks = list(model.stream([HumanMessage("hi")]))
    assert len(call_chunks) == 1 and call_chunks[0].tool_calls[0]["name"] == "search_openings"
    assert [c.content for c in model.stream([HumanMessage("hi")])] == ["dua", " ", "kata"]
