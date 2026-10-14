"""Lesson 8: a supervisor that hands work to specialist agents.

Two specialists are built with create_agent, each with only its own tools. The supervisor is
a plain node whose model is bound to one handoff tool per specialist. A handoff is a
Command(goto=...) returned from the supervisor node; the specialist runs its subgraph on the
shared history and returns Command(goto="supervisor") with its answer as the tool result.

Run: uv run lessons/08_supervisor.py "your question"
     uv run lessons/08_supervisor.py --graph
"""

import sys
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

from agentlab.config import DEFAULT_QUESTION, make_model
from agentlab.display import describe
from agentlab.tools import convert_to_idr, get_opening_details, monthly_salary, search_openings

SUPERVISOR = (
    "You coordinate two specialists for Vincent, a backend engineer in Jakarta. "
    "job_scout searches openings and reads their details. salary_analyst turns a stated "
    "salary into a monthly figure and into IDR. Hand off one task at a time with the "
    "transfer_to_* tools; the specialist's reply comes back as the tool result. When you have "
    "what you need, answer in casual Indonesian without calling a tool. "
    "Only state facts the specialists returned."
)
JOB_SCOUT = (
    "You are job_scout. Use your tools to find openings and read their details. Reply in "
    "English with the key, company, title, locations and stated salary of each opening. "
    "Do not convert currencies or guess exchange rates; salary_analyst does that. Keep it brief."
)
SALARY_ANALYST = (
    "You are salary_analyst. Use your tools to express a salary per month and in IDR. "
    "Reply in English with the tool results only, without commenting on earlier messages. "
    "Keep it brief."
)

Specialist = Literal["job_scout", "salary_analyst"]


@tool
def transfer_to_job_scout(task: str) -> str:
    """Hand a search or opening-details task to job_scout. Say what to find."""
    return task


@tool
def transfer_to_salary_analyst(task: str) -> str:
    """Hand a salary conversion task to salary_analyst. Include amount, currency, period."""
    return task


HANDOFFS = [transfer_to_job_scout, transfer_to_salary_analyst]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_job_scout(model: BaseChatModel | None = None):
    return create_agent(
        model or make_model(),
        tools=[search_openings, get_opening_details],
        system_prompt=JOB_SCOUT,
        name="job_scout",
    )


def build_salary_analyst(model: BaseChatModel | None = None):
    return create_agent(
        model or make_model(),
        tools=[convert_to_idr, monthly_salary],
        system_prompt=SALARY_ANALYST,
        name="salary_analyst",
    )


def build(
    model: BaseChatModel | None = None,
    job_scout_model: BaseChatModel | None = None,
    salary_analyst_model: BaseChatModel | None = None,
):
    supervisor_model = (model or make_model()).bind_tools(HANDOFFS)
    specialists = {
        "job_scout": build_job_scout(job_scout_model),
        "salary_analyst": build_salary_analyst(salary_analyst_model),
    }

    def supervisor(state: State) -> Command[Literal["job_scout", "salary_analyst", "__end__"]]:
        reply = supervisor_model.invoke([SystemMessage(SUPERVISOR), *state["messages"]])
        if not reply.tool_calls:
            return Command(goto=END, update={"messages": [reply]})
        # Several handoffs in one reply run as parallel nodes; each answers its own call id.
        targets = [c["name"].removeprefix("transfer_to_") for c in reply.tool_calls]
        return Command(goto=targets, update={"messages": [reply]})

    def specialist_node(name: Specialist):
        agent = specialists[name]

        def run(state: State) -> Command[Literal["supervisor"]]:
            handoff = state["messages"][-1]
            call = next(c for c in handoff.tool_calls if c["name"] == f"transfer_to_{name}")
            # The specialist sees the shared history up to the open handoff call, plus its task.
            history = state["messages"][:-1]
            task = HumanMessage(f"Task from supervisor: {call['args']['task']}")
            result = agent.invoke({"messages": [*history, task]})
            # Only the final answer goes back. The tool calls and results stay in the subgraph,
            # so the supervisor's prompt does not grow with every search the specialist ran.
            answer = ToolMessage(
                content=result["messages"][-1].text, tool_call_id=call["id"], name=call["name"]
            )
            return Command(goto="supervisor", update={"messages": [answer]})

        return run

    builder = StateGraph(State)
    builder.add_node("supervisor", supervisor)
    builder.add_node("job_scout", specialist_node("job_scout"))
    builder.add_node("salary_analyst", specialist_node("salary_analyst"))
    builder.add_edge(START, "supervisor")
    return builder.compile()


def print_steps(stream) -> None:
    """Prefix each step with the active agent; subgraph steps carry the agent in the namespace."""
    for namespace, update in stream:
        agent = namespace[0].split(":")[0] if namespace else ""
        for node, output in update.items():
            if not isinstance(output, dict):
                continue
            for msg in output.get("messages", []):
                label = f"{agent}/{node}" if agent else node
                print(f"[{label}] {describe(msg)}")


if __name__ == "__main__":
    graph = build()
    if "--graph" in sys.argv:
        print(graph.get_graph().draw_mermaid())
        sys.exit()
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    stream = graph.stream(
        {"messages": [HumanMessage(question)]}, stream_mode="updates", subgraphs=True
    )
    print_steps(stream)
