"""Lesson 2: the same agent in one call with LangChain's create_agent.

create_agent builds the model/tools graph from lesson 1 for you. Compare the --graph output
of both files: the nodes and edges match.

Run: uv run lessons/02_create_agent.py "your question"
"""

import sys

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from agentlab.config import DEFAULT_QUESTION, SYSTEM, make_model
from agentlab.display import print_updates
from agentlab.tools import TOOLS


def build(model: BaseChatModel | None = None):
    return create_agent(model or make_model(), tools=TOOLS, system_prompt=SYSTEM)


if __name__ == "__main__":
    agent = build()
    if "--graph" in sys.argv:
        print(agent.get_graph().draw_mermaid())
        sys.exit()
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    print_updates(agent.stream({"messages": [HumanMessage(question)]}, stream_mode="updates"))
