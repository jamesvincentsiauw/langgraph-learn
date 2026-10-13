"""Lesson 2: the same agent in one call with LangChain's create_agent.

create_agent builds the model/tools graph from lesson 1 for you. Compare the --graph output
of both files: the nodes and edges match.

Run: uv run 02_create_agent.py "your question"
"""

import sys

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from common import SYSTEM, TOOLS, make_model, print_updates

agent = create_agent(make_model(), tools=TOOLS, system_prompt=SYSTEM)


if __name__ == "__main__":
    if "--graph" in sys.argv:
        print(agent.get_graph().draw_mermaid())
        sys.exit()
    question = " ".join(sys.argv[1:]) or "Ada lowongan engineer di Singapore yang nyebutin gaji? Rupiahin gajinya."
    print_updates(agent.stream({"messages": [HumanMessage(question)]}, stream_mode="updates"))
