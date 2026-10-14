"""Lesson 10: create_agent middleware, stream modes and token usage.

Middleware hooks into the agent loop without rewriting it: ToolCallLimitMiddleware caps how
often convert_to_idr may run per thread, SummarizationMiddleware folds old messages into a
summary once the history grows, and ToolTimer (our own) logs every tool call with its
duration. The second half shows three ways to watch the same run: "updates" (one dict per
node), "messages" (text tokens as they arrive) and astream_events (every callback event,
filtered to model tokens and tool results). A UsageMetadataCallbackHandler counts tokens.

Run: uv run lessons/10_middleware_and_streaming.py "your question"
     uv run lessons/10_middleware_and_streaming.py --mode messages
     uv run lessons/10_middleware_and_streaming.py --mode events
     uv run lessons/10_middleware_and_streaming.py --graph
"""

import asyncio
import sys
import time
from collections.abc import Awaitable, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from agentlab.config import SYSTEM, make_model
from agentlab.display import describe, print_updates
from agentlab.tools import TOOLS

DEFAULT_QUESTION = (
    "Lowongan engineer di SG yang nyebutin gaji, rupiahin gaji minimum dari 2 yang pertama."
)
CONVERT_LIMIT = 3


class ToolTimer(AgentMiddleware):
    """Logs each tool call with its duration and keeps the records in `calls`.

    The @wrap_tool_call decorator on a sync function only installs the sync hook, and
    astream_events drives the agent asynchronously, so both hooks are written out here.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def _record(self, request: ToolCallRequest, started: float) -> None:
        entry = {"tool": request.tool_call["name"], "seconds": time.perf_counter() - started}
        self.calls.append(entry)
        print(f"[timer] {entry['tool']} took {entry['seconds'] * 1000:.0f} ms", flush=True)

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable):
        started = time.perf_counter()
        result = handler(request)
        self._record(request, started)
        return result

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Callable[..., Awaitable]):
        started = time.perf_counter()
        result = await handler(request)
        self._record(request, started)
        return result


def build(
    model: BaseChatModel | None = None,
    summary_model: BaseChatModel | None = None,
    timer: ToolTimer | None = None,
):
    model = model or make_model()
    middleware = [
        # thread_limit counts across every run on the same thread_id; "end" stops the run with
        # an explanatory AIMessage instead of letting the model keep trying ("continue").
        ToolCallLimitMiddleware(
            tool_name="convert_to_idr", thread_limit=CONVERT_LIMIT, exit_behavior="end"
        ),
        # Six messages is tiny on purpose so the summary step fires inside a single demo run.
        SummarizationMiddleware(
            summary_model or model, trigger=("messages", 6), keep=("messages", 2)
        ),
        timer or ToolTimer(),
    ]
    return create_agent(model, tools=TOOLS, system_prompt=SYSTEM, middleware=middleware)


def stream_messages(agent, inputs: dict, config: dict) -> None:
    """Print text tokens as they arrive; tool results get one line each."""
    for chunk, metadata in agent.stream(inputs, config, stream_mode="messages"):
        if isinstance(chunk, AIMessageChunk):
            if chunk.text:
                print(chunk.text, end="", flush=True)
            elif chunk.tool_calls:
                print(f"\n[{metadata['langgraph_node']}] {describe(chunk)}")
        elif chunk.type == "tool":
            print(f"[{metadata['langgraph_node']}] {describe(chunk)}")
    print()


async def stream_events(agent, inputs: dict, config: dict) -> None:
    """astream_events emits every callback; keep model tokens and finished tool calls."""
    async for event in agent.astream_events(inputs, config, version="v2"):
        if event["event"] == "on_chat_model_stream":
            print(event["data"]["chunk"].text, end="", flush=True)
        elif event["event"] == "on_tool_end":
            print(f"\n[on_tool_end] {describe(event['data']['output'])}")
    print()


def print_usage(usage: UsageMetadataCallbackHandler) -> None:
    for model_name, totals in usage.usage_metadata.items():
        print(
            f"[usage] {model_name}: {totals['input_tokens']} in, "
            f"{totals['output_tokens']} out, {totals['total_tokens']} total"
        )
    if not usage.usage_metadata:
        print("[usage] the model reported no usage metadata")


if __name__ == "__main__":
    agent = build()
    args = sys.argv[1:]
    if "--graph" in args:
        print(agent.get_graph().draw_mermaid())
        sys.exit()
    mode = args.pop(args.index("--mode") + 1) if "--mode" in args else "updates"
    args = [a for a in args if a != "--mode"]
    question = " ".join(args) or DEFAULT_QUESTION

    usage = UsageMetadataCallbackHandler()
    config = {"callbacks": [usage]}
    inputs = {"messages": [HumanMessage(question)]}
    if mode == "messages":
        stream_messages(agent, inputs, config)
    elif mode == "events":
        asyncio.run(stream_events(agent, inputs, config))
    else:
        print_updates(agent.stream(inputs, config, stream_mode="updates"))
    print_usage(usage)
